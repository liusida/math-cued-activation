#!/usr/bin/env python3
"""Quick local inference script for small local Hugging Face chat models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import re
from threading import Event, Thread
import time

from gb10_load_llm import load_model_to_cuda
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    StoppingCriteria,
    StoppingCriteriaList,
    TextIteratorStreamer,
)


DEFAULT_MODEL = "WeiboAI/VibeThinker-3B"
MODEL_ALIASES = {
    "vibethinker": "WeiboAI/VibeThinker-3B",
    "vibethinker-3b": "WeiboAI/VibeThinker-3B",
    "qwen-coder": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen-coder-3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "qwen2.5-coder-3b-instruct": "Qwen/Qwen2.5-Coder-3B-Instruct",
}
DATASET_ID = "MathArena/aime_2025"
IMO_ANSWERBENCH_ID = "OpenEvals/IMO-AnswerBench"
DEFAULT_MAX_NEW_TOKENS = 1024
FALLBACK_AIME_MAX_NEW_TOKENS = 65536


@dataclass
class GenerationResult:
    text: str
    prompt_tokens: int
    context_window: int | None
    max_new_tokens: int
    generated_tokens: int
    hit_token_limit: bool
    generated_token_ids: list[int]
    sequence_token_ids: list[int]
    activation_capture: ActivationCaptureResult | None = None


@dataclass
class ActivationCaptureResult:
    activations: torch.Tensor
    layer: int
    prompt_tokens: int
    captured_tokens: int
    capture_prompt: bool
    dtype: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a prompt through a local HF chat model.")
    parser.add_argument(
        "prompt",
        nargs="?",
        default="Solve: If 3x + 7 = 22, what is x?",
        help="Prompt to send to the model.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=(
            "Hugging Face model id, local path, or alias. Aliases: "
            f"{', '.join(sorted(MODEL_ALIASES))}."
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help=(
            "Generation cap. In benchmark modes, defaults to the remaining model "
            f"context window. Otherwise defaults to {DEFAULT_MAX_NEW_TOKENS}."
        ),
    )
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--no-stream",
        action="store_true",
        help="Wait until generation finishes before printing model output.",
    )
    parser.add_argument(
        "--stream-mode",
        choices=["token", "sentence"],
        default="token",
        help="When streaming, print every token chunk or buffer until sentence boundaries.",
    )
    parser.add_argument(
        "--aime",
        action="store_true",
        help="Run on a sample from MathArena/aime_2025 instead of the positional prompt.",
    )
    parser.add_argument(
        "--imo-answerbench",
        action="store_true",
        help="Run on a sample from Google DeepMind IMO-AnswerBench instead of the positional prompt.",
    )
    parser.add_argument("--sample-size", type=int, default=3, help="Number of benchmark rows to try.")
    parser.add_argument(
        "--start-index",
        type=int,
        default=0,
        help="0-based dataset row index to start from in sequential benchmark mode.",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle benchmark rows before selecting sample-size rows. Default is sequential order.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Shuffle seed used with --shuffle.")
    parser.add_argument(
        "--problem-idx",
        type=int,
        action="append",
        help="Specific AIME problem index to run. Can be passed more than once.",
    )
    parser.add_argument(
        "--problem-id",
        action="append",
        help="Specific IMO-AnswerBench problem id to run. Can be passed more than once.",
    )
    parser.add_argument(
        "--answer-only",
        action="store_true",
        help="For benchmark modes, ask for only the final answer. Much faster, less reliable.",
    )
    parser.add_argument("--device", default="cuda", help='Device placement. Use "cuda", "cuda:N", or "cpu".')
    parser.add_argument(
        "--dtype",
        choices=["auto", "bfloat16", "float16", "float32"],
        default="auto",
        help="Model dtype. auto uses bfloat16 on CUDA when supported, else float16 on CUDA, else float32.",
    )
    parser.add_argument(
        "--capture-activations",
        action="store_true",
        help="Capture layer activations during generation and save them to disk.",
    )
    parser.add_argument(
        "--capture-layer",
        type=int,
        default=32,
        help="0-based decoder layer index to capture. Default: 32.",
    )
    parser.add_argument(
        "--activation-dir",
        type=Path,
        default=Path("~/data/ICA-data/math-cued-activation"),
        help="Root directory for saved activation .pt bundles.",
    )
    parser.add_argument(
        "--activation-dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float16",
        help="CPU dtype used when saving captured activations.",
    )
    parser.add_argument(
        "--capture-prompt-activations",
        action="store_true",
        help="Also save prompt-token activations. By default only generated-token activations are saved.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Show a token-level progress bar during generation. Use --no-progress to disable.",
    )
    parser.add_argument(
        "--progress-interval",
        type=float,
        default=2.0,
        help="Minimum seconds between progress bar updates. Default: 2.0.",
    )
    parser.add_argument(
        "--stream-with-progress",
        action="store_true",
        help="Allow live model text to stream while the progress bar is visible. This can look messy.",
    )
    return parser.parse_args()


def resolve_model_id(model: str) -> str:
    return MODEL_ALIASES.get(model.lower(), model)


def resolve_max_new_tokens(args: argparse.Namespace, context_window: int | None, prompt_tokens: int) -> int:
    if args.max_new_tokens is not None:
        return args.max_new_tokens
    if args.aime:
        if context_window is not None:
            return max(1, context_window - prompt_tokens)
        return FALLBACK_AIME_MAX_NEW_TOKENS
    return DEFAULT_MAX_NEW_TOKENS


def get_context_window(model, tokenizer) -> int | None:
    candidates = [
        getattr(model.config, "max_position_embeddings", None),
        getattr(model.config, "seq_length", None),
        getattr(model.config, "max_sequence_length", None),
        getattr(tokenizer, "model_max_length", None),
    ]
    plausible = [
        int(value)
        for value in candidates
        if isinstance(value, int) and 0 < value < 1_000_000_000
    ]
    return max(plausible) if plausible else None


def choose_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    if name == "float32":
        return torch.float32
    if torch.cuda.is_available():
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.float32


def activation_save_dtype(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float16":
        return torch.float16
    return torch.float32


class ActivationCapture:
    def __init__(
        self,
        model,
        layer: int,
        prompt_tokens: int,
        save_dtype: torch.dtype,
        capture_prompt: bool,
    ) -> None:
        self.model = model
        self.layer = layer
        self.prompt_tokens = prompt_tokens
        self.save_dtype = save_dtype
        self.capture_prompt = capture_prompt
        self.chunks: list[torch.Tensor] = []
        self._handle = None
        self._seen_full_sequence_tokens = 0 if capture_prompt else prompt_tokens
        self._captured_tokens = 0

    def __enter__(self) -> "ActivationCapture":
        module = self.model.get_submodule(f"model.layers.{self.layer}")
        self._handle = module.register_forward_hook(self._hook)
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._handle is not None:
            self._handle.remove()
            self._handle = None

    def _hook(self, module, inputs, output) -> None:
        del module, inputs
        hidden = output[0] if isinstance(output, tuple) else output
        if hidden.ndim != 3 or hidden.shape[0] != 1:
            raise RuntimeError(f"Expected layer output shape [1, seq, hidden], got {tuple(hidden.shape)}")

        seq_len = hidden.shape[1]
        if seq_len > 1:
            start = min(self._seen_full_sequence_tokens, seq_len)
            if start >= seq_len:
                return
            chunk = hidden[:, start:, :]
            self._seen_full_sequence_tokens = seq_len
        else:
            if not self.capture_prompt and self._seen_full_sequence_tokens < self.prompt_tokens:
                return
            chunk = hidden
            self._seen_full_sequence_tokens += 1

        chunk = chunk.squeeze(0).detach().to(device="cpu", dtype=self.save_dtype)
        self._captured_tokens += chunk.shape[0]
        self.chunks.append(chunk)

    def result(self) -> ActivationCaptureResult:
        if self.chunks:
            activations = torch.cat(self.chunks, dim=0)
        else:
            hidden_size = int(getattr(self.model.config, "hidden_size"))
            activations = torch.empty((0, hidden_size), dtype=self.save_dtype)
        return ActivationCaptureResult(
            activations=activations,
            layer=self.layer,
            prompt_tokens=self.prompt_tokens,
            captured_tokens=self._captured_tokens,
            capture_prompt=self.capture_prompt,
            dtype=str(self.save_dtype).replace("torch.", ""),
        )


class GenerationProgress(StoppingCriteria):
    def __init__(
        self,
        prompt_tokens: int,
        max_new_tokens: int,
        enabled: bool,
        desc: str,
        update_interval: float,
    ) -> None:
        self.prompt_tokens = prompt_tokens
        self.max_new_tokens = max_new_tokens
        self.enabled = enabled
        self.desc = desc
        self.update_interval = max(0.0, update_interval)
        self._displayed_generated_tokens = 0
        self._latest_generated_tokens = 0
        self._last_update = 0.0
        self._bar = None

    def __enter__(self) -> "GenerationProgress":
        if self.enabled:
            from tqdm.auto import tqdm

            self._bar = tqdm(
                total=self.max_new_tokens,
                desc=self.desc,
                unit="tok",
                dynamic_ncols=True,
                leave=True,
                mininterval=self.update_interval,
            )
            self._last_update = time.monotonic()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        if self._bar is not None:
            self._flush()
            self._bar.close()
            self._bar = None

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        del scores, kwargs
        if self._bar is None:
            return False
        self._latest_generated_tokens = max(0, input_ids.shape[-1] - self.prompt_tokens)
        now = time.monotonic()
        if (
            self._latest_generated_tokens >= self.max_new_tokens
            or now - self._last_update >= self.update_interval
        ):
            self._flush()
            self._last_update = now
        return False

    def _flush(self) -> None:
        if self._bar is None:
            return
        delta = self._latest_generated_tokens - self._displayed_generated_tokens
        if delta > 0:
            self._bar.update(delta)
            self._displayed_generated_tokens = self._latest_generated_tokens


class BusyProgress:
    def __init__(self, desc: str, enabled: bool = True, update_interval: float = 1.0) -> None:
        self.desc = desc
        self.enabled = enabled
        self.update_interval = max(0.1, update_interval)
        self._stop = Event()
        self._thread: Thread | None = None
        self._bar = None

    def __enter__(self) -> "BusyProgress":
        if not self.enabled:
            return self
        try:
            from tqdm.auto import tqdm
        except Exception:
            print(f"{self.desc} ...", flush=True)
            return self

        self._bar = tqdm(
            total=None,
            desc=self.desc,
            unit="s",
            dynamic_ncols=True,
            leave=True,
            mininterval=self.update_interval,
        )
        self._thread = Thread(target=self._tick, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.update_interval + 0.5)
            self._thread = None
        if self._bar is not None:
            self._bar.close()
            self._bar = None

    def _tick(self) -> None:
        while not self._stop.wait(self.update_interval):
            if self._bar is not None:
                self._bar.update(self.update_interval)


def load_model_and_tokenizer(args: argparse.Namespace):
    model_id = resolve_model_id(args.model)
    device = args.device
    print(f"Loading model: {model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    model_kwargs = {
        "dtype": choose_dtype(args.dtype),
        "low_cpu_mem_usage": True,
    }
    if device == "cpu":
        model = AutoModelForCausalLM.from_pretrained(model_id, **model_kwargs)
    elif isinstance(device, str) and device.startswith("cuda"):
        model = load_model_to_cuda(
            AutoModelForCausalLM,
            model_id,
            device=device,
            **model_kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported --device {device!r}. Use 'cuda', 'cuda:N', or 'cpu'; "
            "GB10 loading supports only direct CPU or CUDA placement."
        )
    return model, tokenizer


def infer_text(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int | None,
    temperature: float,
    top_p: float,
    stream: bool,
    stream_mode: str,
    auto_max_new_tokens: bool,
    capture_layer: int | None = None,
    activation_dtype: str = "float16",
    capture_prompt_activations: bool = False,
    progress: bool = True,
    progress_desc: str = "Generating",
    progress_interval: float = 2.0,
) -> GenerationResult:
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    prompt_tokens = inputs["input_ids"].shape[-1]
    context_window = get_context_window(model, tokenizer)
    if max_new_tokens is None:
        if auto_max_new_tokens:
            if context_window is not None:
                max_new_tokens = max(1, context_window - prompt_tokens)
            else:
                max_new_tokens = FALLBACK_AIME_MAX_NEW_TOKENS
        else:
            max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        top_k=None,
        use_cache=True,
        pad_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    generation_config = GenerationConfig(**generation_kwargs)
    progress_bar = GenerationProgress(
        prompt_tokens=prompt_tokens,
        max_new_tokens=max_new_tokens,
        enabled=progress,
        desc=progress_desc,
        update_interval=progress_interval,
    )
    stopping_criteria = StoppingCriteriaList([progress_bar])

    if stream:
        streamer = TextIteratorStreamer(
            tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )
        result_holder = []
        with progress_bar:
            thread = Thread(
                target=generate_with_streamer,
                kwargs={
                    "model": model,
                    "inputs": inputs,
                    "generation_config": generation_config,
                    "streamer": streamer,
                    "stopping_criteria": stopping_criteria,
                    "result_holder": result_holder,
                },
            )
            thread.start()

            chunks = []
            display_buffer = ""
            for chunk in streamer:
                chunks.append(chunk)
                if stream_mode == "token":
                    print(chunk, end="", flush=True)
                else:
                    display_buffer += chunk
                    display_buffer = flush_complete_sentences(display_buffer)
            thread.join()
        if stream_mode == "sentence" and display_buffer:
            print(display_buffer, end="", flush=True)
        print(flush=True)
        generated_tokens = 0
        generated_token_ids = []
        activation_capture = None
        if result_holder:
            generated_tokens = result_holder[0].shape[-1] - inputs["input_ids"].shape[-1]
            generated_token_ids = result_holder[0][inputs["input_ids"].shape[-1] :].tolist()
            activation_capture = capture_sequence_activations(
                model=model,
                sequence_ids=result_holder[0].to(inputs["input_ids"].device),
                prompt_tokens=prompt_tokens,
                layer=capture_layer,
                activation_dtype=activation_dtype,
                capture_prompt_activations=capture_prompt_activations,
            )
        return GenerationResult(
            text="".join(chunks).strip(),
            prompt_tokens=prompt_tokens,
            context_window=context_window,
            max_new_tokens=max_new_tokens,
            generated_tokens=generated_tokens,
            hit_token_limit=generated_tokens >= max_new_tokens,
            generated_token_ids=generated_token_ids,
            sequence_token_ids=result_holder[0].tolist() if result_holder else [],
            activation_capture=activation_capture,
        )

    with progress_bar:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                stopping_criteria=stopping_criteria,
            )

    generated = outputs[0][inputs["input_ids"].shape[-1] :]
    generated_tokens = generated.shape[-1]
    activation_capture = capture_sequence_activations(
        model=model,
        sequence_ids=outputs[0],
        prompt_tokens=prompt_tokens,
        layer=capture_layer,
        activation_dtype=activation_dtype,
        capture_prompt_activations=capture_prompt_activations,
    )
    return GenerationResult(
        text=tokenizer.decode(generated, skip_special_tokens=True).strip(),
        prompt_tokens=prompt_tokens,
        context_window=context_window,
        max_new_tokens=max_new_tokens,
        generated_tokens=generated_tokens,
        hit_token_limit=generated_tokens >= max_new_tokens,
        generated_token_ids=generated.tolist(),
        sequence_token_ids=outputs[0].tolist(),
        activation_capture=activation_capture,
        )


def trim_generated_token_ids(token_ids: list[int], eos_token_id: int | None, pad_token_id: int | None) -> list[int]:
    trimmed = list(token_ids)
    if eos_token_id is not None and eos_token_id in trimmed:
        trimmed = trimmed[: trimmed.index(eos_token_id) + 1]
    elif pad_token_id is not None and pad_token_id in trimmed:
        trimmed = trimmed[: trimmed.index(pad_token_id)]
    return trimmed


def infer_text_batch(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int | None,
    temperature: float,
    top_p: float,
    auto_max_new_tokens: bool,
    progress: bool = True,
    progress_desc: str = "Generating batch",
    progress_interval: float = 2.0,
) -> list[GenerationResult]:
    if not prompts:
        return []
    if len(prompts) == 1:
        return [
            infer_text(
                model=model,
                tokenizer=tokenizer,
                prompt=prompts[0],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                stream=False,
                stream_mode="token",
                auto_max_new_tokens=auto_max_new_tokens,
                progress=progress,
                progress_desc=progress_desc,
                progress_interval=progress_interval,
            )
        ]

    texts = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for prompt in prompts
    ]
    original_padding_side = tokenizer.padding_side
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    try:
        inputs = tokenizer(texts, return_tensors="pt", padding=True).to(model.device)
    finally:
        tokenizer.padding_side = original_padding_side

    attention_mask = inputs["attention_mask"]
    prompt_token_ids = [
        inputs["input_ids"][index][attention_mask[index].bool()].tolist()
        for index in range(len(prompts))
    ]
    prompt_lengths = [len(ids) for ids in prompt_token_ids]
    max_prompt_tokens = max(prompt_lengths)
    context_window = get_context_window(model, tokenizer)
    if max_new_tokens is None:
        if auto_max_new_tokens:
            if context_window is not None:
                max_new_tokens = max(1, context_window - max_prompt_tokens)
            else:
                max_new_tokens = FALLBACK_AIME_MAX_NEW_TOKENS
        else:
            max_new_tokens = DEFAULT_MAX_NEW_TOKENS

    generation_kwargs = dict(
        max_new_tokens=max_new_tokens,
        do_sample=temperature > 0,
        top_k=None,
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if temperature > 0:
        generation_kwargs["temperature"] = temperature
        generation_kwargs["top_p"] = top_p
    generation_config = GenerationConfig(**generation_kwargs)
    progress_bar = GenerationProgress(
        prompt_tokens=inputs["input_ids"].shape[-1],
        max_new_tokens=max_new_tokens,
        enabled=progress,
        desc=progress_desc,
        update_interval=progress_interval,
    )

    with progress_bar:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                generation_config=generation_config,
                stopping_criteria=StoppingCriteriaList([progress_bar]),
            )

    input_width = inputs["input_ids"].shape[-1]
    results: list[GenerationResult] = []
    for index, prompt_ids in enumerate(prompt_token_ids):
        generated_ids = trim_generated_token_ids(
            outputs[index][input_width:].tolist(),
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )
        sequence_token_ids = prompt_ids + generated_ids
        results.append(
            GenerationResult(
                text=tokenizer.decode(generated_ids, skip_special_tokens=True).strip(),
                prompt_tokens=len(prompt_ids),
                context_window=context_window,
                max_new_tokens=max_new_tokens,
                generated_tokens=len(generated_ids),
                hit_token_limit=len(generated_ids) >= max_new_tokens,
                generated_token_ids=generated_ids,
                sequence_token_ids=sequence_token_ids,
                activation_capture=None,
            )
        )
    return results


def capture_sequence_activations(
    model,
    sequence_ids: torch.Tensor,
    prompt_tokens: int,
    layer: int | None,
    activation_dtype: str,
    capture_prompt_activations: bool,
    progress: bool = True,
    progress_desc: str | None = None,
) -> ActivationCaptureResult | None:
    if layer is None:
        return None
    input_ids = sequence_ids.reshape(1, -1).to(model.device)
    attention_mask = torch.ones_like(input_ids)
    capture = ActivationCapture(
        model=model,
        layer=layer,
        prompt_tokens=prompt_tokens,
        save_dtype=activation_save_dtype(activation_dtype),
        capture_prompt=capture_prompt_activations,
    )
    desc = progress_desc or f"Activation forward layer {layer} ({input_ids.shape[-1]} tok)"
    with capture:
        with BusyProgress(desc=desc, enabled=progress):
            with torch.inference_mode():
                model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    use_cache=False,
                )
    return capture.result()


def generate_with_streamer(
    model,
    inputs,
    generation_config: GenerationConfig,
    streamer,
    stopping_criteria: StoppingCriteriaList,
    result_holder: list,
) -> None:
    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            generation_config=generation_config,
            streamer=streamer,
            stopping_criteria=stopping_criteria,
        )
    result_holder.append(outputs[0].detach().cpu())


def flush_complete_sentences(buffer: str) -> str:
    """Print complete sentence-ish spans and return the unprinted suffix."""
    last_boundary = -1
    for match in re.finditer(r"(?:</think>|\n\n+|[.!?。！？]\s+)", buffer):
        last_boundary = match.end()

    if last_boundary == -1:
        return buffer

    print(buffer[:last_boundary], end="", flush=True)
    return buffer[last_boundary:]


def build_aime_prompt(problem: str, answer_only: bool) -> str:
    if answer_only:
        return (
            "Solve the following AIME problem. Return only the final integer answer, "
            "with no explanation.\n\n"
            f"{problem}"
        )

    return (
        "Solve the following AIME problem. The answer is an integer from 0 to 999. "
        "Keep your reasoning concise, then end with a line exactly like: "
        "Final answer: <integer>\n\n"
        f"{problem}"
    )


def build_imo_answerbench_prompt(problem: str, answer_only: bool) -> str:
    if answer_only:
        return (
            "Solve the following Olympiad problem. Return only the final short answer, "
            "with no explanation.\n\n"
            f"{problem}"
        )

    return (
        "Solve the following Olympiad problem. The answer is a verifiable short answer. "
        "Keep your reasoning concise, then end with a line exactly like: "
        "Final answer: <answer>\n\n"
        f"{problem}"
    )


def extract_final_integer(text: str) -> int | None:
    boxed_matches = re.findall(r"\\boxed\{([0-9]{1,3})\}", text)
    if boxed_matches:
        return int(boxed_matches[-1])

    final_matches = re.findall(
        r"(?:final answer|answer)\s*[:=]?\s*\$?([0-9]{1,3})\b",
        text,
        flags=re.IGNORECASE,
    )
    if final_matches:
        return int(final_matches[-1])

    stripped = text.strip()
    if re.fullmatch(r"\$?[0-9]{1,3}\$?\.?", stripped):
        return int(re.search(r"[0-9]{1,3}", stripped).group(0))

    return None


def extract_final_answer_text(text: str) -> str | None:
    final_matches = re.findall(
        r"(?:final answer|answer)\s*[:=]\s*(.+?)(?:\n|$)",
        text,
        flags=re.IGNORECASE,
    )
    if final_matches:
        return final_matches[-1].strip()

    boxed_matches = re.findall(r"\\boxed\{(.+?)\}", text)
    if boxed_matches:
        return boxed_matches[-1].strip()

    stripped = text.strip()
    if "\n" not in stripped and stripped:
        return stripped

    return None


def normalize_short_answer(answer: str | None) -> str | None:
    if answer is None:
        return None
    answer = answer.strip()
    answer = answer.rstrip(".")
    answer = answer.replace("$", "")
    answer = re.sub(r"\\left|\\right", "", answer)
    answer = re.sub(r"\s+", "", answer)
    return answer.lower()


def safe_filename(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value.strip("_") or "unknown"


def model_slug(model_id: str) -> str:
    return safe_filename(model_id.replace("/", "__"))


def activation_output_dir(
    root: Path,
    dataset_id: str,
    model_id: str,
    layer: int,
) -> Path:
    return (
        root.expanduser()
        / safe_filename(dataset_id.replace("/", "__"))
        / model_slug(model_id)
        / f"layer_{layer:02d}"
    )


def generated_text_output_dir(
    root: Path,
    dataset_id: str,
    model_id: str,
) -> Path:
    return (
        root.expanduser()
        / safe_filename(dataset_id.replace("/", "__"))
        / model_slug(model_id)
    )


def build_imo_generation_metadata(
    row: dict,
    row_number: int,
    model_id: str,
    prompt: str,
    result: GenerationResult,
    dataset_id: str = IMO_ANSWERBENCH_ID,
) -> dict:
    return {
        "schema": "imo_answerbench_generation_v1",
        "tokens": {
            "sequence_token_ids": result.sequence_token_ids,
            "generated_token_ids": result.generated_token_ids,
        },
        "text": {
            "prompt": prompt,
            "generated": result.text,
        },
        "problem": {
            "dataset": dataset_id,
            "row_number": row_number,
            "dataset_index": row.get("_dataset_index"),
            "problem_id": row["Problem ID"],
            "category": row["Category"],
            "subcategory": row["Subcategory"],
            "source": row["Source"],
            "problem": row["Problem"],
            "short_answer": row["Short Answer"],
        },
        "generation": {
            "model": model_id,
            "context_window": result.context_window,
            "max_new_tokens": result.max_new_tokens,
            "prompt_tokens": result.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "hit_token_limit": result.hit_token_limit,
        },
    }


def save_imo_generation_bundle(
    output_dir: Path,
    row: dict,
    row_number: int,
    model_id: str,
    prompt: str,
    result: GenerationResult,
    flat: bool = False,
    dataset_id: str = IMO_ANSWERBENCH_ID,
) -> Path:
    if flat:
        output_dir = output_dir.expanduser()
        storage_root = output_dir.parent
    else:
        output_dir = generated_text_output_dir(
            root=output_dir,
            dataset_id=dataset_id,
            model_id=model_id,
        )
        storage_root = output_dir.parents[1]
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_id = safe_filename(str(row["Problem ID"]))
    text_path = output_dir / f"{problem_id}.txt"
    json_path = output_dir / f"{problem_id}.json"
    metadata = build_imo_generation_metadata(
        row=row,
        row_number=row_number,
        model_id=model_id,
        prompt=prompt,
        result=result,
        dataset_id=dataset_id,
    )
    metadata["storage"] = {
        "root": str(storage_root),
        "text_relative_path": str(text_path.relative_to(storage_root)),
        "json_relative_path": str(json_path.relative_to(storage_root)),
    }
    json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
    text_path.write_text(format_imo_generation_text(metadata))
    return text_path


def load_imo_generation_bundle(root: Path, row: dict, model_id: str) -> tuple[dict, Path]:
    root = root.expanduser()
    problem_id = safe_filename(str(row["Problem ID"]))
    flat_json_path = root / f"{problem_id}.json"
    if flat_json_path.exists():
        return json.loads(flat_json_path.read_text()), flat_json_path

    output_dir = generated_text_output_dir(
        root=root,
        dataset_id=IMO_ANSWERBENCH_ID,
        model_id=model_id,
    )
    json_path = output_dir / f"{problem_id}.json"
    if not json_path.exists():
        raise FileNotFoundError(
            f"Missing saved generation metadata for {row['Problem ID']}: {json_path}"
        )
    return json.loads(json_path.read_text()), json_path


def result_from_saved_generation(
    model,
    tokenizer,
    metadata: dict,
    capture_layer: int,
    activation_dtype: str,
    capture_prompt_activations: bool,
) -> GenerationResult:
    tokens = metadata["tokens"]
    generation = metadata["generation"]
    sequence_token_ids = [int(token_id) for token_id in tokens["sequence_token_ids"]]
    generated_token_ids = [int(token_id) for token_id in tokens["generated_token_ids"]]
    prompt_tokens = int(generation["prompt_tokens"])
    sequence_ids = torch.tensor(sequence_token_ids, dtype=torch.long)
    activation_capture = capture_sequence_activations(
        model=model,
        sequence_ids=sequence_ids,
        prompt_tokens=prompt_tokens,
        layer=capture_layer,
        activation_dtype=activation_dtype,
        capture_prompt_activations=capture_prompt_activations,
    )
    return GenerationResult(
        text=metadata["text"]["generated"],
        prompt_tokens=prompt_tokens,
        context_window=get_context_window(model, tokenizer),
        max_new_tokens=int(generation["max_new_tokens"]),
        generated_tokens=len(generated_token_ids),
        hit_token_limit=bool(generation["hit_token_limit"]),
        generated_token_ids=generated_token_ids,
        sequence_token_ids=sequence_token_ids,
        activation_capture=activation_capture,
    )


def save_imo_activation_bundle(
    output_dir: Path,
    row: dict,
    row_number: int,
    model_id: str,
    prompt: str,
    result: GenerationResult,
    write_sidecars: bool = True,
    dataset_id: str = IMO_ANSWERBENCH_ID,
) -> Path | None:
    capture = result.activation_capture
    if capture is None:
        return None

    output_dir = activation_output_dir(
        root=output_dir,
        dataset_id=dataset_id,
        model_id=model_id,
        layer=capture.layer,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    problem_id = safe_filename(str(row["Problem ID"]))
    path = output_dir / f"{problem_id}.pt"
    text_path = path.with_suffix(".txt")
    json_path = path.with_suffix(".json")
    generated_token_ids = result.generated_token_ids
    if capture.capture_prompt:
        captured_token_ids = result.sequence_token_ids[: capture.captured_tokens]
    else:
        captured_token_ids = generated_token_ids[: capture.captured_tokens]
    prompt_token_ids = result.sequence_token_ids[: capture.prompt_tokens]
    if capture.capture_prompt:
        prompt_activations = capture.activations[: capture.prompt_tokens]
        generated_start = capture.prompt_tokens
        generated_stop = min(generated_start + result.generated_tokens, capture.activations.shape[0])
        generated_activations = capture.activations[generated_start:generated_stop]
        prompt_captured_tokens = int(prompt_activations.shape[0])
        generated_captured_tokens = int(generated_activations.shape[0])
        capture_segments = {
            "prompt": {
                "token_start": 0,
                "token_end": prompt_captured_tokens,
                "captured_tokens": prompt_captured_tokens,
                "activation_key": "prompt_activations",
                "token_ids_key": "prompt_token_ids",
                "notes": "Prompt segment includes chat-template/system/user/generation-prompt tokens.",
            },
            "generated": {
                "token_start": generated_start,
                "token_end": generated_start + generated_captured_tokens,
                "captured_tokens": generated_captured_tokens,
                "activation_key": "generated_activations",
                "token_ids_key": "generated_token_ids",
                "notes": (
                    "Generated-token rows were captured from a full-sequence forward pass "
                    "over prompt/chat-template plus generated tokens, so prompt context is present."
                ),
            },
        }
    else:
        prompt_activations = torch.empty((0, capture.activations.shape[-1]), dtype=capture.activations.dtype)
        generated_activations = capture.activations
        prompt_captured_tokens = 0
        generated_captured_tokens = capture.captured_tokens
        capture_segments = {
            "generated": {
                "token_start": capture.prompt_tokens,
                "token_end": capture.prompt_tokens + generated_captured_tokens,
                "captured_tokens": generated_captured_tokens,
                "activation_key": "generated_activations",
                "token_ids_key": "generated_token_ids",
                "notes": (
                    "Generated-token rows were captured from a full-sequence forward pass "
                    "over prompt/chat-template plus generated tokens, so prompt context is present."
                ),
            }
        }

    metadata = {
        "activation_name": f"model.layers.{capture.layer}",
        "activation_shape": tuple(capture.activations.shape),
        "prompt_activation_shape": tuple(prompt_activations.shape),
        "generated_activation_shape": tuple(generated_activations.shape),
        "capture": {
            "layer": capture.layer,
            "layer_indexing": "0-based Hugging Face decoder layer index",
            "capture_prompt": capture.capture_prompt,
            "prompt_tokens": capture.prompt_tokens,
            "generated_tokens": result.generated_tokens,
            "captured_tokens": capture.captured_tokens,
            "prompt_captured_tokens": prompt_captured_tokens,
            "generated_captured_tokens": generated_captured_tokens,
            "activation_dtype": capture.dtype,
            "capture_strategy": "post_generation_full_forward",
            "alignment": (
                "Rows in activations align to captured_token_ids. prompt_activations align "
                "to prompt_token_ids, and generated_activations align to generated_token_ids. "
                "Generated activations are computed in the full prompt/chat-template context."
            ),
            "segments": capture_segments,
        },
        "tokens": {
            "sequence_token_ids": result.sequence_token_ids,
            "prompt_token_ids": prompt_token_ids,
            "generated_token_ids": generated_token_ids,
            "captured_token_ids": captured_token_ids,
        },
        "text": {
            "prompt": prompt,
            "generated": result.text,
        },
        "problem": {
            "dataset": dataset_id,
            "row_number": row_number,
            "dataset_index": row.get("_dataset_index"),
            "problem_id": row["Problem ID"],
            "category": row["Category"],
            "subcategory": row["Subcategory"],
            "source": row["Source"],
            "problem": row["Problem"],
            "short_answer": row["Short Answer"],
        },
        "generation": {
            "model": model_id,
            "context_window": result.context_window,
            "max_new_tokens": result.max_new_tokens,
            "hit_token_limit": result.hit_token_limit,
        },
        "storage": {
            "root": str(output_dir.parents[2]),
            "relative_path": str(path.relative_to(output_dir.parents[2])),
        },
    }

    torch.save(
        {
            "activations": capture.activations,
            "prompt_activations": prompt_activations,
            "generated_activations": generated_activations,
            **metadata,
        },
        path,
    )
    if write_sidecars:
        metadata["storage"]["text_relative_path"] = str(text_path.relative_to(output_dir.parents[2]))
        metadata["storage"]["json_relative_path"] = str(json_path.relative_to(output_dir.parents[2]))
        json_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + "\n")
        text_path.write_text(format_imo_text_sidecar(metadata))
    return path


def format_imo_generation_text(metadata: dict) -> str:
    problem = metadata["problem"]
    text = metadata["text"]
    generation = metadata["generation"]
    return (
        f"Dataset: {problem['dataset']}\n"
        f"Problem ID: {problem['problem_id']}\n"
        f"Category: {problem['category']} / {problem['subcategory']}\n"
        f"Source: {problem['source']}\n"
        f"Model: {generation['model']}\n"
        f"Prompt tokens: {generation['prompt_tokens']}\n"
        f"Generated tokens: {generation['generated_tokens']}\n"
        f"Gold short answer: {problem['short_answer']}\n"
        "\n"
        "=== Problem ===\n"
        f"{problem['problem']}\n"
        "\n"
        "=== Prompt ===\n"
        f"{text['prompt']}\n"
        "\n"
        "=== Generated ===\n"
        f"{text['generated']}\n"
    )


def format_imo_text_sidecar(metadata: dict) -> str:
    problem = metadata["problem"]
    text = metadata["text"]
    capture = metadata["capture"]
    generation = metadata["generation"]
    return (
        f"Dataset: {problem['dataset']}\n"
        f"Problem ID: {problem['problem_id']}\n"
        f"Category: {problem['category']} / {problem['subcategory']}\n"
        f"Source: {problem['source']}\n"
        f"Model: {generation['model']}\n"
        f"Activation: {metadata['activation_name']} shape={metadata['activation_shape']}\n"
        f"Prompt activations: shape={metadata.get('prompt_activation_shape', 'unknown')}\n"
        f"Generated activations: shape={metadata.get('generated_activation_shape', 'unknown')}\n"
        f"Generated tokens: {capture['generated_tokens']}\n"
        f"Captured tokens: {capture['captured_tokens']}\n"
        f"Gold short answer: {problem['short_answer']}\n"
        "\n"
        "=== Problem ===\n"
        f"{problem['problem']}\n"
        "\n"
        "=== Prompt ===\n"
        f"{text['prompt']}\n"
        "\n"
        "=== Generated ===\n"
        f"{text['generated']}\n"
    )


def with_dataset_index(row: dict, index: int) -> dict:
    row = dict(row)
    row["_dataset_index"] = index
    return row


def load_aime_rows(
    sample_size: int,
    seed: int,
    problem_indexes: list[int] | None,
    start_index: int,
    shuffle: bool,
) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(DATASET_ID, split="train")
    if problem_indexes:
        wanted = set(problem_indexes)
        return [with_dataset_index(row, index) for index, row in enumerate(dataset) if row["problem_idx"] in wanted]

    if shuffle:
        sample_size = min(sample_size, len(dataset))
        return [with_dataset_index(row, index) for index, row in enumerate(dataset.shuffle(seed=seed).select(range(sample_size)))]

    start_index = max(0, start_index)
    stop_index = min(start_index + sample_size, len(dataset))
    return [with_dataset_index(dataset[index], index) for index in range(start_index, stop_index)]


def load_imo_answerbench_rows(
    sample_size: int,
    seed: int,
    problem_ids: list[str] | None,
    start_index: int,
    shuffle: bool,
) -> list[dict]:
    from datasets import load_dataset

    dataset = load_dataset(IMO_ANSWERBENCH_ID, split="train")
    if problem_ids:
        wanted = set(problem_ids)
        return [with_dataset_index(row, index) for index, row in enumerate(dataset) if row["Problem ID"] in wanted]

    if shuffle:
        sample_size = min(sample_size, len(dataset))
        return [with_dataset_index(row, index) for index, row in enumerate(dataset.shuffle(seed=seed).select(range(sample_size)))]

    start_index = max(0, start_index)
    stop_index = min(start_index + sample_size, len(dataset))
    return [with_dataset_index(dataset[index], index) for index in range(start_index, stop_index)]


def should_stream_output(args: argparse.Namespace) -> bool:
    return not args.no_stream and (args.stream_with_progress or not args.progress)


def run_aime(args: argparse.Namespace) -> None:
    rows = load_aime_rows(args.sample_size, args.seed, args.problem_idx, args.start_index, args.shuffle)
    if not rows:
        raise SystemExit("No AIME rows matched the requested problem index.")

    model = None
    tokenizer = None
    correct = 0

    for row_number, row in enumerate(rows, start=1):
        print("=" * 80, flush=True)
        dataset_index = row.get("_dataset_index")
        print(
            f"Sample {row_number}/{len(rows)} | dataset row {dataset_index} | AIME problem {row['problem_idx']}",
            flush=True,
        )
        print(f"Type: {', '.join(row['problem_type'])}", flush=True)
        print("-" * 80, flush=True)
        print("Problem:", flush=True)
        print(row["problem"], flush=True)
        print("-" * 80, flush=True)

        if model is None or tokenizer is None:
            model, tokenizer = load_model_and_tokenizer(args)

        print("Model response:", flush=True)

        prompt = build_aime_prompt(row["problem"], args.answer_only)
        stream_output = should_stream_output(args)
        result = infer_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=stream_output,
            stream_mode=args.stream_mode,
            auto_max_new_tokens=True,
            progress=args.progress,
            progress_desc=f"AIME {row_number}/{len(rows)}",
            progress_interval=args.progress_interval,
        )
        prediction = extract_final_integer(result.text)
        gold = int(row["answer"])
        is_correct = prediction == gold
        correct += int(is_correct)

        if not stream_output:
            print(result.text)
        print("-" * 80)
        print("Result:")
        print(f"Prompt tokens: {result.prompt_tokens}")
        print(f"Context window: {result.context_window or 'unknown'}")
        print(f"Max new tokens: {result.max_new_tokens}")
        print(f"Generated tokens: {result.generated_tokens}")
        print(f"Hit token limit: {result.hit_token_limit}")
        print(f"Gold answer: {gold}")
        print(f"Parsed prediction: {prediction}")
        print(f"Correct: {is_correct}")

    print("=" * 80)
    print("Final summary:")
    print(f"Score: {correct}/{len(rows)}")


def run_imo_answerbench(args: argparse.Namespace) -> None:
    rows = load_imo_answerbench_rows(
        args.sample_size,
        args.seed,
        args.problem_id,
        args.start_index,
        args.shuffle,
    )
    if not rows:
        raise SystemExit("No IMO-AnswerBench rows matched the requested problem id.")

    model = None
    tokenizer = None
    correct = 0
    model_id = resolve_model_id(args.model)

    for row_number, row in enumerate(rows, start=1):
        print("=" * 80, flush=True)
        dataset_index = row.get("_dataset_index")
        print(
            f"Sample {row_number}/{len(rows)} | dataset row {dataset_index} | IMO-AnswerBench {row['Problem ID']}",
            flush=True,
        )
        print(f"Category: {row['Category']} | Subcategory: {row['Subcategory']}", flush=True)
        print(f"Source: {row['Source']}", flush=True)
        print("-" * 80, flush=True)
        print("Problem:", flush=True)
        print(row["Problem"], flush=True)
        print("-" * 80, flush=True)

        if model is None or tokenizer is None:
            model, tokenizer = load_model_and_tokenizer(args)

        print("Model response:", flush=True)

        prompt = build_imo_answerbench_prompt(row["Problem"], args.answer_only)
        capture_layer = args.capture_layer if args.capture_activations else None
        stream_output = should_stream_output(args)
        result = infer_text(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            stream=stream_output,
            stream_mode=args.stream_mode,
            auto_max_new_tokens=True,
            capture_layer=capture_layer,
            activation_dtype=args.activation_dtype,
            capture_prompt_activations=args.capture_prompt_activations,
            progress=args.progress,
            progress_desc=f"IMO {row_number}/{len(rows)}",
            progress_interval=args.progress_interval,
        )
        activation_path = save_imo_activation_bundle(
            output_dir=args.activation_dir,
            row=row,
            row_number=(dataset_index + 1 if isinstance(dataset_index, int) else row_number),
            model_id=model_id,
            prompt=prompt,
            result=result,
        )
        prediction = extract_final_answer_text(result.text)
        gold = str(row["Short Answer"]).strip()
        is_correct = normalize_short_answer(prediction) == normalize_short_answer(gold)
        correct += int(is_correct)

        if not stream_output:
            print(result.text)
        print("-" * 80)
        print("Result:")
        print(f"Prompt tokens: {result.prompt_tokens}")
        print(f"Context window: {result.context_window or 'unknown'}")
        print(f"Max new tokens: {result.max_new_tokens}")
        print(f"Generated tokens: {result.generated_tokens}")
        print(f"Hit token limit: {result.hit_token_limit}")
        print(f"Gold short answer: {gold}")
        print(f"Parsed prediction: {prediction}")
        print(f"Normalized exact match: {is_correct}")
        if activation_path is not None:
            print(f"Saved activations: {activation_path}")
            print(f"Activation shape: {tuple(result.activation_capture.activations.shape)}")

    print("=" * 80)
    print("Final summary:")
    print(f"Exact-match score: {correct}/{len(rows)}")


def main() -> None:
    args = parse_args()
    if args.aime and args.imo_answerbench:
        raise SystemExit("Choose only one benchmark mode: --aime or --imo-answerbench.")
    if args.aime:
        run_aime(args)
        return
    if args.imo_answerbench:
        run_imo_answerbench(args)
        return

    print("=" * 80, flush=True)
    print("Prompt:", flush=True)
    print(args.prompt, flush=True)
    print("-" * 80, flush=True)

    model, tokenizer = load_model_and_tokenizer(args)
    print("Model response:", flush=True)
    stream_output = should_stream_output(args)
    result = infer_text(
        model=model,
        tokenizer=tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        stream=stream_output,
        stream_mode=args.stream_mode,
        auto_max_new_tokens=False,
        progress=args.progress,
        progress_desc="Generating",
        progress_interval=args.progress_interval,
    )
    if not stream_output:
        print(result.text)
    print("-" * 80)
    print("Result:")
    print(f"Prompt tokens: {result.prompt_tokens}")
    print(f"Context window: {result.context_window or 'unknown'}")
    print(f"Max new tokens: {result.max_new_tokens}")
    print(f"Generated tokens: {result.generated_tokens}")
    print(f"Hit token limit: {result.hit_token_limit}")


if __name__ == "__main__":
    main()
