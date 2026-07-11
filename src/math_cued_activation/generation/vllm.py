#!/usr/bin/env python3
"""Generate IMO-AnswerBench responses through a local vLLM OpenAI-compatible API."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
import json
from pathlib import Path
import time
import urllib.error
import urllib.request

from tqdm.auto import tqdm

from types import SimpleNamespace

from ..config import PipelineConfig
from .._compat_scripts.run_vibethinker import (
    DEFAULT_MODEL,
    GenerationResult,
    build_imo_answerbench_prompt,
    extract_final_answer_text,
    load_imo_answerbench_rows,
    normalize_short_answer,
    safe_filename,
    save_imo_generation_bundle,
)


DEFAULT_API_URL = "http://127.0.0.1:8000/v1/chat/completions"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate IMO-AnswerBench text via local vLLM.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="vLLM served model id.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL)
    parser.add_argument(
        "--api-key-file",
        type=Path,
        default=Path("api_tokens/.vllm"),
        help="Optional local dummy API key file. Uses 'local-vllm' when absent.",
    )
    parser.add_argument(
        "--server-name",
        default="local",
        help="Label used only in saved metadata/output path, e.g. local, gpu0, gpu1.",
    )
    parser.add_argument("--sample-size", type=int, default=3)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--shuffle", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--problem-id", action="append")
    parser.add_argument("--answer-only", action="store_true")
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=None,
        help="Optional generation cap. Omit for full vLLM server-side context window.",
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument(
        "--context-window",
        type=int,
        default=65536,
        help="Metadata only: total context window served by vLLM.",
    )
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--request-timeout", type=float, default=7200.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep", type=float, default=5.0)
    parser.add_argument(
        "--generated-text-dir",
        type=Path,
        default=Path("outputs/imo-answerbench-responses/WeiboAI__VibeThinker-3B"),
    )
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument(
        "--tokenizer-model",
        default=DEFAULT_MODEL,
        help="HF tokenizer id used only to estimate/save token ids for later inspection.",
    )
    parser.add_argument(
        "--no-token-ids",
        action="store_true",
        help="Do not load a tokenizer or save token ids.",
    )
    return parser.parse_args()


def read_api_key(path: Path) -> str:
    path = path.expanduser()
    if not path.exists():
        return "local-vllm"
    key = path.read_text().strip()
    return key or "local-vllm"


def output_json_path(output_root: Path, problem_id: str) -> Path:
    return output_root.expanduser() / f"{safe_filename(problem_id)}.json"


def request_chat_completion(
    *,
    api_url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    top_p: float,
    max_new_tokens: int | None,
    timeout: float,
) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": temperature,
        "top_p": top_p,
    }
    if max_new_tokens is not None:
        payload["max_tokens"] = max_new_tokens

    request = urllib.request.Request(
        api_url,
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def call_with_retries(args: argparse.Namespace, api_key: str, prompt: str) -> dict:
    last_error = None
    for attempt in range(1, args.retries + 2):
        try:
            return request_chat_completion(
                api_url=args.api_url,
                api_key=api_key,
                model=args.model,
                prompt=prompt,
                temperature=args.temperature,
                top_p=args.top_p,
                max_new_tokens=args.max_new_tokens,
                timeout=args.request_timeout,
            )
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            last_error = RuntimeError(f"HTTP {exc.code}: {body[:1000]}")
        except Exception as exc:
            last_error = exc

        if attempt <= args.retries:
            time.sleep(args.retry_sleep * attempt)

    raise RuntimeError(f"vLLM request failed after retries: {last_error}") from last_error


def load_tokenizer(tokenizer_model: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_model, trust_remote_code=True)


def build_result(
    *,
    tokenizer,
    prompt: str,
    generated: str,
    max_new_tokens: int | None,
    context_window: int,
    usage: dict | None,
) -> GenerationResult:
    if tokenizer is None:
        prompt_token_ids = []
        generated_token_ids = []
        sequence_token_ids = []
        prompt_tokens = int((usage or {}).get("prompt_tokens") or 0)
        generated_tokens = int((usage or {}).get("completion_tokens") or 0)
    else:
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        generated_token_ids = tokenizer.encode(generated, add_special_tokens=False)
        sequence_token_ids = prompt_token_ids + generated_token_ids
        prompt_tokens = len(prompt_token_ids)
        generated_tokens = len(generated_token_ids)

    return GenerationResult(
        text=generated.strip(),
        prompt_tokens=prompt_tokens,
        context_window=context_window,
        max_new_tokens=max_new_tokens if max_new_tokens is not None else -1,
        generated_tokens=generated_tokens,
        hit_token_limit=bool(max_new_tokens is not None and generated_tokens >= max_new_tokens),
        generated_token_ids=generated_token_ids,
        sequence_token_ids=sequence_token_ids,
        activation_capture=None,
    )


def generate_one(args: argparse.Namespace, api_key: str, tokenizer, row: dict, row_number: int) -> tuple[str, Path, bool]:
    prompt = build_imo_answerbench_prompt(row["Problem"], args.answer_only)
    response = call_with_retries(args, api_key, prompt)
    choices = response.get("choices") or []
    if not choices:
        raise RuntimeError(f"vLLM response had no choices for {row['Problem ID']}: {response}")
    generated = choices[0].get("message", {}).get("content") or ""
    usage = response.get("usage")

    result = build_result(
        tokenizer=tokenizer,
        prompt=prompt,
        generated=generated,
        max_new_tokens=args.max_new_tokens,
        context_window=args.context_window,
        usage=usage,
    )
    result = replace(result, context_window=args.context_window)
    dataset_index = row.get("_dataset_index")
    text_path = save_imo_generation_bundle(
        output_dir=args.generated_text_dir,
        row=row,
        row_number=(dataset_index + 1 if isinstance(dataset_index, int) else row_number),
        model_id=args.model,
        prompt=prompt,
        result=result,
        flat=True,
    )

    prediction = extract_final_answer_text(result.text)
    gold = str(row["Short Answer"]).strip()
    is_correct = normalize_short_answer(prediction) == normalize_short_answer(gold)
    return str(row["Problem ID"]), text_path, is_correct


def generate_from_config(config: PipelineConfig, *, force: bool = False) -> None:
    """Generate the configured response set while preserving bundle format."""
    args = SimpleNamespace(
        model=config.model.id,
        api_url=config.vllm.api_url,
        api_key_file=config.vllm.api_key_file,
        server_name=config.vllm.server_name,
        sample_size=config.dataset.sample_size,
        start_index=config.dataset.start_index,
        shuffle=False,
        seed=config.ica.seed,
        problem_id=None,
        answer_only=config.prompt.answer_only,
        max_new_tokens=config.vllm.max_tokens or None,
        temperature=config.vllm.temperature,
        top_p=config.vllm.top_p,
        context_window=config.vllm.max_model_len,
        concurrency=config.vllm.concurrency,
        request_timeout=config.vllm.request_timeout,
        retries=config.vllm.retries,
        retry_sleep=config.vllm.retry_sleep,
        generated_text_dir=config.storage.responses,
        rerun_existing=force,
        tokenizer_model=config.model.tokenizer,
        no_token_ids=False,
    )
    run_generation(args)


def run_generation(args: argparse.Namespace | SimpleNamespace) -> None:
    if args.concurrency < 1:
        raise SystemExit("--concurrency must be at least 1.")
    if args.context_window < 1:
        raise SystemExit("--context-window must be at least 1.")
    api_key = read_api_key(args.api_key_file)

    rows = load_imo_answerbench_rows(
        args.sample_size,
        args.seed,
        args.problem_id,
        args.start_index,
        args.shuffle,
    )
    if not rows:
        raise SystemExit("No IMO-AnswerBench rows matched the request.")

    if not args.rerun_existing:
        before = len(rows)
        rows = [
            row
            for row in rows
            if not output_json_path(
                args.generated_text_dir,
                str(row["Problem ID"]),
            ).exists()
        ]
        skipped = before - len(rows)
        if skipped:
            print(f"Skipping {skipped} already generated problem(s).", flush=True)
    if not rows:
        print("All selected problems already have vLLM generation JSON files.", flush=True)
        return

    tokenizer = None if args.no_token_ids else load_tokenizer(args.tokenizer_model)
    correct = 0
    failures = []
    print(
        f"Generating {len(rows)} IMO-AnswerBench problem(s) via vLLM model {args.model} "
        f"at {args.api_url} with concurrency={args.concurrency}.",
        flush=True,
    )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures = {
            executor.submit(generate_one, args, api_key, tokenizer, row, row_number): row
            for row_number, row in enumerate(rows, start=1)
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="vLLM IMO", unit="problem"):
            row = futures[future]
            try:
                problem_id, text_path, is_correct = future.result()
            except Exception as exc:
                failures.append((row["Problem ID"], exc))
                print(f"FAILED {row['Problem ID']}: {exc}", flush=True)
                continue
            correct += int(is_correct)
            print(f"Saved {problem_id}: {text_path}", flush=True)

    print("=" * 80)
    print(f"Exact-match score: {correct}/{len(rows)}")
    if failures:
        print(f"Failures: {len(failures)}")
        for problem_id, exc in failures[:20]:
            print(f"- {problem_id}: {exc}")
        raise SystemExit(1)


def main() -> None:
    run_generation(parse_args())


if __name__ == "__main__":
    main()
