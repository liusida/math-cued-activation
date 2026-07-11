from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm.auto import tqdm

from ..paths import V9_ROOT
from .evidence import COMPACT_EVIDENCE_FILENAME, DEFAULT_OUTPUT_ROOT, LEGACY_COMPACT_EVIDENCE_FILENAME


DEFAULT_ANNOTATION_ROOT = V9_ROOT / "results" / "auto_annotation" / "annotations"
DEFAULT_ICL_EXAMPLE_ROOT = V9_ROOT / "results" / "auto_annotation" / "icl_examples"
DEFAULT_MI_TOKEN_PATH = V9_ROOT / "API_tokens" / ".mi_token"
DEFAULT_OPENAI_TOKEN_PATH = V9_ROOT / "API_tokens" / ".openai_api_token"
DEFAULT_TINKER_TOKEN_PATH = V9_ROOT / "API_tokens" / ".tinker_api_token"
DEFAULT_DEEPSEEK_TOKEN_PATH = V9_ROOT / "API_tokens" / ".deepseek_api_token"
DEFAULT_MI_BASE_URL = "https://api.xiaomimimo.com/v1"
DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_MODEL = "xiaomi/mimo-v2.5"
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_TINKER_SDK_MODEL = "Qwen/Qwen3.5-397B-A17B"
DEFAULT_DEEPSEEK_MODEL = "deepseek-v4-flash"
DEFAULT_OLLAMA_MODEL = "qwen3.6:35b-a3b-q4_K_M"
TOKENIZER_MODEL_BY_FEATURE_PREFIX = {
    "gpt2/": "openai-community/gpt2",
    "gpt2_tok": "openai-community/gpt2",
    "gemma2_2b/": "google/gemma-2-2b",
    "gemma2_2b_tok": "google/gemma-2-2b",
    "qwen3_5_2b_base/": "Qwen/Qwen3.5-2B-Base",
    "qwen3_5_2b_base_tok": "Qwen/Qwen3.5-2B-Base",
}
MODEL_ALIASES = {
    "mi": "xiaomi/mimo-v2.5",
    "mimo-v2.5": "xiaomi/mimo-v2.5",
    "mi_pro": "xiaomi/mimo-v2.5-pro",
    "mimo-v2.5-pro": "xiaomi/mimo-v2.5-pro",
    "openai": DEFAULT_OPENAI_MODEL,
    "gpt4o-mini": DEFAULT_OPENAI_MODEL,
    "gpt-4o-mini": DEFAULT_OPENAI_MODEL,
    "tinker-sdk": DEFAULT_TINKER_SDK_MODEL,
    "deepseek": DEFAULT_DEEPSEEK_MODEL,
    "deepseek-flash": "deepseek-v4-flash",
    "deepseek-pro": "deepseek-v4-pro",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
    "local-qwen": DEFAULT_OLLAMA_MODEL,
}
MODEL_LABEL_ALIASES = {
    "xiaomi/mimo-v2.5": "mi",
    "xiaomi/mimo-v2.5-pro": "mi_pro",
    "mimo-v2.5": "mi",
    "mimo-v2.5-pro": "mi_pro",
    DEFAULT_OPENAI_MODEL: "gpt-4o-mini",
    "openai": "gpt-4o-mini",
    "gpt-5.5-2026-04-23": "gpt-5.5",
    DEFAULT_TINKER_SDK_MODEL: "tinker-qwen397b",
    "tinker-sdk": "tinker-qwen397b",
    "deepseek": "deepseek-v4-flash",
    "deepseek-v4-flash": "deepseek-v4-flash",
    "deepseek-v4-pro": "deepseek-v4-pro",
    DEFAULT_OLLAMA_MODEL: "local-qwen",
    "local-qwen": "local-qwen",
}


ANNOTATION_SCHEMA = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief evidence from the examples and any uncertainty.",
        },
        "label": {
            "type": "string",
            "description": "Compact Title Case noun phrase, usually 1-4 words, following label_naming_policy.",
        },
        "simple_label": {
            "type": "string",
            "description": "Plain easy label for non-native English speakers.",
        },
        "description": {
            "type": "string",
            "description": "One simple sentence describing what activates the feature.",
        },
        "confidence": {
            "type": "string",
            "enum": ["high", "medium", "low"],
            "description": "Confidence in the annotation.",
        },
        "test_cases": {
            "type": "array",
            "description": "Eight compact prompts organized as two contrast ladders from evidence-like positives to expected negatives.",
            "minItems": 8,
            "maxItems": 8,
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "text": {
                        "type": "string",
                        "description": (
                            "Compact prompt to run through the LLM and feature scorer. Put the suspected activating token after the "
                            "left context being tested; later right context is only for readability."
                        ),
                    },
                    "expected": {
                        "type": "string",
                        "enum": ["activate", "not_activate", "ambiguous"],
                        "description": "Expected activation outcome if the label is correct.",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Why this case tests the proposed label.",
                    },
                },
                "required": ["text", "expected", "reason"],
            },
        },
    },
    "required": ["reasoning", "label", "simple_label", "description", "confidence", "test_cases"],
}


ANNOTATION_RESPONSE_FORMAT = {
    "reasoning": "one concise rationale sentence",
    "label": "short Title Case label",
    "simple_label": "plain English label",
    "description": "one sentence describing what activates",
    "confidence": "high | medium | low",
    "test_cases": [
        {
            "text": "test prompt as pure text",
            "expected": "activate | not_activate | ambiguous",
            "reason": "what boundary this tests",
        }
    ],
}


SYSTEM_PROMPT = """You are an expert annotator for neural activation features.

Infer concise, evidence-grounded labels for neural features from activation examples and test results.
Follow the instruction field in each user message; treat other fields as supporting data.
Return only valid JSON when asked.
"""
REPAIR_SYSTEM_PROMPT = """You repair malformed neural feature annotation output.

Return only valid JSON with exactly these keys: reasoning, label, simple_label, description, confidence, test_cases.
Use the malformed answer and original request to preserve the intended label, but make the JSON syntactically valid and schema-compliant.
Keep reasoning to one concise rationale sentence, under 40 words. Do not write step-by-step analysis.
Return exactly 8 test_cases. Each test case must have only text, expected, reason.
Whole-text scoring will find the strongest activating token.
"""

INITIAL_ANNOTATION_INSTRUCTION = {
    "task": "Create the initial annotation from the target compact evidence.",
    "output": "Return JSON with keys: reasoning, label, simple_label, description, confidence, test_cases.",
    "label_style": "Follow label_naming_policy from the style guide. Use simple_label for plain English details.",
    "description_style": "Make description one simple sentence describing what activates the feature.",
    "reasoning_style": "Give only one concise rationale sentence, under 40 words. Do not write step-by-step analysis.",
    "confidence": "Set confidence to high, medium, or low.",
    "evidence_rules": [
        "Infer the broadest simple textual pattern that explains most examples, not details from one document.",
        "Do not label only the shortest effective-receptive-field substring if semantic_examples or surrounding context show a clearer repeated structure.",
        "Use erf_examples, score changes, and sudden_jump rows to decide whether the target token itself or earlier left context matters.",
        "For a marked token in a causal language model, activation is caused by the token and its left context, not by later right context.",
        "right_context_for_readability_not_causal_evidence is future text and only a readability hint; do not use it as causal evidence.",
        "If the evidence says examples share the same position, consider a positional feature.",
    ],
    "test_case_strategy": (
        "In test_cases, propose exactly 8 compact prompts for a follow-up experiment. "
        "Each test case must have text, expected, and reason. "
        "The follow-up experiment scores the whole text and finds the strongest activating token. "
        "Use expected='activate' for positive cases, 'not_activate' for near-miss negatives, and 'ambiguous' when uncertain. "
        "Use test_cases to probe hypothesis boundaries, not to confirm one narrow pattern. "
        "Cover several plausible boundaries from the evidence, including positives, near-misses, and clear negatives. "
        "Avoid single-word prompts. Unless the hypothesis is about beginning-of-text or token position, put the suspected activating token later in a short natural context. "
        "For causal language models, the decisive evidence should be at or before the tested token, not after it."
    ),
}

ANNOTATION_STYLE_GUIDE = {
    "purpose": "Calibrate wording and confidence only; use the target evidence as the source of truth.",
    "label_naming_policy": [
        "label is a compact Title Case noun phrase.",
        "Prefer 1-4 words.",
        "Do not quote ordinary literal tokens in label; use simple_label for quotes when helpful.",
        "Prefer category-first names: Word At, Letter C, Legal Citation Period, Contraction Apostrophe.",
        "Avoid 'or', slashes, lowercase starts, and sentence-like labels.",
        "Put variants and caveats in description, not label.",
    ],
    "label_rewrites": [
        {
            "avoid": "Word 'at'",
            "prefer": "Word At",
        },
        {
            "avoid": "Apostrophe in contractions",
            "prefer": "Contraction Apostrophe",
        },
        {
            "avoid": "Letter 'c' or 'C'",
            "prefer": "Letter C",
        },
        {
            "avoid": "Verb 'get' or 'getting'",
            "prefer": "Get Verb Forms",
        },
        {
            "avoid": "Closing Paren in Legal Clause",
            "prefer": "Legal Closing Paren",
        },
    ],
    "label_style": [
        {
            "label": "Word Same",
            "simple_label": "Word 'same'",
            "description": "Activates on the standalone word 'same'.",
        },
        {
            "label": "Legal Citation V",
            "simple_label": "Legal case 'v'",
            "description": "Activates on the 'v' in legal case names such as 'Smith v. Jones'.",
        },
        {
            "label": "Count Percentage",
            "simple_label": "Number in percent parentheses",
            "description": "Activates on percentage numbers in count-plus-percentage phrases like '10 (32%)'.",
        },
        {
            "label": "Document Start",
            "simple_label": "First token in document",
            "description": "Activates on the first meaningful token of a document.",
        },
    ],
    "confidence_calibration": [
        {
            "confidence": "high",
            "use_when": "One simple pattern explains all examples, and ERF/test evidence agrees.",
        },
        {
            "confidence": "medium",
            "use_when": "A likely pattern exists, but examples are broad, tests partly fail, or context requirements are uncertain.",
        },
        {
            "confidence": "low",
            "use_when": "Examples are mixed/noisy, the pattern is weak, or the label is mostly a guess.",
        },
    ],
    "reasoning_style": "One concise sentence under 40 words; mention the main evidence and any uncertainty.",
    "test_case_style": (
        "Use compact prompts when valid, organized as two four-case contrast ladders from evidence-like positives "
        "to expected negatives, changing one factor at a time."
    ),
}


@dataclass(frozen=True)
class AutoAnnotationConfig:
    input_json: Path | None = None
    evidence_root: Path = DEFAULT_OUTPUT_ROOT
    annotation_root: Path = DEFAULT_ANNOTATION_ROOT
    icl_example_root: Path = DEFAULT_ICL_EXAMPLE_ROOT
    run_id: str | None = None
    layer: str | None = None
    feature_id: int | None = None
    provider: str = "mi"
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_MI_BASE_URL
    api_key_file: Path = DEFAULT_MI_TOKEN_PATH
    max_tokens: int = 1200
    timeout: float = 120.0
    retries: int = 3
    retry_sleep_seconds: float = 15.0
    limit: int | None = None
    use_icl_examples: bool = True
    force: bool = False
    dry_run: bool = False


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run auto-annotation on v9 feature evidence JSONs.")
    parser.add_argument("--input-json", type=Path, default=None, help="Annotate exactly one evidence JSON.")
    parser.add_argument("--evidence-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--annotation-root", type=Path, default=DEFAULT_ANNOTATION_ROOT)
    parser.add_argument("--icl-example-root", type=Path, default=DEFAULT_ICL_EXAMPLE_ROOT)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--layer", default=None)
    parser.add_argument("--feature-id", type=int, default=None)
    parser.add_argument("--provider", choices=["mi", "ollama", "openai", "deepseek", "tinker-sdk"], default="mi")
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--api-key-file", type=Path, default=DEFAULT_MI_TOKEN_PATH)
    parser.add_argument("--max-tokens", type=int, default=1200)
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--retry-sleep-seconds", type=float, default=15.0)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-icl-examples", action="store_true", help="Do not include built-in annotation examples.")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write request preview(s), but do not call the annotation provider.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    cfg = AutoAnnotationConfig(
        input_json=args.input_json,
        evidence_root=args.evidence_root,
        annotation_root=args.annotation_root,
        icl_example_root=args.icl_example_root,
        run_id=args.run_id,
        layer=args.layer,
        feature_id=args.feature_id,
        provider=str(args.provider),
        model=resolve_model(str(args.model or default_model_for_provider(str(args.provider)))),
        base_url=str(
            args.base_url
            or os.environ.get(env_base_url_key(str(args.provider)))
            or default_base_url_for_provider(str(args.provider))
        ),
        api_key_file=args.api_key_file,
        max_tokens=int(args.max_tokens),
        timeout=float(args.timeout),
        retries=int(args.retries),
        retry_sleep_seconds=float(args.retry_sleep_seconds),
        limit=args.limit,
        use_icl_examples=not bool(args.no_icl_examples),
        force=bool(args.force),
        dry_run=bool(args.dry_run),
    )
    outputs = run_auto_annotation(cfg)
    for output in outputs:
        print(output)
    print(f"{'prepared' if cfg.dry_run else 'annotated'} {len(outputs)} feature(s)")


def run_auto_annotation(cfg: AutoAnnotationConfig) -> list[Path]:
    evidence_paths = select_evidence_paths(cfg)
    if cfg.limit is not None:
        evidence_paths = evidence_paths[: max(0, int(cfg.limit))]
    if not evidence_paths:
        raise FileNotFoundError("No evidence JSON files matched the requested selection.")

    api_key = "" if cfg.dry_run else load_api_key_for_provider(cfg.provider, cfg.api_key_file)
    outputs = []
    iterator = tqdm(evidence_paths, desc="auto annotate", unit="feature", dynamic_ncols=True)
    for evidence_path in iterator:
        iterator.set_postfix_str(_progress_label(evidence_path, "load"))
        evidence = read_json(evidence_path)
        request_payload = build_request_payload(
            evidence=evidence,
            provider=cfg.provider,
            model=cfg.model,
            max_tokens=int(cfg.max_tokens),
            evidence_root=cfg.evidence_root,
            annotation_root=cfg.annotation_root,
            icl_example_root=cfg.icl_example_root,
            use_icl_examples=bool(cfg.use_icl_examples),
        )
        paths = output_paths_for_evidence(
            evidence_root=cfg.evidence_root,
            annotation_root=cfg.annotation_root,
            evidence_path=evidence_path,
            provider=cfg.provider,
            model=cfg.model,
        )
        if paths["annotation"].exists() and not cfg.force and not cfg.dry_run:
            iterator.set_postfix_str(_progress_label(evidence_path, "skip"))
            outputs.append(paths["annotation"])
            continue
        write_json(paths["request"], request_payload)
        write_request_debug_bundle(paths["request_debug_dir"], request_payload)
        if cfg.dry_run:
            iterator.set_postfix_str(_progress_label(evidence_path, "dry"))
            outputs.append(paths["request"])
            continue
        iterator.set_postfix_str(_progress_label(evidence_path, "call"))
        response_json = call_annotation_provider(
            request_payload=request_payload,
            provider=cfg.provider,
            api_key=api_key,
            base_url=cfg.base_url,
            timeout=float(cfg.timeout),
            retries=int(cfg.retries),
            retry_sleep_seconds=float(cfg.retry_sleep_seconds),
        )
        write_json(paths["raw_response"], response_json)
        output_text = extract_chat_completion_text(response_json)
        repair_response_json = None
        try:
            annotation = parse_annotation(output_text)
        except Exception as exc:
            try:
                annotation, repair_response_json = repair_annotation_output(
                    malformed_output=output_text,
                    original_request_payload=request_payload,
                    provider=cfg.provider,
                    model=cfg.model,
                    api_key=api_key,
                    base_url=cfg.base_url,
                    timeout=float(cfg.timeout),
                    retries=int(cfg.retries),
                    retry_sleep_seconds=float(cfg.retry_sleep_seconds),
                    max_tokens=int(cfg.max_tokens),
                )
                write_json(paths["repair_raw_response"], repair_response_json)
            except Exception as repair_exc:
                write_json(
                    paths["error"],
                    {
                        "created_at": datetime.now(timezone.utc).isoformat(),
                        "provider": cfg.provider,
                        "model": cfg.model,
                        "input_json": str(evidence_path.resolve()),
                        "feature": evidence.get("feature"),
                        "error": str(exc),
                        "repair_error": str(repair_exc),
                        "output_text": output_text,
                        "raw_response_path": str(paths["raw_response"].resolve()),
                        "repair_raw_response_path": str(paths["repair_raw_response"].resolve()),
                    },
                )
                outputs.append(paths["error"])
                iterator.set_postfix_str(_progress_label(evidence_path, "error"))
                continue
        response_info = {
            "id": response_json.get("id"),
            "usage": response_json.get("usage"),
        }
        if repair_response_json is not None:
            response_info["repaired_from_raw_response"] = str(paths["raw_response"].resolve())
            response_info["repair_raw_response"] = str(paths["repair_raw_response"].resolve())
            response_info["repair_id"] = repair_response_json.get("id")
            response_info["repair_usage"] = repair_response_json.get("usage")
        warn_single_token_test_cases(
            annotation,
            model_id=infer_tokenizer_model_id_from_feature(evidence.get("feature")),
            context=str(evidence.get("feature") or evidence_path.parent.name),
        )
        result = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "provider": cfg.provider,
            "model": cfg.model,
            "input_json": str(evidence_path.resolve()),
            "feature": evidence.get("feature"),
            "annotation": annotation,
            "response": response_info,
        }
        write_json(paths["annotation"], result)
        outputs.append(paths["annotation"])
        iterator.set_postfix_str(_progress_label(evidence_path, "done"))
    return outputs


def _progress_label(evidence_path: Path, status: str) -> str:
    feature = evidence_path.parent.name
    layer = evidence_path.parent.parent.name if evidence_path.parent.parent != evidence_path.parent else ""
    return f"{status} {layer}/{feature}".strip()


def select_evidence_paths(cfg: AutoAnnotationConfig) -> list[Path]:
    if cfg.input_json is not None:
        return [cfg.input_json.resolve()]
    evidence_root = cfg.evidence_root.resolve()
    run_pattern = cfg.run_id or "*"
    layer_pattern = cfg.layer or "layer_*"
    if cfg.feature_id is None:
        feature_pattern = "F*"
    else:
        feature_pattern = f"F{int(cfg.feature_id):06d}"
    paths_by_dir: dict[Path, Path] = {}
    for path in sorted(evidence_root.glob(f"{run_pattern}/{layer_pattern}/{feature_pattern}/{LEGACY_COMPACT_EVIDENCE_FILENAME}")):
        paths_by_dir[path.parent] = path
    for path in sorted(evidence_root.glob(f"{run_pattern}/{layer_pattern}/{feature_pattern}/{COMPACT_EVIDENCE_FILENAME}")):
        paths_by_dir[path.parent] = path
    return [paths_by_dir[key] for key in sorted(paths_by_dir)]


def build_request_payload(
    *,
    evidence: dict[str, Any],
    provider: str,
    model: str,
    max_tokens: int,
    evidence_root: Path,
    annotation_root: Path,
    icl_example_root: Path,
    use_icl_examples: bool,
) -> dict[str, Any]:
    messages = build_initial_annotation_messages(evidence=evidence, use_icl_examples=use_icl_examples)
    if provider == "ollama":
        return {
            "model": model,
            "messages": messages,
            "think": False,
            "stream": False,
            "format": "json",
            "options": {
                "num_predict": int(max_tokens),
            },
        }
    payload = {
        "model": model,
        "max_tokens": int(max_tokens),
        "messages": messages,
    }
    if provider == "tinker-sdk":
        return payload
    if provider == "deepseek":
        payload["response_format"] = {"type": "json_object"}
        payload["thinking"] = {"type": "disabled"}
        return payload
    apply_openai_reasoning_controls(payload, provider=provider, model=model)
    payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ica_lens_feature_annotation",
                "strict": True,
                "schema": ANNOTATION_SCHEMA,
            },
    }
    return payload


def build_initial_annotation_messages(*, evidence: dict[str, Any], use_icl_examples: bool = True) -> list[dict[str, str]]:
    messages = [{"role": "system", "content": SYSTEM_PROMPT.strip()}]
    if use_icl_examples:
        messages.append(
            {
                "role": "user",
                "content": stable_json(
                    {
                        "instruction": "Use this style guide only for wording and confidence calibration.",
                        "style_guide": build_annotation_style_guide(),
                    }
                ),
            }
        )
    messages.append(
        {
            "role": "user",
            "content": stable_json(
                {
                    "instruction": INITIAL_ANNOTATION_INSTRUCTION,
                    "response_format": ANNOTATION_RESPONSE_FORMAT,
                    "target_evidence_json": compact_evidence_for_prompt(evidence),
                }
            ),
        }
    )
    return messages


def compact_evidence_for_prompt(evidence: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in evidence.items()
        if key not in {"annotation_instruction", "requested_response_format", "neuronpedia_reference_label"}
    }


def apply_openai_reasoning_controls(payload: dict[str, Any], *, provider: str, model: str) -> None:
    if provider != "openai":
        return
    if not openai_model_supports_reasoning_effort(model):
        return
    payload["reasoning_effort"] = openai_reasoning_effort(model)
    if "max_tokens" in payload and "max_completion_tokens" not in payload:
        payload["max_completion_tokens"] = payload.pop("max_tokens")


def openai_model_supports_reasoning_effort(model: str) -> bool:
    name = model.lower()
    return name.startswith(("gpt-5", "o1", "o3", "o4"))


def openai_reasoning_effort(model: str) -> str:
    name = model.lower()
    if name.startswith("gpt-5"):
        return "none"
    return "minimal"


def infer_tokenizer_model_id_from_feature(feature: Any) -> str | None:
    text = str(feature or "")
    for prefix, model_id in TOKENIZER_MODEL_BY_FEATURE_PREFIX.items():
        if text.startswith(prefix):
            return model_id
    return None


def warn_single_token_test_cases(annotation: dict[str, Any], *, model_id: str | None, context: str) -> None:
    if not model_id:
        return
    test_cases = annotation.get("test_cases")
    if not isinstance(test_cases, list):
        return
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    except Exception as exc:
        print(
            f"warning: could not check test-case token counts for {context} with tokenizer {model_id}: {exc}",
            file=sys.stderr,
        )
        return
    for index, case in enumerate(test_cases):
        if not isinstance(case, dict):
            continue
        text = str(case.get("text") or "")
        if not text:
            continue
        token_texts = non_special_token_texts(tokenizer, text)
        if len(token_texts) == 1:
            print(
                "warning: "
                f"{context} test_cases[{index}] is a one-token prompt under {model_id}: "
                f"text={text!r}, token={token_texts[0]!r}. "
                "Use a short phrase/sentence so the suspected token is not at beginning-of-text.",
                file=sys.stderr,
            )


def non_special_token_texts(tokenizer: Any, text: str) -> list[str]:
    try:
        encoded = tokenizer(text, return_offsets_mapping=True, add_special_tokens=True)
        input_ids = list(encoded["input_ids"])
        offsets = list(encoded["offset_mapping"])
        return [
            tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False)
            for token_id, (start, end) in zip(input_ids, offsets, strict=False)
            if int(end) > int(start)
        ]
    except Exception:
        input_ids = tokenizer.encode(text, add_special_tokens=False)
        return [tokenizer.decode([int(token_id)], clean_up_tokenization_spaces=False) for token_id in input_ids]


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=False, separators=(",", ":"))


def build_icl_examples(*, target_feature: str, icl_example_root: Path) -> list[dict[str, Any]]:
    examples = []
    for bundle in load_icl_example_bundles(icl_example_root=icl_example_root, target_feature=target_feature):
        response = bundle.get("final_response_json") or bundle.get("initial_response_json")
        if not isinstance(response, dict):
            continue
        examples.append(
            {
                "compact_evidence_summary": bundle.get("compact_evidence_summary"),
                "accepted_response_json": normalize_annotation_for_prompt(response),
            }
        )
    return examples


def build_annotation_style_guide() -> dict[str, Any]:
    return ANNOTATION_STYLE_GUIDE


def normalize_annotation_for_prompt(annotation: dict[str, Any]) -> dict[str, Any]:
    return {
        "label": annotation.get("label"),
        "simple_label": annotation.get("simple_label"),
        "description": annotation.get("description"),
        "reasoning": annotation.get("reasoning"),
        "confidence": annotation.get("confidence"),
        "test_cases": [
            {
                "text": str(case.get("text") or ""),
                "expected": str(case.get("expected") or ""),
                "reason": str(case.get("reason") or ""),
            }
            for case in (annotation.get("test_cases") or [])
            if isinstance(case, dict)
        ],
    }


def load_icl_example_bundles(*, icl_example_root: Path, target_feature: str = "") -> list[dict[str, Any]]:
    root = icl_example_root.resolve()
    candidates = sorted(root.glob("*.json")) + sorted(root.glob("*/*.json"))
    bundles = []
    for path in candidates:
        try:
            bundle = read_json(path)
        except (OSError, json.JSONDecodeError):
            continue
        if not isinstance(bundle, dict):
            continue
        if _icl_source_feature(bundle) == target_feature:
            continue
        bundles.append(bundle)
    return bundles


def _icl_source_feature(bundle: dict[str, Any]) -> str:
    source = bundle.get("source")
    if not isinstance(source, dict):
        return ""
    run_id = str(source.get("run_id") or "")
    layer = str(source.get("layer") or "")
    try:
        feature_id = int(source.get("feature_id"))
    except (TypeError, ValueError):
        return ""
    return f"{run_id}/{layer}/F{feature_id}"


def summarize_evidence_for_icl(evidence: dict[str, Any]) -> dict[str, Any]:
    examples = []
    for sample in evidence.get("semantic_examples", [])[:3]:
        if not isinstance(sample, dict):
            continue
        examples.append(
            {
                "target_token": sample.get("target_token"),
                "position": sample.get("position"),
                "relative_activation": sample.get("relative_activation"),
                "marked_activation_window": sample.get("marked_activation_window"),
            }
        )
    if not examples:
        for sample in evidence.get("examples", [])[:3]:
            if not isinstance(sample, dict):
                continue
            erf = sample.get("effective_receptive_field")
            if not isinstance(erf, dict):
                erf = {}
            examples.append(
                {
                    "target_token": sample.get("target_token") or sample.get("text") or sample.get("token"),
                    "position": sample.get("position"),
                    "relative_activation": sample.get("relative_activation"),
                    "estimated_effective_receptive_field_length": erf.get("estimated_effective_receptive_field_length"),
                    "largest_observed_relative_score_jump": erf.get("largest_observed_relative_score_jump"),
                    "tested_context_lengths": erf.get("tested_context_lengths"),
                }
            )
    effective_examples = []
    erf_source = evidence.get("erf_examples")
    if not isinstance(erf_source, list):
        erf_source = evidence.get("examples", [])
    for sample in erf_source[:3]:
        if not isinstance(sample, dict):
            continue
        erf = sample.get("effective_receptive_field")
        if not isinstance(erf, dict):
            erf = {}
        effective_examples.append(
            {
                "target_token": sample.get("target_token") or sample.get("text") or sample.get("token"),
                "position": sample.get("position"),
                "relative_activation": sample.get("relative_activation"),
                "estimated_effective_receptive_field_length": erf.get("estimated_effective_receptive_field_length"),
                "largest_observed_relative_score_jump": erf.get("largest_observed_relative_score_jump"),
                "tested_context_lengths": erf.get("tested_context_lengths"),
            }
        )
    return {
        "feature": evidence.get("feature"),
        "feature_metadata": evidence.get("feature_metadata"),
        "example_selection_summary": evidence.get("example_selection_summary")
        or {
            key: value
            for key, value in (evidence.get("settings") or {}).items()
            if key
            in {
                "example_selection",
                "effective_receptive_field_definition",
                "replay_relative_score_note",
                "right_context_hint_note",
            }
        },
        "semantic_examples": examples,
        "effective_receptive_field_examples": effective_examples,
    }


def call_annotation_provider(
    *,
    request_payload: dict[str, Any],
    provider: str,
    api_key: str,
    base_url: str,
    timeout: float,
    retries: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    if provider == "ollama":
        return call_ollama_chat(
            request_payload=request_payload,
            base_url=base_url,
            timeout=timeout,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
    if provider in {"openai", "deepseek"}:
        return call_openai_chat_completion(
            request_payload=request_payload,
            api_key=api_key,
            base_url=base_url,
            provider_name="DeepSeek" if provider == "deepseek" else "OpenAI",
            timeout=timeout,
            retries=retries,
            retry_sleep_seconds=retry_sleep_seconds,
        )
    if provider == "tinker-sdk":
        return call_tinker_sdk_chat(
            request_payload=request_payload,
            api_key=api_key,
            timeout=timeout,
        )
    return call_mimo_chat_completion(
        request_payload=request_payload,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        retry_sleep_seconds=retry_sleep_seconds,
    )


def repair_annotation_output(
    *,
    malformed_output: str,
    original_request_payload: dict[str, Any],
    provider: str,
    model: str,
    api_key: str,
    base_url: str,
    timeout: float,
    retries: int,
    retry_sleep_seconds: float,
    max_tokens: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    request_payload = build_repair_request_payload(
        malformed_output=malformed_output,
        original_request_payload=original_request_payload,
        provider=provider,
        model=model,
        max_tokens=max_tokens,
    )
    response_json = call_annotation_provider(
        request_payload=request_payload,
        provider=provider,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        retry_sleep_seconds=retry_sleep_seconds,
    )
    annotation = parse_annotation(extract_chat_completion_text(response_json))
    return annotation, response_json


def build_repair_request_payload(
    *,
    malformed_output: str,
    original_request_payload: dict[str, Any],
    provider: str,
    model: str,
    max_tokens: int,
) -> dict[str, Any]:
    payload = {
        "task": "Repair this malformed feature annotation into valid schema JSON.",
        "response_schema": ANNOTATION_SCHEMA,
        "malformed_output": malformed_output,
        "original_messages": original_request_payload.get("messages", []),
    }
    messages = [
        {"role": "system", "content": REPAIR_SYSTEM_PROMPT.strip()},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))},
    ]
    if provider == "ollama":
        return {
            "model": model,
            "messages": messages,
            "think": False,
            "stream": False,
            "format": "json",
            "options": {"num_predict": max(1800, int(max_tokens))},
        }
    payload = {
        "model": model,
        "max_tokens": max(1800, int(max_tokens)),
        "messages": messages,
    }
    if provider == "tinker-sdk":
        return payload
    if provider == "deepseek":
        payload["response_format"] = {"type": "json_object"}
        payload["thinking"] = {"type": "disabled"}
        return payload
    payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "ica_lens_feature_annotation_repair",
                "strict": True,
                "schema": ANNOTATION_SCHEMA,
            },
    }
    return payload


def call_mimo_chat_completion(
    *,
    request_payload: dict[str, Any],
    api_key: str,
    base_url: str,
    timeout: float,
    retries: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(int(retries) + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            if attempt >= int(retries) or exc.code not in {429, 500, 502, 503, 504}:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Mimo API HTTP {exc.code}: {detail}") from exc
            time.sleep(_retry_sleep(exc, default=float(retry_sleep_seconds)))
        except urllib.error.URLError as exc:
            if attempt >= int(retries):
                raise RuntimeError(f"Mimo API request failed: {exc}") from exc
            time.sleep(float(retry_sleep_seconds))
    raise RuntimeError("unreachable Mimo retry state")


def call_openai_chat_completion(
    *,
    request_payload: dict[str, Any],
    api_key: str,
    base_url: str,
    provider_name: str = "OpenAI",
    timeout: float,
    retries: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    return call_openai_compatible_chat_completion(
        request_payload=request_payload,
        api_key=api_key,
        base_url=base_url,
        timeout=timeout,
        retries=retries,
        retry_sleep_seconds=retry_sleep_seconds,
        provider_name=provider_name,
    )


def call_openai_compatible_chat_completion(
    *,
    request_payload: dict[str, Any],
    api_key: str,
    base_url: str,
    timeout: float,
    retries: int,
    retry_sleep_seconds: float,
    provider_name: str,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/chat/completions"
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    for attempt in range(int(retries) + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            if attempt >= int(retries) or exc.code not in {429, 500, 502, 503, 504}:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"{provider_name} API HTTP {exc.code}: {detail}") from exc
            time.sleep(_retry_sleep(exc, default=float(retry_sleep_seconds)))
        except urllib.error.URLError as exc:
            if attempt >= int(retries):
                raise RuntimeError(f"{provider_name} API request failed: {exc}") from exc
            time.sleep(float(retry_sleep_seconds))
    raise RuntimeError(f"unreachable {provider_name} retry state")


def call_tinker_sdk_chat(
    *,
    request_payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    try:
        import tinker
    except ImportError as exc:
        try:
            return call_tinker_sdk_chat_bridge(
                request_payload=request_payload,
                api_key=api_key,
                timeout=timeout,
            )
        except Exception as bridge_exc:
            raise RuntimeError(
                "Tinker SDK is not installed in this Python environment, and the v9 Tinker bridge failed. "
                "Run `uv add tinker` in v9 first, or check the v9 .venv."
            ) from bridge_exc

    return call_tinker_sdk_chat_with_module(
        tinker=tinker,
        request_payload=request_payload,
        api_key=api_key,
        timeout=timeout,
    )


def call_tinker_sdk_chat_with_module(
    *,
    tinker: Any,
    request_payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    model = str(request_payload.get("model") or DEFAULT_TINKER_SDK_MODEL)
    messages = request_payload.get("messages")
    if not isinstance(messages, list):
        raise RuntimeError("Tinker SDK request payload must contain chat messages.")
    max_tokens = int(request_payload.get("max_tokens") or 1200)

    old_key = os.environ.get("TINKER_API_KEY")
    os.environ["TINKER_API_KEY"] = api_key
    try:
        service_client = tinker.ServiceClient()
        sampling_client = service_client.create_sampling_client(base_model=model)
        sampling_client = resolve_tinker_future(sampling_client, timeout=float(timeout))
        tokenizer = sampling_client.get_tokenizer()
        prompt_text = render_chat_prompt_for_sampling(tokenizer=tokenizer, messages=messages)
        token_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt = tinker.ModelInput(chunks=[tinker.EncodedTextChunk(tokens=token_ids)])
        sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=0,
            top_p=1,
        )
        response = sampling_client.sample(prompt, num_samples=1, sampling_params=sampling_params)
        response = resolve_tinker_future(response, timeout=float(timeout))
    finally:
        if old_key is None:
            os.environ.pop("TINKER_API_KEY", None)
        else:
            os.environ["TINKER_API_KEY"] = old_key

    sequences = getattr(response, "sequences", None)
    if not sequences:
        raise RuntimeError(f"Tinker SDK returned no sampled sequences: {response!r}")
    sequence = sequences[0]
    sampled_tokens = getattr(sequence, "tokens_np", None)
    if sampled_tokens is not None:
        sampled_token_ids = sampled_tokens.tolist()
    else:
        sampled_token_ids = list(getattr(sequence, "_tokens_list", []) or getattr(sequence, "tokens", []) or [])
    output_text = tokenizer.decode(sampled_token_ids, skip_special_tokens=True)
    return {
        "provider": "tinker-sdk",
        "model": model,
        "choices": [
            {
                "message": {
                    "role": "assistant",
                    "content": output_text,
                }
            }
        ],
        "tinker_stop_reason": str(getattr(sequence, "stop_reason", "")),
    }


def call_tinker_sdk_chat_bridge(
    *,
    request_payload: dict[str, Any],
    api_key: str,
    timeout: float,
) -> dict[str, Any]:
    bridge = V9_ROOT / "scripts" / "tinker_sdk_chat_bridge.py"
    python = V9_ROOT / ".venv" / "bin" / "python3"
    if not python.is_file():
        python = V9_ROOT / ".venv" / "bin" / "python"
    if not python.is_file():
        raise FileNotFoundError(f"Missing v9 Python interpreter for Tinker bridge: {python}")
    packet = {
        "request_payload": request_payload,
        "api_key": api_key,
        "timeout": float(timeout),
    }
    try:
        completed = subprocess.run(
            [str(python), str(bridge)],
            input=json.dumps(packet, ensure_ascii=False),
            text=True,
            capture_output=True,
            check=True,
            cwd=str(V9_ROOT),
        )
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or str(exc)).strip()
        raise RuntimeError(f"Tinker bridge failed: {detail}") from exc
    return json.loads(completed.stdout)


def render_chat_prompt_for_sampling(*, tokenizer: Any, messages: list[Any]) -> str:
    normalized_messages = [
        {"role": str(message.get("role") or "user"), "content": str(message.get("content") or "")}
        for message in messages
        if isinstance(message, dict)
    ]
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        except TypeError:
            pass
        try:
            return tokenizer.apply_chat_template(
                normalized_messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    chunks = []
    for message in normalized_messages:
        chunks.append(f"{message['role'].upper()}:\n{message['content']}")
    chunks.append("ASSISTANT:\n")
    return "\n\n".join(chunks)


def resolve_tinker_future(value: Any, *, timeout: float) -> Any:
    if hasattr(value, "result") and callable(value.result):
        try:
            return value.result(timeout=timeout)
        except TypeError:
            return value.result()
    return value


def call_ollama_chat(
    *,
    request_payload: dict[str, Any],
    base_url: str,
    timeout: float,
    retries: int,
    retry_sleep_seconds: float,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/api/chat"
    body = json.dumps(request_payload, ensure_ascii=False).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    for attempt in range(int(retries) + 1):
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=float(timeout)) as response:
                payload = response.read().decode("utf-8")
            return json.loads(payload)
        except urllib.error.HTTPError as exc:
            if attempt >= int(retries) or exc.code not in {429, 500, 502, 503, 504}:
                detail = exc.read().decode("utf-8", errors="replace")
                raise RuntimeError(f"Ollama API HTTP {exc.code}: {detail}") from exc
            time.sleep(_retry_sleep(exc, default=float(retry_sleep_seconds)))
        except urllib.error.URLError as exc:
            if attempt >= int(retries):
                raise RuntimeError(f"Ollama API request failed: {exc}") from exc
            time.sleep(float(retry_sleep_seconds))
    raise RuntimeError("unreachable Ollama retry state")


def _retry_sleep(exc: urllib.error.HTTPError, *, default: float) -> float:
    retry_after = exc.headers.get("Retry-After") if exc.headers else None
    if retry_after:
        try:
            return max(1.0, float(retry_after))
        except ValueError:
            pass
    return max(1.0, default)


def output_paths_for_evidence(
    *,
    evidence_root: Path,
    annotation_root: Path,
    evidence_path: Path,
    provider: str,
    model: str,
) -> dict[str, Path]:
    relative = evidence_path.resolve().relative_to(evidence_root.resolve())
    output_dir = annotation_root.resolve() / relative.parent
    label = model_output_label(model, provider=provider)
    return {
        "annotation": output_dir / f"{label}_annotation.json",
        "raw_response": output_dir / f"{label}_raw_response.json",
        "repair_raw_response": output_dir / f"{label}_repair_raw_response.json",
        "request": output_dir / f"{label}_request_preview.json",
        "request_debug_dir": output_dir / f"{label}_request_debug",
        "error": output_dir / f"{label}_error.json",
    }


def resolve_model(model: str) -> str:
    return MODEL_ALIASES.get(model, model)


def default_model_for_provider(provider: str) -> str:
    if provider == "ollama":
        return DEFAULT_OLLAMA_MODEL
    if provider == "openai":
        return DEFAULT_OPENAI_MODEL
    if provider == "deepseek":
        return os.environ.get("DEEPSEEK_MODEL", "").strip() or DEFAULT_DEEPSEEK_MODEL
    if provider == "tinker-sdk":
        return os.environ.get("TINKER_MODEL", "").strip() or DEFAULT_TINKER_SDK_MODEL
    return DEFAULT_MODEL


def default_base_url_for_provider(provider: str) -> str:
    if provider == "ollama":
        return DEFAULT_OLLAMA_BASE_URL
    if provider == "openai":
        return DEFAULT_OPENAI_BASE_URL
    if provider == "deepseek":
        return DEFAULT_DEEPSEEK_BASE_URL
    if provider == "tinker-sdk":
        return ""
    return DEFAULT_MI_BASE_URL


def env_base_url_key(provider: str) -> str:
    if provider == "ollama":
        return "OLLAMA_BASE_URL"
    if provider == "openai":
        return "OPENAI_BASE_URL"
    if provider == "deepseek":
        return "DEEPSEEK_BASE_URL"
    if provider == "tinker-sdk":
        return "TINKER_BASE_URL"
    return "MI_BASE_URL"


def model_output_label(model: str, *, provider: str = "mi") -> str:
    model = MODEL_LABEL_ALIASES.get(model, model)
    return model.replace("/", "_").replace(":", "_")


def write_request_debug_bundle(output_dir: Path, request_payload: dict[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    messages = request_payload.get("messages")
    messages = messages if isinstance(messages, list) else []
    manifest = {
        "model": request_payload.get("model"),
        "message_count": len(messages),
        "request_keys": sorted(str(key) for key in request_payload.keys()),
        "message_files": [],
    }
    write_json(
        output_dir / "request_options.json",
        {key: value for key, value in request_payload.items() if key != "messages"},
    )
    for index, message in enumerate(messages, start=1):
        role = str(message.get("role") or "unknown") if isinstance(message, dict) else "unknown"
        content = str(message.get("content") or "") if isinstance(message, dict) else str(message)
        parsed = parse_json_content(content)
        filename = f"message_{index:02d}_{safe_filename(role)}.json"
        message_packet: dict[str, Any] = {
            "index": index,
            "role": role,
            "content_characters": len(content),
        }
        if parsed is None:
            message_packet["content_text"] = content
        else:
            message_packet["content_json"] = parsed
        write_json(output_dir / filename, message_packet)
        manifest["message_files"].append(
            {
                "index": index,
                "role": role,
                "file": filename,
                "content_characters": len(content),
                "content_type": "json" if parsed is not None else "text",
            }
        )
    write_json(output_dir / "manifest.json", manifest)


def parse_json_content(content: str) -> Any | None:
    try:
        return json.loads(content)
    except Exception:
        return None


def safe_filename(value: str) -> str:
    cleaned = "".join(ch.lower() if ch.isalnum() else "_" for ch in value).strip("_")
    return cleaned or "message"


def load_api_key_for_provider(provider: str, api_key_file: Path) -> str:
    if provider == "ollama":
        return ""
    if provider == "openai":
        return load_openai_api_key(api_key_file)
    if provider == "deepseek":
        return load_deepseek_api_key(api_key_file)
    if provider == "tinker-sdk":
        return load_tinker_api_key(api_key_file)
    return load_mi_api_key(api_key_file)


def load_mi_api_key(api_key_file: Path) -> str:
    env_key = os.environ.get("MI_API_KEY", "").strip()
    if env_key:
        return env_key
    if api_key_file.exists():
        key = api_key_file.read_text(encoding="utf-8").strip()
        if key:
            return key
    raise FileNotFoundError(f"No Mimo API key found. Set MI_API_KEY or create {api_key_file}.")


def load_openai_api_key(api_key_file: Path) -> str:
    env_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key
    candidates = []
    if api_key_file != DEFAULT_MI_TOKEN_PATH:
        candidates.append(api_key_file)
    if DEFAULT_OPENAI_TOKEN_PATH not in candidates:
        candidates.append(DEFAULT_OPENAI_TOKEN_PATH)
    for path in candidates:
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
    raise FileNotFoundError(
        f"No OpenAI API key found. Set OPENAI_API_KEY or create {DEFAULT_OPENAI_TOKEN_PATH}."
    )


def load_tinker_api_key(api_key_file: Path) -> str:
    env_key = os.environ.get("TINKER_API_KEY", "").strip()
    if env_key:
        return env_key
    candidates = []
    if api_key_file != DEFAULT_MI_TOKEN_PATH:
        candidates.append(api_key_file)
    if DEFAULT_TINKER_TOKEN_PATH not in candidates:
        candidates.append(DEFAULT_TINKER_TOKEN_PATH)
    for path in candidates:
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
    raise FileNotFoundError(
        f"No Tinker API key found. Set TINKER_API_KEY or create {DEFAULT_TINKER_TOKEN_PATH}."
    )


def load_deepseek_api_key(api_key_file: Path) -> str:
    env_key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if env_key:
        return env_key
    candidates = []
    if api_key_file != DEFAULT_MI_TOKEN_PATH:
        candidates.append(api_key_file)
    if DEFAULT_DEEPSEEK_TOKEN_PATH not in candidates:
        candidates.append(DEFAULT_DEEPSEEK_TOKEN_PATH)
    for path in candidates:
        if path.exists():
            key = path.read_text(encoding="utf-8").strip()
            if key:
                return key
    raise FileNotFoundError(
        f"No DeepSeek API key found. Set DEEPSEEK_API_KEY or create {DEFAULT_DEEPSEEK_TOKEN_PATH}."
    )


def extract_chat_completion_text(response_json: dict[str, Any]) -> str:
    message = response_json.get("message")
    if isinstance(message, dict):
        content = message.get("content", "")
        if isinstance(content, str) and content.strip():
            return content.strip()
    choices = response_json.get("choices", [])
    if not choices:
        raise RuntimeError(f"No chat completion choices found in response id={response_json.get('id')!r}.")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str) and content.strip():
        return content.strip()
    if isinstance(content, list):
        chunks = []
        for part in content:
            if isinstance(part, dict):
                chunks.append(str(part.get("text", "") or part.get("content", "")))
        text = "".join(chunks).strip()
        if text:
            return text
    raise RuntimeError(f"No chat completion text found in response id={response_json.get('id')!r}.")


def parse_annotation(output_text: str) -> dict[str, Any]:
    text = output_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    text = _repair_common_json_key_typos(text)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        parsed = _parse_jsonish_annotation(text, output_text, exc)
    required = set(ANNOTATION_SCHEMA["required"])
    missing = required.difference(parsed)
    if missing:
        raise RuntimeError(f"Annotation missing required keys {sorted(missing)}: {parsed}")
    extra = set(parsed).difference(required)
    if extra:
        raise RuntimeError(f"Annotation has unexpected keys {sorted(extra)}: {parsed}")
    string_keys = required.difference({"test_cases"})
    for key in string_keys:
        if not isinstance(parsed[key], str) or not parsed[key].strip():
            raise RuntimeError(f"Annotation field {key!r} must be a non-empty string: {parsed}")
    test_cases = _parse_test_cases(parsed["test_cases"])
    confidence = parsed["confidence"].strip().lower()
    if confidence not in {"high", "medium", "low"}:
        raise RuntimeError(f"Annotation confidence must be high, medium, or low: {parsed}")
    return {
        "label": parsed["label"].strip(),
        "simple_label": parsed["simple_label"].strip(),
        "description": parsed["description"].strip(),
        "reasoning": parsed["reasoning"].strip(),
        "confidence": confidence,
        "test_cases": test_cases,
    }


def _parse_jsonish_annotation(text: str, output_text: str, json_exc: json.JSONDecodeError) -> dict[str, Any]:
    try:
        parsed = ast.literal_eval(_pythonize_json_literals(text))
    except (SyntaxError, ValueError, TypeError) as exc:
        raise RuntimeError(f"Model output was not valid JSON: {output_text}") from json_exc
    if not isinstance(parsed, dict):
        raise RuntimeError(f"Model output was not a JSON object: {output_text}") from json_exc
    return parsed


def _repair_common_json_key_typos(text: str) -> str:
    keys = (
        "label",
        "simple_label",
        "description",
        "reasoning",
        "confidence",
        "test_cases",
        "text",
        "expected",
        "reason",
    )
    key_pattern = "|".join(re.escape(key) for key in keys)
    text = re.sub(rf'([{{,]\s*)({key_pattern})"(\s*:)', r'\1"\2"\3', text)
    text = re.sub(rf'([{{,]\s*)({key_pattern})(\s*:)', r'\1"\2"\3', text)
    return text


def _pythonize_json_literals(text: str) -> str:
    text = re.sub(r"\btrue\b", "True", text)
    text = re.sub(r"\bfalse\b", "False", text)
    text = re.sub(r"\bnull\b", "None", text)
    return text


def _parse_test_cases(value: Any) -> list[dict[str, str]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"Annotation test_cases must be a non-empty list: {value!r}")
    if len(value) != 8:
        raise RuntimeError(f"Annotation test_cases must contain exactly 8 cases, got {len(value)}: {value!r}")
    out = []
    for index, item in enumerate(value):
        if not isinstance(item, dict):
            raise RuntimeError(f"Annotation test case {index} must be an object: {item!r}")
        if "expected" not in item and "example" in item:
            item = {**item, "expected": item["example"]}
            item.pop("example", None)
        required = {"text", "expected", "reason"}
        missing = required.difference(item)
        if missing:
            raise RuntimeError(f"Annotation test case {index} missing keys {sorted(missing)}: {item!r}")
        parsed = {}
        if not isinstance(item["text"], str):
            raise RuntimeError(f"Annotation test case {index} field 'text' must be a string: {item!r}")
        if item["text"] == "":
            continue
        parsed["text"] = normalize_test_case_text(item["text"])
        for key in ("expected", "reason"):
            if not isinstance(item[key], str) or not item[key].strip():
                raise RuntimeError(f"Annotation test case {index} field {key!r} must be a non-empty string: {item!r}")
            parsed[key] = item[key].strip()
        parsed["expected"] = parsed["expected"].lower()
        if parsed["expected"] not in {"activate", "not_activate", "ambiguous"}:
            raise RuntimeError(f"Annotation test case {index} expected must be activate, not_activate, or ambiguous: {item!r}")
        out.append(parsed)
    if not out:
        raise RuntimeError(f"Annotation test_cases contained no non-empty prompts: {value!r}")
    return out


def normalize_test_case_text(text: str) -> str:
    text = text.replace("\\r\\n", "\n").replace("\\n", "\n").replace("\\r", "\r").replace("\\t", "\t")
    return re.sub(r"\[\d+\]", "", text)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
