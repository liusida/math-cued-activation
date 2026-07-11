#!/usr/bin/env python3
"""Write an HTML heatmap of activation cosine similarities across model pairs."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from run_vibethinker import capture_sequence_activations, load_model_and_tokenizer


MODELS = {
    "Qwen Coder Base": "Qwen/Qwen2.5-Coder-3B",
    "Qwen Coder Instruct": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "VibeThinker": "WeiboAI/VibeThinker-3B",
}

PAIRS = [
    ("Qwen Coder Base", "Qwen Coder Instruct"),
    ("Qwen Coder Instruct", "VibeThinker"),
    ("Qwen Coder Base", "VibeThinker"),
]

DEFAULT_NON_MATH_PROMPT = (
    "A small cafe opens before sunrise. The owner wipes the counter, starts the "
    "coffee machine, and writes a short note for the first customer: take your "
    "time, breathe, and choose the pastry that makes the morning feel lighter."
)

DEFAULT_MATH_PROMPT = (
    "Let n be a positive integer. Consider all pairs of integers (a, b) with "
    "1 <= a, b <= n. For each pair, write down the remainder of ab when divided "
    "by n + 1. Determine for which n the sum of all these remainders has a "
    "simple closed form, and explain the pattern carefully."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--layer", type=int, default=32)
    parser.add_argument("--max-tokens", type=int, default=160)
    parser.add_argument("--device", default="cuda", help='Device placement. Use "cuda", "cuda:N", or "cpu".')
    parser.add_argument("--dtype", choices=["auto", "bfloat16", "float16", "float32"], default="auto")
    parser.add_argument("--activation-dtype", choices=["float32", "float16", "bfloat16"], default="float32")
    parser.add_argument("--non-math-prompt", default=DEFAULT_NON_MATH_PROMPT)
    parser.add_argument("--math-prompt", default=DEFAULT_MATH_PROMPT)
    parser.add_argument("--non-math-prompt-file", type=Path)
    parser.add_argument("--math-prompt-file", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/visualizations/activation_cosine_heatmap.html"),
    )
    return parser.parse_args()


def read_prompt(text: str, path: Path | None) -> str:
    if path is not None:
        return path.expanduser().read_text()
    return text


def encode_prompt(tokenizer, prompt: str, max_tokens: int) -> list[int]:
    kwargs = {"add_special_tokens": False}
    if max_tokens > 0:
        kwargs.update({"truncation": True, "max_length": max_tokens})
    return [int(token_id) for token_id in tokenizer(prompt, **kwargs)["input_ids"]]


def token_texts(tokenizer, token_ids: list[int]) -> list[str]:
    return [
        tokenizer.decode([token_id], skip_special_tokens=False).replace("\n", "\\n")
        for token_id in token_ids
    ]


def capture_model_prompt(
    *,
    label: str,
    model_id: str,
    prompt: str,
    args: argparse.Namespace,
) -> dict:
    print(f"Loading/capturing {label}: {model_id}", flush=True)
    load_args = argparse.Namespace(model=model_id, device=args.device, dtype=args.dtype)
    model, tokenizer = load_model_and_tokenizer(load_args)
    ids = encode_prompt(tokenizer, prompt, args.max_tokens)
    sequence_ids = torch.tensor(ids, dtype=torch.long)
    capture = capture_sequence_activations(
        model=model,
        sequence_ids=sequence_ids,
        prompt_tokens=len(ids),
        layer=args.layer,
        activation_dtype=args.activation_dtype,
        capture_prompt_activations=True,
        progress=True,
        progress_desc=f"{label} layer {args.layer} ({len(ids)} tok)",
    )
    if capture is None:
        raise RuntimeError(f"No activations captured for {label}.")
    activations = capture.activations.to(dtype=torch.float32)
    return {
        "label": label,
        "model_id": model_id,
        "token_ids": ids,
        "token_texts": token_texts(tokenizer, ids),
        "normalized": F.normalize(activations, p=2, dim=1, eps=1e-12).cpu(),
    }


def cosine_row(left: dict, right: dict) -> tuple[list[float], list[bool]]:
    n = min(len(left["token_ids"]), len(right["token_ids"]))
    cosines = (left["normalized"][:n] * right["normalized"][:n]).sum(dim=1)
    same_tokens = [
        left["token_ids"][index] == right["token_ids"][index]
        for index in range(n)
    ]
    return [float(value) for value in cosines.tolist()], same_tokens


def build_prompt_payload(name: str, prompt: str, args: argparse.Namespace) -> dict:
    captures = {
        label: capture_model_prompt(label=label, model_id=model_id, prompt=prompt, args=args)
        for label, model_id in MODELS.items()
    }
    base_tokens = captures["Qwen Coder Instruct"]["token_texts"]
    rows = []
    for left_label, right_label in PAIRS:
        values, same_tokens = cosine_row(captures[left_label], captures[right_label])
        rows.append(
            {
                "label": f"{left_label} vs {right_label}",
                "values": values,
                "same_tokens": same_tokens,
            }
        )

    return {
        "name": name,
        "prompt": prompt,
        "tokens": base_tokens[: min(len(row["values"]) for row in rows)],
        "rows": rows,
    }


def write_html(payload: dict, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    data_json = json.dumps(payload, ensure_ascii=False)
    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Activation Cosine Heatmap</title>
<style>
  :root {{
    color-scheme: light;
    --fg: #1d2430;
    --muted: #667085;
    --grid: #d0d5dd;
    --bg: #f6f7f9;
    --panel: #ffffff;
  }}
  body {{
    margin: 0;
    font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    background: var(--bg);
    color: var(--fg);
  }}
  main {{
    max-width: 1180px;
    margin: 0 auto;
    padding: 28px 24px 48px;
  }}
  h1 {{
    font-size: 28px;
    margin: 0 0 8px;
  }}
  h2 {{
    font-size: 20px;
    margin: 32px 0 8px;
  }}
  .meta, .prompt, .stats {{
    color: var(--muted);
    font-size: 13px;
    line-height: 1.5;
  }}
  .prompt {{
    max-height: 92px;
    overflow: auto;
    border-left: 3px solid var(--grid);
    padding-left: 12px;
    margin-bottom: 16px;
  }}
  .section {{
    background: var(--panel);
    border: 1px solid #eaecf0;
    border-radius: 8px;
    padding: 18px;
    margin-top: 20px;
    box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04);
  }}
  .heatmap-wrap {{
    overflow-x: auto;
    padding-bottom: 12px;
  }}
  .token-axis, .heatmap {{
    display: grid;
    align-items: center;
  }}
  .token-axis {{
    grid-auto-rows: 124px;
    column-gap: 2px;
    margin-bottom: 10px;
  }}
  .heatmap {{
    grid-auto-rows: 22px;
    gap: 2px;
  }}
  .cell {{
    width: 12px;
    height: 22px;
    border-radius: 2px;
  }}
  .row-label {{
    width: 260px;
    padding-right: 10px;
    font-size: 12px;
    text-align: right;
    white-space: nowrap;
    color: #344054;
  }}
  .token-label {{
    position: relative;
    width: 12px;
    height: 124px;
    font-size: 10px;
    color: #475467;
  }}
  .token-label span {{
    position: absolute;
    left: 50%;
    bottom: 22px;
    transform: translateX(-50%) rotate(-90deg);
    transform-origin: center;
    max-width: 88px;
    overflow: hidden;
    white-space: nowrap;
    text-overflow: ellipsis;
    text-align: right;
  }}
  .legend {{
    display: flex;
    align-items: center;
    gap: 10px;
    margin-top: 12px;
    color: var(--muted);
    font-size: 12px;
  }}
  .bar {{
    width: 220px;
    height: 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, #fff5f0, #fb6a4a, #a50f15);
    border: 1px solid #d0d5dd;
  }}
</style>
</head>
<body>
<main>
  <h1>Activation Cosine Heatmap</h1>
  <div class="meta">Layer {html.escape(str(payload["layer"]))}; cells show cosine similarity between corresponding normalized activation vectors at each token position.</div>
  <div id="app"></div>
</main>
<script>
const DATA = {data_json};

function clamp(x, lo, hi) {{ return Math.max(lo, Math.min(hi, x)); }}
function colorFor(value) {{
  const v = clamp(value, 0, 1);
  const stops = [
    [0.0, [255, 245, 240]],
    [0.5, [251, 106, 74]],
    [1.0, [165, 15, 21]]
  ];
  let a = stops[0], b = stops[2];
  for (let i = 0; i < stops.length - 1; i++) {{
    if (v >= stops[i][0] && v <= stops[i+1][0]) {{
      a = stops[i]; b = stops[i+1]; break;
    }}
  }}
  const t = (v - a[0]) / (b[0] - a[0]);
  const rgb = a[1].map((c, i) => Math.round(c + t * (b[1][i] - c)));
  return `rgb(${{rgb[0]}}, ${{rgb[1]}}, ${{rgb[2]}})`;
}}
function esc(s) {{
  return String(s).replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}
function stats(values) {{
  const mean = values.reduce((a,b) => a+b, 0) / values.length;
  return `mean=${{mean.toFixed(3)}}`;
}}
function renderSection(section) {{
  const n = section.tokens.length;
  const cols = `260px repeat(${{n}}, 12px)`;
  const tokenCols = `260px repeat(${{n}}, 12px)`;
  const rowStats = section.rows.map(row => `<div class="stats"><b>${{esc(row.label)}}:</b> ${{stats(row.values)}}</div>`).join("");
  let html = `<section class="section">
    <h2>${{esc(section.name)}}</h2>
    <div class="prompt">${{esc(section.prompt)}}</div>
    ${{rowStats}}
    <div class="heatmap-wrap">`;
  html += `<div class="token-axis" style="grid-template-columns:${{tokenCols}}"><div></div>`;
  section.tokens.forEach((tok, i) => {{
    html += `<div class="token-label" title="${{i}}: ${{esc(tok)}}"><span>${{esc(tok)}}</span></div>`;
  }});
  html += `</div><div class="heatmap" style="grid-template-columns:${{cols}}">`;
  section.rows.forEach(row => {{
    html += `<div class="row-label">${{esc(row.label)}}</div>`;
    row.values.forEach((value, i) => {{
      const token = section.tokens[i] ?? "";
      const same = row.same_tokens[i] ? "same token id" : "different token id";
      html += `<div class="cell" title="${{esc(row.label)}}\\npos ${{i}} token ${{esc(token)}}\\ncos ${{value.toFixed(4)}}\\n${{same}}" style="background:${{colorFor(value)}}"></div>`;
    }});
  }});
  html += `</div></div>
    <div class="legend"><span>0</span><div class="bar"></div><span>1</span></div>
  </section>`;
  return html;
}}
document.getElementById("app").innerHTML = DATA.sections.map(renderSection).join("");
</script>
</body>
</html>
""",
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    prompts = [
        ("Non-Mathy Prompt", read_prompt(args.non_math_prompt, args.non_math_prompt_file)),
        ("Mathy Prompt", read_prompt(args.math_prompt, args.math_prompt_file)),
    ]
    sections = [build_prompt_payload(name, prompt, args) for name, prompt in prompts]
    write_html(
        {
            "layer": args.layer,
            "max_tokens": args.max_tokens,
            "sections": sections,
        },
        args.output.expanduser(),
    )
    print(f"Saved visualization: {args.output.expanduser()}")


if __name__ == "__main__":
    main()
