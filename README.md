# Math-Cued Activation

Utilities for generating IMO-AnswerBench responses from `WeiboAI/VibeThinker-3B`
and replaying those responses to capture model activations.

## vLLM Serving

Run vLLM in a separate environment from the project environment. The project
environment is used for dataset/client scripts; the vLLM environment is used
only to serve the model.

```bash
cd ~
uv venv vllm-env --python 3.10
source ~/vllm-env/bin/activate

uv pip install -U pip
uv pip install -U vllm --torch-backend=auto
```

On minimal cloud images, vLLM/Triton may need Python headers:

```bash
apt update
apt install -y python3.10-dev build-essential
```

Start a one-GPU vLLM server with the model-faithful 64K context:

```bash
cd ~/math-cued-activation
source ~/vllm-env/bin/activate

VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0 vllm serve WeiboAI/VibeThinker-3B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95
```

`VLLM_USE_FLASHINFER_SAMPLER=0` avoids FlashInfer sampler JIT failures on
systems without `nvcc` or `/usr/local/cuda`. The `--max-model-len` value is the
total context length: prompt plus generated tokens. VibeThinker-3B was trained
with a single 64K long-context window, so `65536` is the preferred setting. If
vLLM rejects this because the KV cache does not fit on a 24GB GPU, fall back to
`--max-model-len 32768`.

Check readiness from another shell:

```bash
curl http://127.0.0.1:8000/v1/models
```

For two 4090 GPUs, prefer two independent one-GPU vLLM servers:

```bash
mkdir -p logs/vllm
source ~/vllm-env/bin/activate

VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0 nohup vllm serve WeiboAI/VibeThinker-3B \
  --host 127.0.0.1 \
  --port 8000 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95 \
  > logs/vllm/gpu0.log 2>&1 &

VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=1 nohup vllm serve WeiboAI/VibeThinker-3B \
  --host 127.0.0.1 \
  --port 8001 \
  --dtype bfloat16 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.95 \
  > logs/vllm/gpu1.log 2>&1 &
```

Smoke test generation through the OpenAI-compatible vLLM endpoint:

```bash
printf "local-vllm\n" > api_tokens/.vllm

uv run python scripts/generate_imo_text_vllm.py \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --model WeiboAI/VibeThinker-3B \
  --server-name gpu0 \
  --sample-size 4 \
  --start-index 0 \
  --concurrency 4 \
  --generated-text-dir outputs/imo-answerbench-responses/WeiboAI__VibeThinker-3B
```

For a two-server throughput test, split the dataset:

```bash
uv run python scripts/generate_imo_text_vllm.py \
  --api-url http://127.0.0.1:8000/v1/chat/completions \
  --model WeiboAI/VibeThinker-3B \
  --server-name gpu0 \
  --sample-size 200 \
  --start-index 0 \
  --concurrency 6 \
  --generated-text-dir outputs/imo-answerbench-responses/WeiboAI__VibeThinker-3B

uv run python scripts/generate_imo_text_vllm.py \
  --api-url http://127.0.0.1:8001/v1/chat/completions \
  --model WeiboAI/VibeThinker-3B \
  --server-name gpu1 \
  --sample-size 200 \
  --start-index 200 \
  --concurrency 6 \
  --generated-text-dir outputs/imo-answerbench-responses/WeiboAI__VibeThinker-3B
```

## Activation Capture

Stop any vLLM servers before activation capture so the Hugging Face replay has
clean GPU memory. Captures replay the saved full token sequence and save layer
32 activations for both prompt/chat-template tokens and generated tokens:

```bash
uv run python scripts/capture_imo_activations.py \
  --sample-size 400
```

Each saved `.pt` bundle contains:

- `activations`: full captured sequence, prompt followed by generated tokens.
- `prompt_activations`: prompt plus chat-template/generation-prompt tokens.
- `generated_activations`: generated-token activations, computed with the full
  prompt/chat-template context present.

Run a one-example sanity check before a full capture:

```bash
uv run python scripts/capture_imo_activations.py \
  --sample-size 1 \
  --sanity-check-next-token \
  --sanity-check-max-positions 0
```

The next-token sanity check replays the final decoder layer once, applies the
model final norm and `lm_head`, and compares top-1 predictions against the saved
generated token sequence. This is an alignment diagnostic, not a requirement
that every saved token be top-1 under Hugging Face replay. The saved responses
were generated through vLLM and replayed through Hugging Face Transformers, so
small backend/kernel/precision differences can flip a few argmax decisions.

Observed sanity result for `imo-bench-algebra-001` with the full generated
continuation:

```text
checked=10050, same=9986, different=64, top1_match=99.36%
```

Treat roughly 99%+ top-1 agreement as a healthy replay/alignment check; do not
expect 100% agreement.
