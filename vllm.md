GPU 0:

```bash
cd ~/math-cued-activation
source ~/vllm-env/bin/activate

VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=0 vllm serve WeiboAI/VibeThinker-3B   --host 127.0.0.1   --port 8000   --dtype bfloat16   --max-model-len 65536   --gpu-memory-utilization 0.95
```

GPU 1:

```bash
cd ~/math-cued-activation
source ~/vllm-env/bin/activate

VLLM_USE_FLASHINFER_SAMPLER=0 CUDA_VISIBLE_DEVICES=1 vllm serve WeiboAI/VibeThinker-3B   --host 127.0.0.1   --port 8001   --dtype bfloat16   --max-model-len 65536   --gpu-memory-utilization 0.95
```

Run Infer on GPU 0:

```bash
cd ~/math-cued-activation

uv run python scripts/generate_imo_text_vllm.py   --api-url http://127.0.0.1:8000/v1/chat/completions   --model WeiboAI/VibeThinker-3B   --server-name gpu0   --sample-size 200   --start-index 0   --concurrency 6
```

Run Infer on GPU 1:

```bash
cd ~/math-cued-activation

uv run python scripts/generate_imo_text_vllm.py   --api-url http://127.0.0.1:8001/v1/chat/completions   --model WeiboAI/VibeThinker-3B   --server-name gpu1   --sample-size 200   --start-index 200   --concurrency 6
```