# vLLM serving benchmark (AWQ W4A16, stage-3 distilled)

- Endpoint: `http://127.0.0.1:8801` · served via `serve_vllm.sh`
- Artifact: `artifacts/awq-w4a16-stage3`

## Exam (same 100 held-out questions, greedy)

- **80/100** correct = 0.8000
- Wall time for all 100 questions: 2.3s

## Decode throughput vs concurrency (128 forced tokens per request)

| concurrency | wall (s) | tokens/s |
|---|---|---|
| 1 | 0.85 | 151 |
| 4 | 0.94 | 543 |
| 16 | 1.65 | 1,240 |
| 32 | 2.31 | 1,773 |
| 48 | 2.91 | 2,114 |
| 64 | 21.59 | 380 |

## GPU memory while serving

- GPU total in use: 24038 MiB
- per-process readout unavailable (WSL2); see server log facts below

## From the server log

- weights_gib: 3.88
- weights_load_seconds: 3.587943
- kv_cache_gib: 17.36
- kv_cache_tokens: 142208
- engine_init_seconds: 21.45

## Footprint on the card: compressed vs fp16 (both served via vLLM)

Same RTX 4090 (24 GiB), same `--max-model-len 4096`, vLLM 0.15.1.

| | weights on GPU | weight load | KV cache free | KV cache capacity | concurrent 4k contexts |
|---|---|---|---|---|---|
| AWQ W4A16 (ships) | 3.88 GiB | 3.6 s | 17.36 GiB | 142,208 tok | ~34 |
| fp16 dense (distilled) | 13.5 GiB | 9.3 s | 7.75 GiB | 63,456 tok | ~15 |

Compression leaves **2.24×** the room for KV cache — so it buys serving concurrency and
context length, not merely the ability to fit.
