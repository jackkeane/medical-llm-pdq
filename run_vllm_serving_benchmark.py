"""Benchmark the vLLM-served compressed model (see serve_vllm.sh).

Measures, against the running OpenAI-compatible endpoint:
  1. Accuracy on the same 100 held-out questions every other step used
     (same prompt, greedy decoding, first-word grading).
  2. Decode throughput at increasing request concurrency.
  3. GPU memory actually held by the server process.

Writes reports/vllm_serving_metrics.{json,md}.
"""

import argparse
import json
import re
import statistics
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

PDQ = Path(__file__).resolve().parent
ROOT = PDQ.parent
PROJECT = ROOT / 'medical-llm'

TEST_JSON = PROJECT / 'data' / 'processed' / 'medical_test.json'
REPORT_JSON = PDQ / 'reports' / 'vllm_serving_metrics.json'
REPORT_MD = PDQ / 'reports' / 'vllm_serving_metrics.md'

GEN_TOKENS = 128
CONCURRENCY_LEVELS = [1, 4, 16, 32, 48, 64]


@dataclass
class ExamResult:
    total: int
    correct: int
    accuracy: float
    wall_seconds: float


@dataclass
class ThroughputPoint:
    concurrency: int
    requests: int
    tokens_per_request: int
    wall_seconds: float
    decode_tokens_per_second: float


def load_json(path: Path):
    with path.open() as f:
        return json.load(f)


def parse_label(text: str) -> str:
    t = text.strip().lower()
    m = re.match(r'^(yes|no|maybe)\b', t)
    return m.group(1) if m else 'unknown'


def build_prompt(ex: dict) -> str:
    return (
        'You are a medical QA assistant. Answer with yes, no, or maybe first, then a brief rationale.\n\n'
        f"Instruction: {ex.get('instruction', '').strip()}\n"
        f"Context: {ex.get('input', '').strip()}\n"
        'Answer:'
    )


def post_completion(base_url: str, model: str, prompt: str, max_tokens: int,
                    ignore_eos: bool = False, timeout: float = 300.0) -> str:
    payload = {
        'model': model,
        'prompt': prompt,
        'max_tokens': max_tokens,
        'temperature': 0.0,
    }
    if ignore_eos:
        payload['ignore_eos'] = True
    req = urllib.request.Request(
        f'{base_url}/v1/completions',
        data=json.dumps(payload).encode(),
        headers={'Content-Type': 'application/json'},
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = json.load(resp)
    return body['choices'][0]['text']


def run_exam(base_url: str, model: str, examples: List[dict], workers: int = 8) -> ExamResult:
    def grade(ex: dict) -> int:
        gen = post_completion(base_url, model, build_prompt(ex), max_tokens=8)
        return int(parse_label(gen) == parse_label(ex.get('output', '')))

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=workers) as pool:
        marks = list(pool.map(grade, examples))
    wall = time.perf_counter() - start
    total, correct = len(marks), sum(marks)
    return ExamResult(total=total, correct=correct,
                      accuracy=(correct / total if total else 0.0), wall_seconds=wall)


def run_throughput(base_url: str, model: str, prompt: str) -> List[ThroughputPoint]:
    points = []
    for level in CONCURRENCY_LEVELS:
        repeats = 3 if level == 1 else 2
        walls = []
        for _ in range(repeats):
            start = time.perf_counter()
            with ThreadPoolExecutor(max_workers=level) as pool:
                list(pool.map(
                    lambda _i: post_completion(base_url, model, prompt,
                                               max_tokens=GEN_TOKENS, ignore_eos=True),
                    range(level),
                ))
            walls.append(time.perf_counter() - start)
        wall = statistics.mean(walls)
        tokens = level * GEN_TOKENS
        points.append(ThroughputPoint(
            concurrency=level, requests=level, tokens_per_request=GEN_TOKENS,
            wall_seconds=wall, decode_tokens_per_second=tokens / wall))
    return points


def gpu_memory_mib() -> dict:
    out = subprocess.run(
        ['nvidia-smi', '--query-compute-apps=pid,used_memory', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    per_process = []
    for line in out.strip().splitlines():
        pid, mem = [p.strip() for p in line.split(',')]
        if not mem.isdigit():  # WSL2 reports [N/A] per process
            continue
        per_process.append({'pid': int(pid), 'used_mib': int(mem)})
    total = subprocess.run(
        ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
        capture_output=True, text=True, check=True).stdout
    return {
        'largest_process_mib': max((p['used_mib'] for p in per_process), default=0),
        'gpu_total_used_mib': int(total.strip().splitlines()[0]),
    }


def parse_server_log(path: Path) -> dict:
    """Best-effort extraction of load-time facts from the vllm serve log."""
    text = path.read_text(errors='replace')
    facts = {}
    m = re.search(r'Model loading took ([\d.]+) GiB(?: memory)? and ([\d.]+) seconds', text)
    if m:
        facts['weights_gib'] = float(m.group(1))
        facts['weights_load_seconds'] = float(m.group(2))
    m = re.search(r'Available KV cache memory:\s*([\d.]+)\s*GiB', text)
    if m:
        facts['kv_cache_gib'] = float(m.group(1))
    m = re.search(r'GPU KV cache size:\s*([\d,]+)\s*tokens', text)
    if m:
        facts['kv_cache_tokens'] = int(m.group(1).replace(',', ''))
    m = re.search(r'init engine.*?took\s*([\d.]+)\s*seconds', text)
    if m:
        facts['engine_init_seconds'] = float(m.group(1))
    return facts


def wait_ready(base_url: str, timeout: float = 30.0) -> None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(f'{base_url}/v1/models', timeout=5):
                return
        except OSError:
            time.sleep(1)
    raise SystemExit(f'server at {base_url} not responding')


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--base-url', default='http://127.0.0.1:8801')
    ap.add_argument('--model', default='medical-7b-compressed')
    ap.add_argument('--server-log', type=Path, default=None,
                    help='optional: vllm serve log to mine for load-time facts')
    args = ap.parse_args()

    wait_ready(args.base_url)
    examples = load_json(TEST_JSON)

    print(f'exam: {len(examples)} questions ...')
    exam = run_exam(args.base_url, args.model, examples)
    print(f'  {exam.correct}/{exam.total} in {exam.wall_seconds:.1f}s')

    print('throughput sweep ...')
    throughput = run_throughput(args.base_url, args.model, build_prompt(examples[0]))
    for p in throughput:
        print(f'  c={p.concurrency:>3}: {p.decode_tokens_per_second:,.0f} tok/s')

    memory = gpu_memory_mib()
    server_log = parse_server_log(args.server_log) if args.server_log else {}

    report = {
        'endpoint': args.base_url,
        'served_model': args.model,
        'artifact': 'artifacts/awq-w4a16-stage3',
        'exam': asdict(exam),
        'throughput': [asdict(p) for p in throughput],
        'gpu_memory': memory,
        'server_log_facts': server_log,
    }
    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    REPORT_JSON.write_text(json.dumps(report, indent=2) + '\n')

    md = ['# vLLM serving benchmark (AWQ W4A16, stage-3 distilled)', '',
          f'- Endpoint: `{args.base_url}` · served via `serve_vllm.sh`',
          f'- Artifact: `artifacts/awq-w4a16-stage3`', '',
          '## Exam (same 100 held-out questions, greedy)', '',
          f'- **{exam.correct}/{exam.total}** correct = {exam.accuracy:.4f}',
          f'- Wall time for all 100 questions: {exam.wall_seconds:.1f}s', '',
          '## Decode throughput vs concurrency '
          f'({GEN_TOKENS} forced tokens per request)', '',
          '| concurrency | wall (s) | tokens/s |', '|---|---|---|']
    for p in throughput:
        md.append(f'| {p.concurrency} | {p.wall_seconds:.2f} | {p.decode_tokens_per_second:,.0f} |')
    md += ['', '## GPU memory while serving', '',
           f"- GPU total in use: {memory['gpu_total_used_mib']} MiB"]
    if memory['largest_process_mib']:
        md.append(f"- vLLM server process: {memory['largest_process_mib']} MiB")
    else:
        md.append('- per-process readout unavailable (WSL2); see server log facts below')
    if server_log:
        md += ['', '## From the server log', '']
        md += [f'- {k}: {v}' for k, v in server_log.items()]
    REPORT_MD.write_text('\n'.join(md) + '\n')
    print(f'wrote {REPORT_JSON.name} and {REPORT_MD.name}')


if __name__ == '__main__':
    main()
