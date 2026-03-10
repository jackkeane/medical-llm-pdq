import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path('/home/zz79jk/clawd')
PDQ = ROOT / 'medical-llm-pdq'
PROJECT = ROOT / 'medical-llm'

MODEL_DIR = PDQ / 'artifacts' / 'iterative-wanda-2of4-distilled' / 'stage3_add_down_proj'
TEST_JSON = PROJECT / 'data' / 'processed' / 'medical_test.json'

REPORT_JSON = PDQ / 'reports' / 'quantization_step_metrics.json'
REPORT_MD = PDQ / 'reports' / 'quantization_step_metrics.md'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


@dataclass
class EvalResult:
    total: int
    correct: int
    accuracy: float


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


def evaluate(model, tokenizer, examples: List[dict], max_new_tokens: int = 8) -> EvalResult:
    model.eval()
    total = 0
    correct = 0
    for ex in tqdm(examples, desc='eval', leave=False):
        prompt = build_prompt(ex)
        ref = parse_label(ex.get('output', ''))
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1536)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = parse_label(gen)
        correct += int(pred == ref)
        total += 1
    return EvalResult(total=total, correct=correct, accuracy=(correct / total if total else 0.0))


def load_model(mode: str):
    if mode == 'fp16':
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
            device_map='auto' if DEVICE == 'cuda' else None,
            low_cpu_mem_usage=True,
        )
        return model

    if mode == 'int8':
        qcfg = BitsAndBytesConfig(load_in_8bit=True)
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            quantization_config=qcfg,
            device_map='auto' if DEVICE == 'cuda' else None,
            low_cpu_mem_usage=True,
        )
        return model

    if mode == 'nf4':
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16 if DEVICE == 'cuda' else torch.float32,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_DIR,
            quantization_config=qcfg,
            device_map='auto' if DEVICE == 'cuda' else None,
            low_cpu_mem_usage=True,
        )
        return model

    raise ValueError(mode)


def main():
    (PDQ / 'reports').mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    test_data = load_json(TEST_JSON)

    results = []
    for mode in ['fp16', 'int8', 'nf4']:
        model = load_model(mode)
        if getattr(model.config, 'pad_token_id', None) is None:
            model.config.pad_token_id = tokenizer.pad_token_id

        ev = evaluate(model, tokenizer, test_data)
        footprint = int(model.get_memory_footprint())
        results.append(
            {
                'mode': mode,
                'accuracy': asdict(ev),
                'memory_footprint_bytes': footprint,
                'memory_footprint_gb': footprint / (1024 ** 3),
            }
        )

        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    fp = next(r for r in results if r['mode'] == 'fp16')
    for r in results:
        r['accuracy_delta_vs_fp16'] = r['accuracy']['accuracy'] - fp['accuracy']['accuracy']
        r['memory_saving_vs_fp16'] = 1.0 - (r['memory_footprint_bytes'] / fp['memory_footprint_bytes'])

    report = {
        'source_model': str(MODEL_DIR),
        'note': 'Quantization done with bitsandbytes (INT8/NF4). AWQ package not installed in env.',
        'results': results,
        'recommended_for_next_step': 'nf4' if next(r for r in results if r['mode'] == 'nf4')['accuracy']['accuracy'] >= fp['accuracy']['accuracy'] - 0.02 else 'int8',
    }

    with REPORT_JSON.open('w') as f:
        json.dump(report, f, indent=2)

    lines = ['# Quantization Step Report', '']
    lines.append(f"- Source model: `{MODEL_DIR}`")
    lines.append(f"- Note: {report['note']}")
    lines.append('')
    lines.append('## Results')
    for r in results:
        acc = r['accuracy']
        lines.append(f"### {r['mode']}")
        lines.append(f"- Accuracy: {acc['correct']}/{acc['total']} = {acc['accuracy']:.4f}")
        lines.append(f"- Accuracy delta vs fp16: {r['accuracy_delta_vs_fp16']:+.4f}")
        lines.append(f"- Memory footprint: {r['memory_footprint_gb']:.3f} GB")
        lines.append(f"- Memory saving vs fp16: {r['memory_saving_vs_fp16']:.2%}")
        lines.append('')

    lines.append(f"## Recommendation\n- {report['recommended_for_next_step']}")

    with REPORT_MD.open('w') as f:
        f.write('\n'.join(lines))

    print(f'Report JSON: {REPORT_JSON}')
    print(f'Report MD: {REPORT_MD}')


if __name__ == '__main__':
    main()
