import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

from llmcompressor.entrypoints.oneshot import oneshot
from llmcompressor.modifiers.awq.base import AWQModifier

PDQ = Path(__file__).resolve().parent
ROOT = PDQ.parent
PROJECT = ROOT / 'medical-llm'

# Distilled checkpoint is adapter-only; merge it with base before AWQ.
DISTILLED_ADAPTER_DIR = PDQ / 'artifacts' / 'iterative-wanda-2of4-distilled' / 'stage3_add_down_proj'
MERGED_DISTILLED_DIR = PDQ / 'artifacts' / 'iterative-wanda-2of4-distilled-merged' / 'stage3_add_down_proj'
MODEL_DIR = MERGED_DISTILLED_DIR
TEST_JSON = PROJECT / 'data' / 'processed' / 'medical_test.json'
TRAIN_JSON = PROJECT / 'data' / 'processed' / 'medical_train.json'

AWQ_DIR = PDQ / 'artifacts' / 'awq-w4a16-stage3'
AWQ_RECIPE = PDQ / 'artifacts' / 'awq_recipe_w4a16.yaml'
AWQ_CALIB_DIR = PDQ / 'artifacts' / 'awq_calibration_data'
AWQ_CALIB_FILE = AWQ_CALIB_DIR / 'train.json'

REPORT_JSON = PDQ / 'reports' / 'awq_quantization_metrics.json'
REPORT_MD = PDQ / 'reports' / 'awq_quantization_metrics.md'

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


def write_awq_recipe(path: Path):
    recipe = """
oneshot_stage:
  quant_modifiers:
    AWQModifier:
      ignore: ["lm_head"]
      duo_scaling: false
      n_grid: 8
      mappings: []
      config_groups:
        group_0:
          targets: ["Linear"]
          input_activations: null
          output_activations: null
          weights:
            num_bits: 4
            type: int
            symmetric: false
            strategy: group
            group_size: 128
""".strip() + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(recipe)


def write_calibration_json(path: Path, train_data: List[dict], n: int = 256):
    rows = []
    for ex in train_data[:n]:
        rows.append({"text": build_prompt(ex)})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w') as f:
        json.dump(rows, f, ensure_ascii=False)


def load_model_fp16(model_path: Path):
    return AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
        device_map='auto' if DEVICE == 'cuda' else None,
        low_cpu_mem_usage=True,
    )


def ensure_merged_distilled_checkpoint():
    if (MERGED_DISTILLED_DIR / 'config.json').exists() and any(MERGED_DISTILLED_DIR.glob('model*.safetensors')):
        return

    MERGED_DISTILLED_DIR.mkdir(parents=True, exist_ok=True)
    adapter_cfg = load_json(DISTILLED_ADAPTER_DIR / 'adapter_config.json')
    base_model_path = Path(adapter_cfg['base_model_name_or_path'])

    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
        device_map='auto' if DEVICE == 'cuda' else None,
        low_cpu_mem_usage=True,
    )
    peft_model = PeftModel.from_pretrained(base_model, DISTILLED_ADAPTER_DIR)
    merged_model = peft_model.merge_and_unload()

    tokenizer = AutoTokenizer.from_pretrained(DISTILLED_ADAPTER_DIR, use_fast=True)
    merged_model.save_pretrained(MERGED_DISTILLED_DIR, safe_serialization=True)
    tokenizer.save_pretrained(MERGED_DISTILLED_DIR)

    del merged_model
    del peft_model
    del base_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def patch_awq_empty_metrics_bug():
    original = AWQModifier._log_error_metrics

    def _safe_log_error_metrics(self):
        if not getattr(self, '_error_metrics', None):
            return
        return original(self)

    AWQModifier._log_error_metrics = _safe_log_error_metrics


def main():
    patch_awq_empty_metrics_bug()
    (PDQ / 'reports').mkdir(parents=True, exist_ok=True)
    AWQ_DIR.mkdir(parents=True, exist_ok=True)

    ensure_merged_distilled_checkpoint()

    test_data = load_json(TEST_JSON)
    train_data = load_json(TRAIN_JSON)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print('Evaluating fp16 baseline...')
    model_fp16 = load_model_fp16(MODEL_DIR)
    if getattr(model_fp16.config, 'pad_token_id', None) is None:
        model_fp16.config.pad_token_id = tokenizer.pad_token_id
    fp16_ev = evaluate(model_fp16, tokenizer, test_data)
    fp16_mem = int(model_fp16.get_memory_footprint())

    del model_fp16
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print('Preparing AWQ recipe + calibration set...')
    write_awq_recipe(AWQ_RECIPE)
    write_calibration_json(AWQ_CALIB_FILE, train_data, n=256)

    print('Running AWQ oneshot quantization...')
    oneshot(
        model=str(MODEL_DIR),
        tokenizer=str(MODEL_DIR),
        recipe=str(AWQ_RECIPE),
        dataset='json',
        dataset_path=str(AWQ_CALIB_DIR),
        text_column='text',
        num_calibration_samples=128,
        max_seq_length=512,
        pad_to_max_length=True,
        batch_size=1,
        output_dir=str(AWQ_DIR),
    )

    print('Evaluating AWQ model...')
    awq_tokenizer = AutoTokenizer.from_pretrained(AWQ_DIR, use_fast=True)
    if awq_tokenizer.pad_token is None:
        awq_tokenizer.pad_token = awq_tokenizer.eos_token

    awq_model = AutoModelForCausalLM.from_pretrained(
        AWQ_DIR,
        torch_dtype=torch.float16 if DEVICE == 'cuda' else torch.float32,
        device_map='auto' if DEVICE == 'cuda' else None,
        low_cpu_mem_usage=True,
    )
    if getattr(awq_model.config, 'pad_token_id', None) is None:
        awq_model.config.pad_token_id = awq_tokenizer.pad_token_id

    awq_ev = evaluate(awq_model, awq_tokenizer, test_data)
    awq_mem = int(awq_model.get_memory_footprint())

    report = {
        'source_model': str(MODEL_DIR),
        'awq_output_dir': str(AWQ_DIR),
        'awq_recipe': str(AWQ_RECIPE),
        'calibration_dataset': str(AWQ_CALIB_FILE),
        'results': {
            'fp16': {
                'accuracy': asdict(fp16_ev),
                'memory_footprint_bytes': fp16_mem,
                'memory_footprint_gb': fp16_mem / (1024 ** 3),
            },
            'awq_w4a16': {
                'accuracy': asdict(awq_ev),
                'memory_footprint_bytes': awq_mem,
                'memory_footprint_gb': awq_mem / (1024 ** 3),
            },
        },
    }

    report['results']['awq_w4a16']['accuracy_delta_vs_fp16'] = (
        report['results']['awq_w4a16']['accuracy']['accuracy']
        - report['results']['fp16']['accuracy']['accuracy']
    )
    report['results']['awq_w4a16']['memory_saving_vs_fp16'] = (
        1.0
        - (
            report['results']['awq_w4a16']['memory_footprint_bytes']
            / report['results']['fp16']['memory_footprint_bytes']
        )
    )

    with REPORT_JSON.open('w') as f:
        json.dump(report, f, indent=2)

    lines = ['# AWQ Quantization Step Report', '']
    lines.append(f"- Source model: `{MODEL_DIR}`")
    lines.append(f"- AWQ output: `{AWQ_DIR}`")
    lines.append(f"- Recipe: `{AWQ_RECIPE}`")
    lines.append('')
    lines.append('## Results')

    fp = report['results']['fp16']
    awq = report['results']['awq_w4a16']

    lines.append('### fp16')
    lines.append(
        f"- Accuracy: {fp['accuracy']['correct']}/{fp['accuracy']['total']} = {fp['accuracy']['accuracy']:.4f}"
    )
    lines.append(f"- Memory footprint: {fp['memory_footprint_gb']:.3f} GB")
    lines.append('')

    lines.append('### awq_w4a16')
    lines.append(
        f"- Accuracy: {awq['accuracy']['correct']}/{awq['accuracy']['total']} = {awq['accuracy']['accuracy']:.4f}"
    )
    lines.append(f"- Accuracy delta vs fp16: {awq['accuracy_delta_vs_fp16']:+.4f}")
    lines.append(f"- Memory footprint: {awq['memory_footprint_gb']:.3f} GB")
    lines.append(f"- Memory saving vs fp16: {awq['memory_saving_vs_fp16']:.2%}")

    with REPORT_MD.open('w') as f:
        f.write('\n'.join(lines))

    print(f'Report JSON: {REPORT_JSON}')
    print(f'Report MD: {REPORT_MD}')


if __name__ == '__main__':
    main()
