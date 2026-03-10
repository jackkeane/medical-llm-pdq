import json
import math
import os
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


ROOT = Path('/home/zz79jk/clawd')
PROJECT = ROOT / 'medical-llm'
PDQ_DIR = ROOT / 'medical-llm-pdq'

BASE_MODEL = PROJECT / 'models' / 'biomistral-7b'
ADAPTER = PROJECT / 'outputs' / 'biomistral-medical'
TEST_JSON = PROJECT / 'data' / 'processed' / 'medical_test.json'
CALIB_JSON = PROJECT / 'data' / 'processed' / 'medical_train.json'
OUT_DIR = PDQ_DIR / 'artifacts' / 'pruned-wanda-2of4'
REPORT_JSON = PDQ_DIR / 'reports' / 'pruning_step_metrics.json'
REPORT_MD = PDQ_DIR / 'reports' / 'pruning_step_metrics.md'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32


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
    instruction = ex.get('instruction', '').strip()
    inp = ex.get('input', '').strip()
    return (
        'You are a medical QA assistant. Answer with yes, no, or maybe first, then a brief rationale.\n\n'
        f'Instruction: {instruction}\n'
        f'Context: {inp}\n'
        'Answer:'
    )


def evaluate_yes_no_maybe(model, tokenizer, examples: List[dict], max_new_tokens: int = 8) -> EvalResult:
    model.eval()
    correct = 0
    total = 0
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
                temperature=0.0,
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = parse_label(gen)
        if pred == ref:
            correct += 1
        total += 1
    return EvalResult(total=total, correct=correct, accuracy=(correct / total if total else 0.0))


def get_target_linears(model: nn.Module) -> Dict[str, nn.Linear]:
    targets = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and any(k in name for k in ['up_proj', 'gate_proj', 'down_proj']):
            targets[name] = module
    return targets


def collect_activation_stats(model, tokenizer, calib_examples: List[dict], max_samples: int = 128) -> Dict[str, torch.Tensor]:
    targets = get_target_linears(model)
    stats: Dict[str, torch.Tensor] = {}
    counts: Dict[str, int] = {}

    hooks = []

    for name, module in targets.items():
        def _hook(mod, inp, n=name):
            x = inp[0]
            if x.dim() == 3:
                x = x.reshape(-1, x.shape[-1])
            x = x.detach().float().abs().mean(dim=0).cpu()
            if n not in stats:
                stats[n] = x
                counts[n] = 1
            else:
                stats[n] += x
                counts[n] += 1

        hooks.append(module.register_forward_pre_hook(_hook))

    model.eval()
    subset = calib_examples[:max_samples]
    for ex in tqdm(subset, desc='calib', leave=False):
        prompt = build_prompt(ex)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)

    for h in hooks:
        h.remove()

    for n in list(stats.keys()):
        stats[n] = stats[n] / max(counts[n], 1)

    return stats


def prune_2of4_wanda(model, act_stats: Dict[str, torch.Tensor]) -> Tuple[int, int]:
    targets = get_target_linears(model)
    total_pruned = 0
    total_considered = 0

    for name, module in tqdm(targets.items(), desc='prune', leave=False):
        w = module.weight.data
        in_features = w.shape[1]
        groups = in_features // 4
        if groups == 0:
            continue

        stat = act_stats.get(name)
        if stat is None:
            continue
        stat = stat.to(w.device, dtype=w.dtype)

        usable = groups * 4
        w_use = w[:, :usable]
        score = w_use.abs() * stat[:usable].unsqueeze(0)

        score_g = score.view(score.shape[0], groups, 4)
        top2 = torch.topk(score_g, k=2, dim=2).indices

        mask = torch.zeros_like(score_g, dtype=torch.bool)
        mask.scatter_(2, top2, True)
        mask = mask.view_as(w_use)

        before_nonzero = (w_use != 0).sum().item()
        w_use.mul_(mask)
        after_nonzero = (w_use != 0).sum().item()

        total_pruned += int(before_nonzero - after_nonzero)
        total_considered += int(w_use.numel())

    return total_pruned, total_considered


def count_nonzero_params(model) -> Tuple[int, int]:
    nz = 0
    total = 0
    for p in model.parameters():
        total += p.numel()
        nz += (p != 0).sum().item()
    return int(nz), int(total)


def main():
    PDQ_DIR.mkdir(parents=True, exist_ok=True)
    (PDQ_DIR / 'reports').mkdir(parents=True, exist_ok=True)
    (PDQ_DIR / 'artifacts').mkdir(parents=True, exist_ok=True)

    print('Loading tokenizer/model...')
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map='auto' if DEVICE == 'cuda' else None,
        low_cpu_mem_usage=True,
    )

    if ADAPTER.exists():
        print('Merging LoRA adapter...')
        model = PeftModel.from_pretrained(base, ADAPTER)
        model = model.merge_and_unload()
    else:
        model = base

    model.eval()

    test_data = load_json(TEST_JSON)
    calib_data = load_json(CALIB_JSON)

    print('Evaluating before pruning...')
    before_eval = evaluate_yes_no_maybe(model, tokenizer, test_data)
    nz_before, total_params = count_nonzero_params(model)

    print('Collecting activation stats...')
    act_stats = collect_activation_stats(model, tokenizer, calib_data, max_samples=128)

    print('Applying Wanda-style 2:4 pruning...')
    pruned, considered = prune_2of4_wanda(model, act_stats)

    print('Evaluating after pruning...')
    after_eval = evaluate_yes_no_maybe(model, tokenizer, test_data)
    nz_after, _ = count_nonzero_params(model)

    print(f'Saving model to {OUT_DIR} ...')
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)

    sparsity_considered = pruned / considered if considered else 0.0
    global_sparsity = 1.0 - (nz_after / total_params)

    report = {
        'method': 'Wanda-style activation-aware 2:4 semi-structured pruning (MLP layers)',
        'device': DEVICE,
        'dtype': str(DTYPE),
        'before': {
            'accuracy': asdict(before_eval),
            'nonzero_params': nz_before,
            'total_params': total_params,
            'global_sparsity': 1.0 - (nz_before / total_params),
        },
        'after': {
            'accuracy': asdict(after_eval),
            'nonzero_params': nz_after,
            'total_params': total_params,
            'global_sparsity': global_sparsity,
        },
        'delta': {
            'accuracy_abs': after_eval.accuracy - before_eval.accuracy,
            'nonzero_params': nz_after - nz_before,
            'pruned_in_target_layers': pruned,
            'target_layers_considered_params': considered,
            'target_layer_effective_sparsity': sparsity_considered,
        },
        'artifacts': {
            'pruned_model_dir': str(OUT_DIR),
        },
    }

    REPORT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with REPORT_JSON.open('w') as f:
        json.dump(report, f, indent=2)

    md = f"""# Pruning Step Report

- **Method**: {report['method']}
- **Pruned model**: `{OUT_DIR}`

## Accuracy (yes/no/maybe on medical_test.json)
- Before: {before_eval.correct}/{before_eval.total} = {before_eval.accuracy:.4f}
- After: {after_eval.correct}/{after_eval.total} = {after_eval.accuracy:.4f}
- Delta: {after_eval.accuracy - before_eval.accuracy:+.4f}

## Weight Change
- Total params: {total_params:,}
- Non-zero before: {nz_before:,}
- Non-zero after: {nz_after:,}
- Delta non-zero params: {nz_after - nz_before:,}
- Global sparsity after: {global_sparsity:.4%}

## Target-layer pruning stats (MLP linear layers)
- Considered params: {considered:,}
- Pruned params: {pruned:,}
- Effective sparsity in considered params: {sparsity_considered:.4%}
"""

    with REPORT_MD.open('w') as f:
        f.write(md)

    print('Done.')
    print(f'Report JSON: {REPORT_JSON}')
    print(f'Report MD:   {REPORT_MD}')


if __name__ == '__main__':
    main()
