import json
import re
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch
from peft import PeftModel
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

PDQ_DIR = Path(__file__).resolve().parent
ROOT = PDQ_DIR.parent
PROJECT = ROOT / 'medical-llm'

BASE_MODEL = PROJECT / 'models' / 'biomistral-7b'
ADAPTER = PROJECT / 'outputs' / 'biomistral-medical'
TEST_JSON = PROJECT / 'data' / 'processed' / 'medical_test.json'
CALIB_JSON = PROJECT / 'data' / 'processed' / 'medical_train.json'

OUT_ROOT = PDQ_DIR / 'artifacts' / 'iterative-wanda-2of4'
REPORT_JSON = PDQ_DIR / 'reports' / 'iterative_pruning_metrics.json'
REPORT_MD = PDQ_DIR / 'reports' / 'iterative_pruning_metrics.md'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

STAGES = [
    ('stage1_up_proj', ['up_proj']),
    ('stage2_add_gate_proj', ['up_proj', 'gate_proj']),
    ('stage3_add_down_proj', ['up_proj', 'gate_proj', 'down_proj']),
]


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
                pad_token_id=tokenizer.eos_token_id,
            )
        gen = tokenizer.decode(out[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
        pred = parse_label(gen)
        correct += int(pred == ref)
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
    for ex in tqdm(calib_examples[:max_samples], desc='calib', leave=False):
        prompt = build_prompt(ex)
        inputs = tokenizer(prompt, return_tensors='pt', truncation=True, max_length=1024)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            model(**inputs)

    for h in hooks:
        h.remove()

    for k in list(stats.keys()):
        stats[k] = stats[k] / max(counts[k], 1)

    return stats


def count_nonzero_params(model):
    nz, total = 0, 0
    for p in model.parameters():
        total += p.numel()
        nz += (p != 0).sum().item()
    return int(nz), int(total)


def prune_selected_modules_2of4(model, act_stats: Dict[str, torch.Tensor], include_keys: List[str], already_pruned: set):
    targets = get_target_linears(model)
    pruned = 0
    considered = 0
    newly_pruned_modules = []

    for name, module in tqdm(targets.items(), desc='prune', leave=False):
        if name in already_pruned:
            continue
        if not any(k in name for k in include_keys):
            continue

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

        before = (w_use != 0).sum().item()
        w_use.mul_(mask)
        after = (w_use != 0).sum().item()

        pruned += int(before - after)
        considered += int(w_use.numel())
        newly_pruned_modules.append(name)

    already_pruned.update(newly_pruned_modules)
    return pruned, considered, newly_pruned_modules


def main():
    (PDQ_DIR / 'reports').mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=DTYPE,
        device_map='auto' if DEVICE == 'cuda' else None,
        low_cpu_mem_usage=True,
    )

    if ADAPTER.exists():
        model = PeftModel.from_pretrained(base, ADAPTER)
        model = model.merge_and_unload()
    else:
        model = base

    test_data = load_json(TEST_JSON)
    calib_data = load_json(CALIB_JSON)

    baseline_eval = evaluate(model, tokenizer, test_data)
    nz0, total = count_nonzero_params(model)
    act_stats = collect_activation_stats(model, tokenizer, calib_data, max_samples=128)

    stage_results = []
    already_pruned = set()

    for stage_name, include in STAGES:
        pruned, considered, modules = prune_selected_modules_2of4(model, act_stats, include, already_pruned)
        ev = evaluate(model, tokenizer, test_data)
        nz, _ = count_nonzero_params(model)

        stage_dir = OUT_ROOT / stage_name
        stage_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(stage_dir)
        tokenizer.save_pretrained(stage_dir)

        stage_results.append(
            {
                'stage': stage_name,
                'include_keys': include,
                'newly_pruned_modules': len(modules),
                'pruned_params_this_stage': pruned,
                'considered_params_this_stage': considered,
                'effective_sparsity_this_stage': (pruned / considered) if considered else 0.0,
                'eval': asdict(ev),
                'nonzero_params': nz,
                'global_sparsity': 1.0 - (nz / total),
                'accuracy_delta_vs_baseline': ev.accuracy - baseline_eval.accuracy,
            }
        )

    report = {
        'method': 'Iterative Wanda-style activation-aware 2:4 semi-structured pruning',
        'device': DEVICE,
        'dtype': str(DTYPE),
        'baseline': {
            'eval': asdict(baseline_eval),
            'nonzero_params': nz0,
            'total_params': total,
            'global_sparsity': 1.0 - (nz0 / total),
        },
        'stages': stage_results,
        'artifact_root': str(OUT_ROOT),
    }

    with REPORT_JSON.open('w') as f:
        json.dump(report, f, indent=2)

    lines = []
    lines.append('# Iterative Pruning Report\n')
    lines.append(f"- **Method**: {report['method']}")
    lines.append(f"- **Artifact root**: `{OUT_ROOT}`\n")
    lines.append('## Baseline')
    lines.append(f"- Accuracy: {baseline_eval.correct}/{baseline_eval.total} = {baseline_eval.accuracy:.4f}")
    lines.append(f"- Non-zero params: {nz0:,} / {total:,}")
    lines.append(f"- Global sparsity: {report['baseline']['global_sparsity']:.4%}\n")

    lines.append('## Stages')
    for s in stage_results:
        ev = s['eval']
        lines.append(f"### {s['stage']}")
        lines.append(f"- Include keys: {s['include_keys']}")
        lines.append(f"- Newly pruned modules: {s['newly_pruned_modules']}")
        lines.append(
            f"- Stage sparsity (considered): {s['pruned_params_this_stage']:,} / {s['considered_params_this_stage']:,} = {s['effective_sparsity_this_stage']:.4%}"
        )
        lines.append(f"- Accuracy: {ev['correct']}/{ev['total']} = {ev['accuracy']:.4f}")
        lines.append(f"- Accuracy delta vs baseline: {s['accuracy_delta_vs_baseline']:+.4f}")
        lines.append(f"- Non-zero params: {s['nonzero_params']:,}")
        lines.append(f"- Global sparsity: {s['global_sparsity']:.4%}\n")

    with REPORT_MD.open('w') as f:
        f.write('\n'.join(lines))

    print(f'Report JSON: {REPORT_JSON}')
    print(f'Report MD: {REPORT_MD}')


if __name__ == '__main__':
    main()
