import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.nn.functional as F
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

PDQ_DIR = Path(__file__).resolve().parent
ROOT = PDQ_DIR.parent
PROJECT = ROOT / 'medical-llm'

BASE_MODEL = PROJECT / 'models' / 'biomistral-7b'
ADAPTER = PROJECT / 'outputs' / 'biomistral-medical'
TRAIN_JSON = PROJECT / 'data' / 'processed' / 'medical_train.json'
TEST_JSON = PROJECT / 'data' / 'processed' / 'medical_test.json'

PRUNE_ROOT = PDQ_DIR / 'artifacts' / 'iterative-wanda-2of4'
OUT_ROOT = PDQ_DIR / 'artifacts' / 'iterative-wanda-2of4-distilled'
REPORT_JSON = PDQ_DIR / 'reports' / 'iterative_distillation_metrics.json'
REPORT_MD = PDQ_DIR / 'reports' / 'iterative_distillation_metrics.md'

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32

STAGES = [
    'stage1_up_proj',
    'stage2_add_gate_proj',
    'stage3_add_down_proj',
]

ALPHA = 0.6
BETA = 0.4
TEMP = 2.0
MAX_LEN = 512
BATCH_SIZE = 1
MAX_STEPS = 80
LR = 1e-4


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


def build_train_text(ex: dict) -> str:
    return f"{build_prompt(ex)} {ex.get('output', '').strip()}"


class SFTDataset(Dataset):
    def __init__(self, records: List[dict], tokenizer):
        self.records = records
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        text = build_train_text(self.records[idx])
        tok = self.tokenizer(
            text,
            truncation=True,
            max_length=MAX_LEN,
            padding='max_length',
            return_tensors='pt',
        )
        input_ids = tok['input_ids'][0]
        attention_mask = tok['attention_mask'][0]
        labels = input_ids.clone()
        labels[attention_mask == 0] = -100
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


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


def distill_stage(teacher, student, train_records, tokenizer):
    student.config.use_cache = False
    ds = SFTDataset(train_records, tokenizer)
    dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True)

    opt = torch.optim.AdamW((p for p in student.parameters() if p.requires_grad), lr=LR)
    student.train()
    teacher.eval()

    step = 0
    losses = []

    pbar = tqdm(total=MAX_STEPS, desc='distill', leave=False)
    while step < MAX_STEPS:
        for batch in dl:
            if step >= MAX_STEPS:
                break
            batch_student = {k: v.to(student.device) for k, v in batch.items()}
            batch_teacher = {k: v.to('cpu') for k, v in batch.items()}

            with torch.no_grad():
                t_out = teacher(
                    input_ids=batch_teacher['input_ids'],
                    attention_mask=batch_teacher['attention_mask'],
                )

            s_out = student(
                input_ids=batch_student['input_ids'],
                attention_mask=batch_student['attention_mask'],
                labels=batch_student['labels'],
            )

            ce = s_out.loss

            s_logits = s_out.logits / TEMP
            t_logits = t_out.logits / TEMP

            # shift for next-token KL
            s_shift = s_logits[:, :-1, :].contiguous()
            t_shift = t_logits[:, :-1, :].contiguous().to(s_shift.device)
            m_shift = batch_student['attention_mask'][:, 1:].contiguous().float()

            kl_token = F.kl_div(
                F.log_softmax(s_shift, dim=-1),
                F.softmax(t_shift, dim=-1),
                reduction='none',
            ).sum(dim=-1)

            kl = (kl_token * m_shift).sum() / (m_shift.sum() + 1e-8)
            kl = kl * (TEMP * TEMP)

            loss = ALPHA * ce + BETA * kl

            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            opt.step()

            losses.append(float(loss.item()))
            step += 1
            pbar.update(1)

    pbar.close()
    return {
        'steps': step,
        'loss_last': losses[-1] if losses else None,
        'loss_mean': sum(losses) / len(losses) if losses else None,
    }


def load_teacher(tokenizer):
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float32,
        device_map='cpu',
        low_cpu_mem_usage=True,
    )
    if ADAPTER.exists():
        model = PeftModel.from_pretrained(base, ADAPTER)
        model = model.merge_and_unload()
    else:
        model = base
    model.config.pad_token_id = tokenizer.pad_token_id
    return model


def load_student(stage_dir: Path, tokenizer):
    quant_cfg = None
    if DEVICE == 'cuda':
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4',
        )

    model = AutoModelForCausalLM.from_pretrained(
        stage_dir,
        torch_dtype=DTYPE,
        device_map='auto' if DEVICE == 'cuda' else None,
        low_cpu_mem_usage=True,
        quantization_config=quant_cfg,
    )

    model.config.pad_token_id = tokenizer.pad_token_id
    model.gradient_checkpointing_enable()
    if DEVICE == 'cuda':
        model = prepare_model_for_kbit_training(model)

    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        inference_mode=False,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj'],
    )
    model = get_peft_model(model, lora_cfg)
    return model


def main():
    (PDQ_DIR / 'reports').mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    train_data = load_json(TRAIN_JSON)
    test_data = load_json(TEST_JSON)

    teacher = load_teacher(tokenizer)
    teacher.eval()

    results = []

    for stage in STAGES:
        stage_dir = PRUNE_ROOT / stage
        out_stage_dir = OUT_ROOT / stage
        out_stage_dir.mkdir(parents=True, exist_ok=True)

        student = load_student(stage_dir, tokenizer)

        before = evaluate(student, tokenizer, test_data)
        distill_stats = distill_stage(teacher, student, train_data, tokenizer)
        after = evaluate(student, tokenizer, test_data)

        student.save_pretrained(out_stage_dir)
        tokenizer.save_pretrained(out_stage_dir)

        results.append(
            {
                'stage': stage,
                'before': asdict(before),
                'after': asdict(after),
                'delta_accuracy': after.accuracy - before.accuracy,
                'distill': distill_stats,
                'artifact_dir': str(out_stage_dir),
            }
        )

        del student
        torch.cuda.empty_cache()

    report = {
        'method': 'Per-stage distillation after iterative Wanda 2:4 pruning',
        'distill_hparams': {
            'alpha_ce': ALPHA,
            'beta_kl': BETA,
            'temperature': TEMP,
            'max_steps': MAX_STEPS,
            'lr': LR,
            'batch_size': BATCH_SIZE,
            'max_len': MAX_LEN,
        },
        'results': results,
    }

    with REPORT_JSON.open('w') as f:
        json.dump(report, f, indent=2)

    lines = ['# Iterative Distillation Report', '']
    lines.append(f"- **Method**: {report['method']}")
    lines.append(f"- **Output root**: `{OUT_ROOT}`")
    lines.append('')
    lines.append('## Hyperparameters')
    for k, v in report['distill_hparams'].items():
        lines.append(f"- {k}: {v}")
    lines.append('')
    lines.append('## Stage Results')

    for r in results:
        lines.append(f"### {r['stage']}")
        lines.append(f"- Before: {r['before']['correct']}/{r['before']['total']} = {r['before']['accuracy']:.4f}")
        lines.append(f"- After: {r['after']['correct']}/{r['after']['total']} = {r['after']['accuracy']:.4f}")
        lines.append(f"- Delta: {r['delta_accuracy']:+.4f}")
        lines.append(f"- Distill steps: {r['distill']['steps']}")
        lines.append(f"- Mean loss: {r['distill']['loss_mean']:.6f}")
        lines.append(f"- Artifact: `{r['artifact_dir']}`")
        lines.append('')

    with REPORT_MD.open('w') as f:
        f.write('\n'.join(lines))

    print(f'Report JSON: {REPORT_JSON}')
    print(f'Report MD: {REPORT_MD}')


if __name__ == '__main__':
    main()
