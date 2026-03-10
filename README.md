# medical-llm-pdq

Pruning + Distillation + Quantization workflow for the medical LLM checkpoint.

> Original base/adapter weights come from the sibling project: `../medical-llm` (relative path from this repo).

## Upstream Weight Source (important)

This project does **not** train the original medical model from scratch.
It consumes artifacts produced in `medical-llm`:

- Base model: `../medical-llm/models/biomistral-7b`
- Medical adapter: `../medical-llm/outputs/biomistral-medical`
- Data splits: `../medical-llm/data/processed/{medical_train,medical_test}.json`

If those paths are missing, regenerate them in the `medical-llm` project first.

## Environment

Recommended environment:

```bash
conda create -n pdq python=3.12 -y
conda activate pdq
python -V
```

Main dependencies used here:
- `torch`
- `transformers`
- `peft`
- `bitsandbytes`
- `tqdm`

Install example:
```bash
pip install torch transformers peft bitsandbytes tqdm
```

## Project Structure

```text
medical-llm-pdq/
├── PLAN.md
├── FINAL_SUMMARY.md
├── README.md
├── run_pruning_step.py
├── run_iterative_pruning.py
├── run_iterative_distillation.py
├── run_quantization_step.py
├── reports/
└── artifacts/
```

## How to Run

From your workspace root (folder containing both repos):

```bash
cd <your-workspace>
# expected layout:
# <your-workspace>/medical-llm
# <your-workspace>/medical-llm-pdq
```

### 1) (Optional) Single-step pruning sanity run

```bash
python medical-llm-pdq/run_pruning_step.py
```

Outputs:
- `medical-llm-pdq/artifacts/pruned-wanda-2of4/`
- `medical-llm-pdq/reports/pruning_step_metrics.{md,json}`

### 2) Iterative pruning

```bash
python medical-llm-pdq/run_iterative_pruning.py
```

Outputs:
- `medical-llm-pdq/artifacts/iterative-wanda-2of4/`
- `medical-llm-pdq/reports/iterative_pruning_metrics.{md,json}`

### 3) Distillation recovery per pruning stage

```bash
python medical-llm-pdq/run_iterative_distillation.py
```

Outputs:
- `medical-llm-pdq/artifacts/iterative-wanda-2of4-distilled/`
- `medical-llm-pdq/reports/iterative_distillation_metrics.{md,json}`

### 4) Quantization benchmark (fp16/int8/nf4)

```bash
python medical-llm-pdq/run_quantization_step.py
```

Outputs:
- `medical-llm-pdq/reports/quantization_step_metrics.{md,json}`

## Results Summary

See:
- `FINAL_SUMMARY.md`

Includes:
- pruning method choice + rationale
- accuracy trajectory across pruning/distillation stages
- quantization comparison (fp16/int8/nf4)
- memory difference before/after pruning (effective sparse vs dense runtime caveat)

## Notes

- Pruning zeros many weights, but dense tensor storage can still keep similar runtime footprint unless sparse-aware kernels/storage are used.
- For medical use, run broader safety/clinical validation beyond the 100-sample test set before any real deployment.
