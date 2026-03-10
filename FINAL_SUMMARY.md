# Medical LLM PDQ Final Summary

## Scope
Completed end-to-end compression workflow on the medical LLM checkpoint:
1. **Pruning** (iterative Wanda-style 2:4)
2. **Distillation** (per pruning stage)
3. **Quantization** (FP16 vs INT8 vs NF4)

---

## 1) Pruning

### Method chosen
**Wanda-style activation-aware 2:4 semi-structured pruning** on MLP layers (`up_proj`, `gate_proj`, `down_proj`).

### Why this method
- Activation-aware importance is more LLM-safe than plain magnitude pruning.
- 2:4 structured sparsity gives practical efficiency potential on NVIDIA stacks.
- Better controllability for stage-wise pruning and recovery.

### Iterative pruning results
Source: `reports/iterative_pruning_metrics.md`

- **Baseline**: 74/100 = **0.7400**, sparsity 0.00%
- **Stage 1 (up_proj)**: 77/100 = **0.7700**, sparsity 12.97%
- **Stage 2 (+ gate_proj)**: 68/100 = **0.6800**, sparsity 25.95%
- **Stage 3 (+ down_proj)**: 61/100 = **0.6100**, sparsity 38.92%

### Memory difference before vs after pruning
(Estimated **effective weight memory** from non-zero params at FP16: 2 bytes/param)

- **Before pruning** (non-zero params: 7,241,732,092): ~**13.49 GiB**
- **After pruning Stage 3** (non-zero params: 4,423,159,806): ~**8.24 GiB**
- **Difference**: ~**5.25 GiB reduction** (~38.9%)

> Note: this is effective memory implied by sparsity. Actual runtime memory may be higher unless sparse-aware kernels/storage are used.

Observation: quality drops at deeper sparsity stages without recovery.

---

## 2) Distillation

### Strategy
Teacher = original merged medical checkpoint, student = each pruned stage.
Loss: `0.6 * CE + 0.4 * KL(T=2.0)`.

### Distillation results
Source: `reports/iterative_distillation_metrics.md`

- **stage1_up_proj**: 0.7200 -> **0.7800** (+0.0600)
- **stage2_add_gate_proj**: 0.6700 -> **0.7400** (+0.0700)
- **stage3_add_down_proj**: 0.6200 -> **0.8000** (+0.1800)

Observation: distillation recovered and improved accuracy substantially, especially for the most pruned stage.

---

## 3) Quantization

### Note
AWQ package was not available in environment, so quantization used bitsandbytes baselines (INT8/NF4).

### Quantization results (on distilled stage3)
Source: `reports/quantization_step_metrics.md`

- **FP16**: accuracy 0.7900, memory 13.514 GB
- **INT8**: accuracy 0.8100, memory 7.040 GB (47.91% less)
- **NF4 (4-bit)**: accuracy 0.8100, memory 3.790 GB (71.96% less)

---

## Final Recommended Checkpoint

**Primary deployment candidate:**
- Quantized mode: **NF4 (4-bit)**
- Base: `artifacts/iterative-wanda-2of4-distilled/stage3_add_down_proj`

Why:
- Best memory efficiency (~72% reduction vs FP16)
- No observed accuracy loss vs INT8, and +0.02 vs FP16 on current 100-sample test

---

## Files Produced

### Plans / scripts
- `PLAN.md`
- `run_pruning_step.py`
- `run_iterative_pruning.py`
- `run_iterative_distillation.py`
- `run_quantization_step.py`

### Reports
- `reports/pruning_step_metrics.md`
- `reports/pruning_step_metrics.json`
- `reports/iterative_pruning_metrics.md`
- `reports/iterative_pruning_metrics.json`
- `reports/iterative_distillation_metrics.md`
- `reports/iterative_distillation_metrics.json`
- `reports/quantization_step_metrics.md`
- `reports/quantization_step_metrics.json`

### Artifacts
- `artifacts/pruned-wanda-2of4/`
- `artifacts/iterative-wanda-2of4/`
- `artifacts/iterative-wanda-2of4-distilled/`

---

## Caveats
- Current evaluation set is small (100 examples); results may have variance.
- Medical deployment requires expanded safety/hallucination validation before real-world use.
- AWQ can still be added later for direct AWQ-vs-NF4 comparison.
