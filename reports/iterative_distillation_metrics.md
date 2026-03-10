# Iterative Distillation Report

- **Method**: Per-stage distillation after iterative Wanda 2:4 pruning
- **Output root**: `/home/zz79jk/clawd/medical-llm-pdq/artifacts/iterative-wanda-2of4-distilled`

## Hyperparameters
- alpha_ce: 0.6
- beta_kl: 0.4
- temperature: 2.0
- max_steps: 80
- lr: 0.0001
- batch_size: 1
- max_len: 512

## Stage Results
### stage1_up_proj
- Before: 72/100 = 0.7200
- After: 78/100 = 0.7800
- Delta: +0.0600
- Distill steps: 80
- Mean loss: 1.040849
- Artifact: `/home/zz79jk/clawd/medical-llm-pdq/artifacts/iterative-wanda-2of4-distilled/stage1_up_proj`

### stage2_add_gate_proj
- Before: 67/100 = 0.6700
- After: 74/100 = 0.7400
- Delta: +0.0700
- Distill steps: 80
- Mean loss: 1.145548
- Artifact: `/home/zz79jk/clawd/medical-llm-pdq/artifacts/iterative-wanda-2of4-distilled/stage2_add_gate_proj`

### stage3_add_down_proj
- Before: 62/100 = 0.6200
- After: 80/100 = 0.8000
- Delta: +0.1800
- Distill steps: 80
- Mean loss: 1.258154
- Artifact: `/home/zz79jk/clawd/medical-llm-pdq/artifacts/iterative-wanda-2of4-distilled/stage3_add_down_proj`
