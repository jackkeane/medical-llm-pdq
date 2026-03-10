# Iterative Pruning Report

- **Method**: Iterative Wanda-style activation-aware 2:4 semi-structured pruning
- **Artifact root**: `/home/zz79jk/clawd/medical-llm-pdq/artifacts/iterative-wanda-2of4`

## Baseline
- Accuracy: 74/100 = 0.7400
- Non-zero params: 7,241,732,092 / 7,241,732,096
- Global sparsity: 0.0000%

## Stages
### stage1_up_proj
- Include keys: ['up_proj']
- Newly pruned modules: 32
- Stage sparsity (considered): 939,524,096 / 1,879,048,192 = 50.0000%
- Accuracy: 77/100 = 0.7700
- Accuracy delta vs baseline: +0.0300
- Non-zero params: 6,302,207,996
- Global sparsity: 12.9737%

### stage2_add_gate_proj
- Include keys: ['up_proj', 'gate_proj']
- Newly pruned modules: 32
- Stage sparsity (considered): 939,524,096 / 1,879,048,192 = 50.0000%
- Accuracy: 68/100 = 0.6800
- Accuracy delta vs baseline: -0.0600
- Non-zero params: 5,362,683,900
- Global sparsity: 25.9475%

### stage3_add_down_proj
- Include keys: ['up_proj', 'gate_proj', 'down_proj']
- Newly pruned modules: 32
- Stage sparsity (considered): 939,524,094 / 1,879,048,192 = 50.0000%
- Accuracy: 61/100 = 0.6100
- Accuracy delta vs baseline: -0.1300
- Non-zero params: 4,423,159,806
- Global sparsity: 38.9212%
