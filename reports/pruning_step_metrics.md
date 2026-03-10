# Pruning Step Report

- **Method**: Wanda-style activation-aware 2:4 semi-structured pruning (MLP layers)
- **Pruned model**: `/home/zz79jk/clawd/medical-llm-pdq/artifacts/pruned-wanda-2of4`

## Accuracy (yes/no/maybe on medical_test.json)
- Before: 74/100 = 0.7400
- After: 61/100 = 0.6100
- Delta: -0.1300

## Weight Change
- Total params: 7,241,732,096
- Non-zero before: 7,241,732,092
- Non-zero after: 4,423,159,806
- Delta non-zero params: -2,818,572,286
- Global sparsity after: 38.9212%

## Target-layer pruning stats (MLP linear layers)
- Considered params: 5,637,144,576
- Pruned params: 2,818,572,286
- Effective sparsity in considered params: 50.0000%
