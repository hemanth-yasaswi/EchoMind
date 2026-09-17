# EchoMind Experiment Log

## E00 - Baseline

Date: 15 Sep 2026

Branch: master

Objective:
Establish a working CNN baseline.

Changes:
- 3-layer CNN
- LR = 1e-4
- SpecAugment = 12x12
- Epochs = 100

Results:
Train Accuracy: 29.8%
Validation Accuracy: 39.4%
Test Accuracy: 36.25%

Conclusion:
Working end-to-end pipeline.



## E01 - Reduced SpecAugment

Date: 15 Sep 2026

Branch: model-improvements

Hypothesis:
Large masks remove too much information.

Changes:
- SpecAugment 12x12 → 8x8

Everything else unchanged.

Results:
Train Accuracy: 34.8%
Validation Accuracy: 41.6%
Test Accuracy: 40.63%

Difference from baseline:
+4.38%

Conclusion:
Hypothesis supported.



## E02 — Learning Rate 3e-4

Status: Completed

Hypothesis:
A higher learning rate will allow faster convergence without hurting generalization.

Changed:
- Adam learning rate: 1e-4 → 3e-4

Constants:
- SpecAugment: 8×8
- Epochs: 100
- Batch size: 32
- Label smoothing: 0.1

Results:
- Train Accuracy: 50.47%
- Validation Accuracy: 56.25%
- Test Accuracy: 44.00%

Difference from E01:
+3.37%

Conclusion:
The higher learning rate improved convergence and produced the best test accuracy so far, supporting the hypothesis.



## E03 — Reduced Dropout

Status: Completed

Hypothesis:
Reducing dropout from 0.5 to 0.4 will retain more useful feature representations while maintaining regularization.

Changed:
- Dropout: 0.5 → 0.4

Constants:
- Learning rate: 3e-4
- SpecAugment: 8×8
- Epochs: 100
- Batch size: 32
- Label smoothing: 0.1

Results:
- Train Accuracy: 47.27%
- Validation Accuracy: 54.06%
- Test Accuracy: 44.75%

Difference from E02:
+0.75%

Conclusion:
Dropout reduction produced only a marginal improvement, suggesting model capacity is no longer the primary bottleneck.



## E04 — Add Residual Block

Status: Completed

Hypothesis:
A residual block will improve feature learning and gradient flow, leading to better generalization.

Changed:
- Added one residual block after the 128-channel convolution stage.

Constants:
- Learning rate: 3e-4
- SpecAugment: 8×8
- Dropout: 0.4
- Epochs: 100
- Batch size: 32
- Label smoothing: 0.1

Results:
- Train Accuracy: 68.91%
- Validation Accuracy: 69.69%
- Test Accuracy: 57.75%

Difference from E03:
+13.00%

Conclusion:
The residual block produced the largest improvement of the project so far, confirming that architecture was the primary bottleneck rather than hyperparameter tuning.



