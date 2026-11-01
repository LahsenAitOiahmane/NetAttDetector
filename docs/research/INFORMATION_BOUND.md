# Registered connection information bound

Registered 1 November 2026 after the fixed-thread statistical study, before this bound is
computed. This is NAD-051: a **no-fit diagnostic** on already observed calibration/validation
captures, not a new model, threshold selection, fresh-test result or prediction of future quality.

## Why the bound is valid

Current detectors score each flow using its measured features and flag a UID when **any**
accepted segment crosses a fixed threshold. Thus UIDs with the same **set** of feature vectors
must receive the same decision. Repetition/order of an identical vector cannot affect a maximum.

Group accepted UIDs by exact vector sets, separately for each partition, feature policy and
float64/float32 representation. Use full byte equality, not hash digests, to avoid merging
different vectors through a hash collision. Each group has a benign cost and attack benefit if
flagged. Pure-attack groups are free in an optimistic relaxation; pure-benign groups need not
be flagged. Solve a 0/1 knapsack over mixed-label groups, maximizing detected attack UIDs under
floor(0.005 * benign_UIDs). Use exact rational arithmetic for the budget and ceil(0.8 * attacks).

This relaxation lets every distinct vector-set group choose independently, even when groups
share individual vectors. Actual flow-score/max classifiers may have more constraints. It also
ignores precision constraints. Therefore the optimum is an **upper bound**, not a deployable
classifier or proof that a passing operating point can be learned. If even this optimistic
recall is below 80%, no deterministic detector in the current representation/max family can
meet both recall/FPR requirements on these exact observed accepted UIDs. More expressive
aggregation/context or additional measurements change that family and are outside this bound.

## Fixed scope and output

Compare core11, statistics27 and no-port statistics25, each in float64 and float32, on the
same accepted calibration/validation partitions. Validate core/sidecar identities before reading;
keep exclusions and all conflicting rows unchanged. No external source is accessed. Report
counts of indistinguishable/mixed groups, optimistic maximum recall at the registered FPR
budget, minimum benign flags needed for 80% recall in the relaxation, and aggregate frontier
breakpoints. Do not publish signature contents, feature/UID rows or oracle decisions.

The bound describes observed candidate labels, not attack truth or independent-capture
uncertainty. A float32 result applies to the float32 representation; do not transfer it to
float64 without checking that representation separately. Additional model depth cannot restore
information removed by a feature policy, but a passing bound is not evidence of good transfer.

Process only the two small evaluation partitions; do not load training rows unnecessarily.
Cap each partition at 250k rows, dynamic-program updates at ten million, and the analysis at
600 seconds / 512 MiB sampled process-tree RSS (excluding acquisition/preparation/verification).
Fail bounds rather than publishing an incomplete approximation. No fitting, promotion, live
capture or release of private data/models occurs.
