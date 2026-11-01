# A concrete no-port information limit

Measured 1 November 2026. **No-port statistical inputs cannot meet the current recall/FPR
research screens on the accepted validation connections.** Even an optimistic relaxation
detects at most **4,203/6,923 attacks (60.71%)** with at most **9/1,904 benign connections
flagged**. Reaching 80% recall requires at least **39 benign flags (2.05% FPR)** in that
relaxation, exceeding the registered 0.5% screen. This is an observed-input bound, not a model
performance estimate or a claim about all future networks.

## Registered proof, inputs and exact computation

Registered [the protocol](../docs/research/INFORMATION_BOUND.md) at `fc71910` before computing
the bound. [Run 41317674415](https://github.com/LahsenAitOiahmane/NetAttDetector/actions/runs/41317674415)
at `faf9e26` passed provenance, exact-update and resource screens with **zero model fits**.
The original core/statistical logical digests remain unchanged; no additional source is used.

A per-flow deterministic classifier followed by UID-max aggregation must make the same decision
for UIDs with identical sets of input vectors. Multiplicity and order of an identical vector
do not change a maximum. We group by full byte/tuple equality, not digest equality, then solve
an exact 0/1 knapsack over mixed-label groups. Benign counts are costs and attack counts are
benefits. Budget uses floor(.005 * benign_UIDs), with ceil(.8 * attack_UIDs) required attacks.

The relaxation permits different feature-set groups to choose independently even if they
share individual vectors, and ignores precision constraints. This is more permissive than
the actual per-flow model family. Failing its ceiling therefore establishes a limit under the
declared representation/accepted-UID assumptions. Passing it **does not** prove that a model
can learn or transfer a qualifying decision rule. No oracle classifier/threshold is produced.

## Results

Float64 and float32 give identical ceilings in all reported cases:

| Partition / inputs | Exact vector-set groups | Mixed-label groups | Benign-flag budget | Optimistic maximum recall | Minimum benign flags for 80% recall |
|---|---:|---:|---:|---:|---:|
| Calibration / core11 | 4,131 | 0 | 10 | 100% | 0 |
| Calibration / statistics27 | 4,131 | 0 | 10 | 100% | 0 |
| Calibration / no-port statistics25 | 100 | 0 | 10 | 100% | 0 |
| Validation / core11 | 7,348 | 0 | 9 | 100% | 0 |
| Validation / statistics27 | 7,350 | 0 | 9 | 100% | 0 |
| **Validation / no-port statistics25** | **1,941** | **12** | **9** | **60.71%** | **39** |

The apparent easy calibration source does not reveal this validation ambiguity. Port-bearing
inputs are not ruled out by this bound, but those results are not evidence for memorizing
ports, validation labels or feature-set lookups. Every actual fitted candidate still failed its
quality gate. These exact finite observations do not supply independent-device/capture or
external-test uncertainty, and candidate-label limitations remain.

## Resource evidence and implications

The analysis reads only the two small evaluation partitions, avoiding unnecessary training
rows and model imports. Exact byte-set grouping/knapsack work took **.39 seconds** and sampled
**139.0 MiB** process-tree RSS. This excludes input acquisition, preparation, imports and
verification. Samples every 50 ms are not an OS-enforced memory bound. The shared read-only
verifier now lives with the lightweight sidecar utilities, preserving model-study behavior.

Only [aggregate bounds](evidence/information-bound/information-bound.json) and
[configuration](evidence/information-bound/configuration.json) are committed. No feature-set
contents, oracle decisions, endpoint/UID values or model artifacts are published. Tests cover
exact knapsack budgets, whole mixed groups, vector multiplicity/order, shared-vector relaxation,
float32/float64 distinctions, invalid groups and refusing incomplete approximations.

The next work must add actual information or change the declared aggregation family before
expecting the no-port target to pass. Prioritize causal traffic history measured before a flow
starts, with explicit same-time, warm-up, late-packet and state-budget behavior; validate online-
style/offline replay parity before fitting. More representative development captures and label
review also remain useful. Do not relax the quality gate or call this bound a successful model.
Reserved independent data, release terms and the owner-deferred pilot remain separate.
See the [runbook](../docs/guides/INFORMATION_BOUND.md).
