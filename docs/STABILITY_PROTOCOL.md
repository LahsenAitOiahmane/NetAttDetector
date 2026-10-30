# Development-only measurement, ablation and stability protocol

Registered 30 October 2026 before this phase's fits. The prior CSE result is known and failed;
CSE is excluded from this phase's inputs, selection and metrics. UNSW and ToN are development
data. This study can diagnose stability and rank research directions; it cannot establish fresh
source generalization. No further independent source is consumed by this study.

1. Reuse the checksummed corrected development cache. Record source/cached-cohort provenance.
   Report zeros, nonfinite values and range issues by source, label and IP protocol. Zero TTLs
   remain recorded values: without PCAP/exporter evidence we cannot call them missing values
   or silently reinterpret them. Supplied dictionary agreement is not actual exporter parity.
2. Create group-isolated partitions at seeds 7, 42 and 101 using the common 30 numeric fields
   retained by every ablation (the prior 34 grouping fields minus four exporter-rate fields).
   This conservative regrouping changes partitions from the prior experiment. Retain ambiguous
   groups and record their sizes; do not remove difficult observations. Sort each partition
   by its full numeric fingerprint, so selected rows and input order can be checked separately.
   Record ordered feature/label/weight digests, raw membership digests, and post-preprocessing
   training/probe digests. Retain source identity as metadata only.
3. Make training weights directly from source counts, normalized to equal source contribution
   and cast to float32. Materialize matrices in C order, including after preprocessing. These
   controls are explicit changes from the prior experiment, not retroactive corrections to it.
4. On the identical seed-42 partition, fit XGBoost controls twice for each of: original 0.8
   row/column sampling with one/two threads, and full row/column sampling with one/two threads.
   Use 100 trees for the control. Persist scores, order/payload digests, stable tree hashes and
   repeat differences. Comparing identical digests across platforms can localize a difference
   to fitting; matching only sorted row membership cannot establish identical numerical inputs.
5. Compare eight feature policies separately: full, no TTL, no ports, no protocol/application
   IDs, no four exporter-rate fields, no TTL/ports, no TTL/exporter-rate fields, and the previous
   six-field reduction. They all retain the common 30 grouping fields, keeping identical raw
   ablated observations in a single partition. No feature-family removal is claimed
   to be beneficial in advance. Do not delete ambiguous reduced observations to improve scores.
6. For each policy, fit histogram boosting and full-sampling XGBoost at three partition seeds:
   at most 48 classifier fits plus eight controls and four transfer diagnostics. Use 150 iterations, two CPU threads, no
   histogram early-stopping split, and equal-source weights. Do not search extra parameters.
7. Choose each threshold on the calibration partition using the existing per-source precision,
   FPR confidence and support constraints. Evaluate that frozen threshold on validation. This
   phase does not optimize a threshold directly on its reported validation observations.
8. Report source-wise point metrics, conditional Wilson bounds, per-family recall, feasibility,
   and transformed-input overlaps. Reject contaminated candidate evaluations. A development
   screen requires precision >=0.98, precision lower bound >=0.95, FPR upper bound <=0.005,
   >=25 alerts and recall >=0.80 in every source at every registered partition seed. This recall
   requirement avoids promoting a detector that meets precision constraints by missing attacks.
9. Stop starting new fits after one search hour; finish/record the active fit. Rank only complete
   three-seed results; mark incomplete studies explicitly. No deployment/promotion occurs from
   this development screen. All old failures and frozen artifacts remain unchanged.
10. On seed 42, compare full/no-TTL XGBoost in both source-transfer directions. Train and
    choose the threshold using only the originating source's training/calibration partitions;
    score the other known development source's validation partition without target fitting.
    These four diagnostics are not independent new-source evidence. They are registered before
    fits together with the other controls and share the total search deadline.

The 80% recall level is a research screen, not an operator-approved production SLO. Binomial
intervals remain conditional on independent labeled rows. Source/capture dependence and the
unknown accuracy of benchmark labels remain limitations. Source/host/day isolation and a new
independent source are required before making deployment claims.

Primary references: [nProbe information-element definitions](https://www.ntop.org/guides/nprobe/flow_information_elements.html)
and [XGBoost's numerical nondeterminism discussion](https://xgboost.readthedocs.io/en/stable/faq.html).
These explain intended fields and possible causes; they do not prove the reason for our observed
cross-platform discrepancy or the meaning of zeros in these particular published captures.
