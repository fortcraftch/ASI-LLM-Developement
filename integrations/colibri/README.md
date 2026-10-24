# Colibri + category restriction: completed exploratory timing screen

Reviewed 19 October 2026. The patched Vulkan build and all 33 runs completed:
six unrestricted/all64 controls and 27 timed runs (three prompts, three variants,
three repetitions). Portable w64devkit/Vulkan build setup is resolved.

## Results and scope

Same INT8 weights/build, top8 active, CPU cache8/layer and a 2 GiB GPU expert
tier plus dense weights. Fixed category masks constrain eligibility; Colibri
still manages CPU/GPU execution and placement. The technologies work together.
Median prefill-plus-64-token speedups over unrestricted Colibri: 1.99-2.53x
for 32 candidates and 4.07-5.57x for 16. All64 output equality and all 33 route
traces were verified. At 16 candidates median evictions are zero, but loads
and uploads remain nonzero. This is not a quality-equivalent speedup or proof
that recovery adapters work in Colibri.

Evidence: results/colibri_candidates_vulkan_20261014/REPORT.md, runs.json,
summary.json, validation.json, logs and traces. On 19 October all archived
log/trace hashes and masks were rechecked; this is not a new GPU replication.
The provenance JSON lacks the budget field; logs explicitly record the 2 GiB
cap and 272-expert capacity. Preserve provenance rather than rewriting it.

## Integration contract

Mask text format: `16 64 N` then N distinct expert IDs (0..63) for each of the
16 layers. Existing calibration rankings must supply category selections; never
select from the timed prompt itself or substitute arbitrary IDs as categories.
Set ASI_CANDIDATES to its path for a fresh process per request. Not intended for
multi-request serving with changing categories; reject that usage in the runner.

Validation required before timing: malformed/duplicate/out-of-range masks fail;
all64 equals unrestricted tokens; route traces contain only allowed IDs and eight
unique active experts. Verify CPU and Vulkan both traverse the patched selector.
Fixed candidates constrain execution eligibility, not preload or GPU residency.
Disable history/warm start consistently. Measure promotions and transfers; do not
claim zero swaps without observing it.

## Remaining controls

Compare all-fit and constrained GPU tiers, host-memory limits including OS
file cache, longer generations and category-switch workloads. Separate setup,
selection, prefill and decode; measure RSS, device memory and transfers. Evaluate
quality on fresh prompts. Logical loads are not physical SSD reads. Do not
simulate host limits by creating system-wide memory pressure.

AirLLM BF16 versus Colibri INT8 has different precision and effective memory
budgets. It is an integration pilot, not a controlled engine-speed ranking.
Matched engine comparisons require compatible weights, arithmetic and budgets.

## Historical scripts

run_candidates_vulkan.py uses a fixed historical path and refuses existing runs;
do not relaunch over those results. It lacks the current V2 resume/report contract.
New experiments require an isolated resumable runner and fresh protocol outputs.
report_candidates.py rewrites historical reports; do not run it for read-only
audits. The cancelled/deleted CPU timing batch is not evidence and must not be
relaunched. Keep the original release binary separate from the patched build.
