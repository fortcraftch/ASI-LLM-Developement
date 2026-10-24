# Colibri / AirLLM OLMoE pilot

Run: `python asi/experiments/colibri_pilot.py` (reuses completed AirLLM results after checking prompt IDs and output length; do not
launch while another copy is active). Output: results/colibri_airllm_20261014.
Uses existing public_models/olmoe checkpoint and tokenizer; unrestricted routing.
Three identical chat inputs, greedy fixed16 generation, batch1. AirLLM first,
then Colibri CPU and Vulkan; models execute sequentially. This is an integration
pilot, not a repeated/precision-matched performance benchmark.

Colibri source reviewed at bf2442915d6e3dd4cdfd2eb9c2a3d2aa44a25850,
Apache-2.0. Executable is official v2.0.0 Windows release (not built from that
source checkout); archive SHA256 fdb8ea2fad7dcfe44d9d37bfb3e444f1299207f268ab499719f98bd772397d79
matches published release SHA256SUMS. Record release/source distinction.

OLMoE engine uses INT8 experts and FP32 dense weights; AirLLM bridge uses BF16.
Colibri keeps expert cache of eight per layer but may route ANY expert; misses
load experts, unlike ASI's fixed resident candidate restrictions. Prefetch and
hot pinning disabled for pilot. No conversion overwrites original weights.
Colibri elapsed generation includes prefill; compare AirLLM generation_seconds,
not its decode-only speed. Colibri sampled RSS is not GPU allocation and does
not include OS file cache. Output agreement is not correctness. No disk-cache
flush, timing replication or end-to-end quality suite yet.

Before publication: verify engine GPU activation in logs, repeat with warmup
and randomized execution order, audit checkpoint identity and generation,
measure VRAM/transfer bytes consistently, and add a quantization-matched
control before attributing gains to execution engine alone. Do not substitute
Colibri for the project architecture until semantic/numerical compatibility
has been tested. Sources: https://github.com/JustVugg/colibri and its c/olmoe.c.

## Conversion retry (14 October 2026)

AirLLM completed all three prompts. The first Colibri CPU attempt failed before
generation: the original BF16 checkpoint lacks the required `merged_weight`
tensors. Its log is preserved. The runner now invokes the upstream merged INT8
converter into `public_models/olmoe_colibri_int8`, with 32-tensor output buffers
and four CPU threads, before running CPU then Vulkan. Original weights remain
intact. Conversion has its own log and source hashes; it is preparation time,
not generation time. No Colibri performance conclusion is available until the
converted model runs successfully.

## Completed pilot

All 3 AirLLM and 6 Colibri generations completed and passed prompt/token-count
checks. Vulkan GPU execution is confirmed; CPU and Vulkan tokens match on all
three inputs. See `results/colibri_airllm_20261014/REPORT.md`,
`validation.json` and `generation_comparison.png` for outputs and timings.
Important correction: cache=8/layer controls CPU caching. Vulkan automatically
budgets up to 1,024 experts, promoting them during execution; this is not an
8-resident-per-layer GPU experiment. Timings are exploratory and confounded by
precision, residency, process lifetime and OS cache. No architecture substitution
or quality-equivalence claim follows from this pilot. Longer repeated and
precision-matched measurements remain necessary for publication.

## Memory-equivalence audit (14 October 2026)

Converted safetensors total 7,412,882,416 bytes (6.90 GiB); original BF16
weight shards total 13,838,721,960 bytes (12.89 GiB). Physical host RAM is
17,100,144,640 bytes (15.93 GiB). Thus the converted file set fits nominally
in RAM while leaving much more headroom than BF16. Available RAM measured
after the pilot was 2,320,285,696 bytes; this is NOT a measurement during runs.

CPU process RSS (~2.54 GiB) rules out all converted weights simultaneously
resident in that process's working set. It does NOT rule out the full file
set in Windows system cache. The reviewed source uses st_read_raw for expert
misses, EXPERT_DROP defaults to zero, and Windows DONTNEED is a no-op in
compat.h. The source checkout and released binary are distinct: this explains
a plausible cache path, not an instrumented proof of every binary disk read.
The logged 'disk loads' are logical expert misses, not physical SSD reads.

Vulkan also benefits from a larger GPU tier: 576/608/655 experts resident at
exit out of 1,024, budget permitting all 1,024, with zero evictions. Reported
dense weights are 1,425 MiB, expert-tier component peaks 3,850â€“4,337 MiB.
These components are not a full-device peak measurement. AirLLM allocated
VRAM is about 815 MiB, so GPU residency budgets were not equivalent either.

Conclusion: integration and observed wall time are valid; out-of-core speed
advantage under equivalent memory pressure is NOT established. Do not use
this pilot as evidence that Colibri is faster when neither model fits RAM.
A proper next experiment needs equal quantization, capped GPU expert tier,
and a host/VM RAM limit including page cache (with swap disabled or measured),
plus physical disk I/O, process RSS and device memory sampling. Limiting only
process RSS, setting RAM_GB or shrinking expert cache cannot enforce the OS
page-cache limit. Do not create host-wide memory pressure to simulate it.

## Quality follow-up

Launched separate 64-token diagnostic in results/colibri_quality_20261014.
See its PROTOCOL.md for predeclared review criteria and ASI port requirements.
This run completed; see the completed 64-token diagnostic below. Do not infer quality equivalence from the integration pilot.

## Next authorized control: equivalent quantization

Requested by Marc after launching the quality run. Do not overlap GPU jobs.
Use identical quantized tensor values, scales and grouping, not merely two
formats both called INT8. First establish a reference that can consume Colibri's
rowwise INT8 expert weights with the same FP32 dense computation; document any
dequantization or arithmetic differences. AirLLM BF16 versus Colibri INT8 remains
a mixed-precision comparison, even if reconstructed INT8 weights are cast to BF16.
Verify logits/tokenization/routing before timing. Equalize explicit RAM/VRAM
budgets and account for OS file cache; separate model preparation, loading,
prefill and decode. Repeat measurements and report dispersion. If AirLLM cannot
support this format faithfully, state the limitation and use a compatible
reference rather than describing incompatible INT8 methods as equivalent.
Integration with ASI remains conditional on correctness and measured benefit.

## Completed 64-token quality diagnostic

Results: results/colibri_quality_20261014/REPORT.md and validation.json.
All nine generations completed without EOS. CPU/Vulkan INT8 outputs match exactly
on the three prompts. No obvious repetition loops; two-sentence compliance fails,
all outputs truncate, and AirLLM adds an incorrect general derivative rule after
answering 2x correctly. Colibri's general rule is unfinished. Python sorting is
appropriate by static inspection; no generated code was executed. These cases do
not establish quality equivalence. Quantization-matched control remains pending.


## Completed Vulkan categorization timing screen

Same INT8 weights and patched Vulkan build, top8, fixed 2 GiB expert tier.
27 randomized repetitions plus six controls completed; all64 token equivalence
and route masks verified. Median speedups versus free routing: 32 candidates
1.99–2.53x; 16 candidates 4.07–5.57x across three prompts. This supports pursuing
recovery under this regime, not a quality-equivalent performance claim.
See results/colibri_candidates_vulkan_20261014/REPORT.md and timing.png.
CPU-only timing experiment was cancelled and deleted, not used as evidence.

## Review on 19 October 2026

The unrestricted/32/16 same-build test already compares Colibri alone against
Colibri plus category restriction. All 33 archived log/trace hashes, route masks
and all64 equality were rechecked. No new timing run is claimed. Temporal benefit
is observed under the 2 GiB expert-tier cap; quality-equivalent benefit remains
unproven. The paper now includes cache counters. The integration README is updated.
