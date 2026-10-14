# Bank matched to ASI participating parameter inventory

This is a new experiment, not a replacement for the historical 124M-total bank.
Each of eight dense specialists has 93,850,112 parameters; collection total
750,800,896. The historical bank has 15,496,576 per response.

ASI inventory: common/shared 89,885,184 + selected routed 4,055,040 =
93,940,224, including 90,112 gate weights bypassed by uniform_pool.
Subtract those unused gates: 93,850,112. Full embedding tables count toward
this inventory; it is not a count of scalar weights read for a single token.
The external classifier is excluded and must be measured separately.

Dense design preserves ASI's 12 layers, width 512, vocabulary, attention and
context. All 12 FFNs are dense with intermediate width 568. Their aggregate
FFN width matches ASI's first dense FFN plus eleven selected/shared branches:
12*568 = 1536 + 11*(2+2)*120. Allocation across layers differs; nonlinear
mixtures, FLOPs and routing behavior are not equivalent.

Each specialist still trains only on its category. Existing bank recipes'
steps, random sampling and token budgets are retained, approximating expected
category exposure under ASI sqrt sampling, not its realized batch history.
ASI's shared weights saw all categories; each independent backbone does not.
A generalist dense model trained on all categories would be a separate control.
No checkpoint from the small bank should be resumed into this architecture.

From repository root on homelab:

```bash
bash configs/bank_active_matched_v1/train_homelab.sh results/dataset_trial --dry-run
bash configs/bank_active_matched_v1/train_homelab.sh results/dataset_trial
```

Use the actual dataset path. A leading /results means the filesystem root.
Microbatch 2/context 1024, accumulation to 32768 tokens; intended for 16GB GPU,
not GPU-tested here. Eight models train sequentially. Automatic retries,
checkpoint every 25 steps, retention of 3 and resume are enabled. Dry-run
validates recipes but does not check dataset availability or actual GPU fit.
After training, audit and run the same suite with the new config directory;
preserve the old bank as the total-storage-matched comparison.
