# Sudoku TRM scaling — one experiment per job

Target directory: `/work/p26037/p26037kshk/trm-sudoku-scaling`.
Hardware: **one Kairos ARM node, four GH200 GPUs**. Every training submission runs **one configuration**. It never starts a sweep or a successor job automatically.

## 1. Install for the new account

Copy `trm-sudoku-scaling.zip` to `/work/p26037/p26037kshk`, then run on the internet-connected login node:

```bash
cd /work/p26037/p26037kshk
unzip -n trm-sudoku-scaling.zip
cd trm-sudoku-scaling
mkdir -p logs
export TRM_PYTHON_MINOR=312
bash scripts/download_wheels.sh
python3 scripts/repro_data.py download-sudoku
sbatch slurm/setup.sbatch
```

Wait for the setup job to succeed. It creates **this account's own** `.venv-arm` inside the shared ARM container, builds the native CUDA optimizer, and prepares Sudoku. It does not use another user's Python installation. A login-node `.venv-download` is only for downloads; do not activate it in GPU jobs.

Required cluster facilities: `apptainer`, Slurm account `p26037`, partition `gpu`, `gpu:gh200` resources and read access to `/work/conteneurs/shared/AI/nemo_25.04.03_arm.sif`. The container supplies Python 3.12, CUDA, PyTorch and Triton. `configs/cluster.env.sh` sets the paths; change `TRM_CONTAINER` there if the administrator provides a different compatible image. Torch/CUDA are not replaced by downloaded wheels. If the new user lacks the account allocation or filesystem permissions, those must be granted by the cluster administrator.

A successful build creates `data/paper/sudoku-extreme-1k-aug-800/DATA_READY.json`. An interrupted build remains in the adjacent `.building` directory and is never accepted as ready; inspect that directory before deleting it and retrying setup.

## 2. Profile once to establish a common batch and training budget

```bash
sbatch slurm/profile.sbatch
```

This is hardware calibration, not a completed scientific experiment. It submits no training jobs. The profile uses fresh, discarded model weights and:

- Tests global batches 384, 768, 1536, 3072, 6144 and 8192, stopping growth at failure or less than **20% memory headroom**; tries 192 if necessary. `--max-batch` limits the search.
- Profiles the largest untied model, then checks the selected batch on the deepest native cases, tied full-gradient control and preliminary configuration.
- Measures 20 warm-up updates, 40 timed native optimizer/EMA updates, a diagnostic update, and three timed full 16-step EMA evaluation batches after one warm-up batch. Training model, optimizer and EMA remain resident during evaluation.
- Chooses the fastest tested safe batch by projected training plus evaluation time. A larger batch need not be faster. This is a bounded search, not proof of a global optimum or a guarantee against every OOM.
- Keeps **one batch and one epoch budget** for the planned axes. Keeps 50,000 epochs if they fit; otherwise reduces the common epoch count, rounded to a multiple of 10. It retains ten evaluations and at least 4,000 optimizer updates, including 2,000 warm-up updates. It fails rather than creating a plan that misses these checks.

The result is `configs/plan.json`. It contains the selected batch, epochs, projected times and profile provenance. Estimates include a 25% compute allowance and target **10 hours**, leaving two hours inside the 12-hour job for startup, checkpoints and uncertainty. They exclude queue time. GPU capture, I/O and learned halting can change actual runtime; the first completed run is the stronger timing measurement. No measured GH200 runtime is bundled with this code.

The default profiling bounds are k ≤ 6 at T=3 for axes 1/3, and k=3 with T ≤ 4 for axis 2. To preselect another fixed k for axis 2 before profiling:

```bash
sbatch slurm/profile.sbatch --axis2-k 4
```

The profiler refuses to replace an existing plan. Preserve the plan while collecting an axis. Changes to width, k/T beyond these bounds, batch or compilation mode invalidate its memory/time estimate.

## 3. Run one experiment at a time

Start with the reduced-depth preliminary experiment (width 256, **k=1 → native N=2**, T=3):

```bash
sbatch slurm/train.sbatch --axis preliminary --k 1 --t 3
```

Then run each desired configuration separately. These are examples, not a script that starts the whole sweep:

```bash
# Axis 1: vary paper-table k over 1, 2, 4, 6; T remains 3.
sbatch slurm/train.sbatch --axis 1 --k 2

# Axis 2: vary T over 1, 2, 3, 4; keep k=3 (native N=6).
sbatch slurm/train.sbatch --axis 2 --k 3 --t 2

# Axis 3: run each arm separately at the SAME k, T, width, batch and epochs.
sbatch slurm/train.sbatch --axis 3 --k 2 --t 3 --arm tied_full
sbatch slurm/train.sbatch --axis 3 --k 2 --t 3 --arm untied_full
```

For axis 3, repeat the pair for k=1, 2, 4, 6. There are 16 unique planned runs: four axis-1 runs, four axis-2 runs, and eight axis-3 runs. The preliminary run is reused as axis-1 k=1.

All settings are available through argparse; inspect them with:

```bash
python3 scripts/launch.py --help
python3 scripts/launch.py --axis 1 --k 4 --dry-run
```

Explicit override example (also works without a profile plan):

```bash
sbatch slurm/train.sbatch --axis 1 --k 4 --t 3 --width 256 --batch 768 --epochs 50000 --mode reduce-overhead
```

Overrides bypass the profiled runtime/memory assumptions. Keep batch and epochs identical within an axis to attribute differences to k/T or weight sharing. Changing these arguments creates a distinct run identity. Seed is fixed to 0. The launcher cannot launch multiple configurations in one invocation.

## 4. Exact experimental settings

The pinned official TRM source is included in `vendor/TinyRecursiveModels`, commit `c01103738605ba39d1430519b1ee0c62f4c707f8`. `scripts/verify_vendor.py` checks every vendored source file. The vendor remains unmodified.

**Data:** `sapientinc/sudoku-extreme`; download revision and raw-file hashes are saved. The native builder selects **1,000 original training puzzles** with NumPy seed 42, and applies **800 augmentations plus the original**, producing 801,000 stored training examples. Digit permutations, transpose and legal row/column permutations are unchanged. Native test split, **no test augmentation**. There is no ARC data in this experiment. The native sampler uses original puzzle groups: 800 instead of 1000 augmentations does **not** reduce the epoch-based update count by 20%.

**Architecture:** native bidirectional attention TRM, hidden width **256 instead of 512**, two attention/SwiGLU blocks, eight attention heads (head dimension 32), expansion 4 with native intermediate-dimension rounding, RoPE, BF16 forward. Puzzle embedding dimension is also 256; prefix length stays 16. The 81 Sudoku cells plus prefix give sequence length 97. Learned initialization states, input injection, normalization and output/halting heads are native. ACT maximum 16 supervision steps, exploration probability 0.1, `no_ACT_continue=True`. Evaluation always uses all 16 steps.

**Notation:** the command-line `--k` is the paper table's k. Native `L_cycles=N=2k`; `--t` is `H_cycles=T`. Executed attention-block depth per supervision step is **2T(2k+1)**. At T=3, k=1/2/4/6 therefore gives depth 18/30/54/78. ACT's 16-step cap is separate from T.

**Optimization:** native AdamATan2 for dense parameters, LR 1e-4, betas (0.9, 0.95), weight decay 1.0. Native distributed sparse SignSGD for puzzle embeddings, LR 1e-4 and weight decay 1.0. Linear LR warm-up over 2,000 optimizer updates; subsequent LR constant (`lr_min_ratio=1`). EMA rate 0.999. Native stablemax token-mean cross entropy plus 0.5 × halting loss. No added gradient clipping, accumulation or activation checkpointing. Global batch and epoch count come from the common plan; no LR scaling is applied when batch changes. The official starting schedule is 50,000 epochs, evaluation every 5,000; a shortened plan evaluates every `epochs/10`.

**Compilation:** regular updates request `torch.compile(mode='reduce-overhead')`; `--mode default` and `--mode eager` are available. The wrapper copies outputs that must survive CUDA-graph iterations and handles EMA model copies. Diagnostic updates run eagerly on the same weights and optimizer. These changes have overhead; CUDA graphs may be skipped by PyTorch, and no 2× training speedup is assumed.

## 5. What each axis tests

**Axis 1:** native truncated-gradient TRM, N=2k varies; T=3, two blocks and width 256 stay fixed. **Axis 2:** native TRM, T varies at fixed k=3 unless explicitly changed. Choosing “best k” using test accuracy would leak test information: use the prespecified k=3, or establish a separate validation protocol before selecting another k. The code never automatically selects k from test results.

**Axis 3:** same width, same native attention blocks, same z/y input-injection graph and same executed depth. `tied_full` reuses two blocks; `untied_full` gives each of the T(N+1) calls its own two blocks. Untied copies start with identical values but independent storage, so both arms initially compute the same function. Parameter counts differ deliberately and are saved for each run.

Both axis-3 arms backpropagate through **all T cycles**. The official native model uses `no_grad` on the first T−1 cycles; retaining that rule in the untied model would leave early layers untrained. Thus compare the two axis-3 arms with each other. Axis 1/2 remain the native gradient protocol. The control removes inner-block weight sharing; it is a finite unrolled attention graph with TRM's skip connections, not a plain serial Transformer. The outer ACT supervision mechanism and detached carry between supervision steps remain the same in both arms.

These are width-reduced, 800-augmentation, single-seed experiments, potentially with a changed batch and shorter schedule—not an exact reproduction of the paper's accuracy. Report the resolved configuration and final scheduled test accuracy; do not choose the best test checkpoint. One seed provides descriptive comparisons, not seed-level uncertainty or a guaranteed accuracy within ±2% of the paper.

## 6. Checkpoints, accuracy and the three plots

Each run has one directory under `results/`; its name includes k/T/arm/width/batch/epochs/mode/seed. The directory stores resolved hyperparameters, source/data fingerprints, dense/puzzle-table parameter counts, loss and diagnostics logs, ten evaluation records and plots.

Full checkpoints alternate between two atomic generations every 1,000 updates, every 128 evaluation batches and at phase/deadline boundaries. They include model, optimizer, EMA, latent carry, RNG, sampler position and evaluation accumulation. An unexpected kill may lose work since the last committed checkpoint. `COMPLETE.json` appears only after the configured training schedule and all ten evaluations finish.

A deadline saves a checkpoint and exits with pause status 4; **no next job is submitted automatically**. Resubmit the exact same command to resume with four GPUs. Do not change training source/data/config during a run. To request a clean manual pause, create a file named `STOP` in that run's result directory; remove only that flag before resuming.

Diagnostics are taken on updates 1–10 and every 100 updates thereafter:

1. `01_loss.png/pdf`: training LM/objective loss and EMA test loss.
2. `02_latent_norms.png/pdf`: **z and y** gradient RMS and actual forward-state change RMS. These are two latent states; state change is distinct from gradient magnitude. Gradients are normalized to per-example loss, and ACT resets are excluded from the forward-change measurement.
3. `03_layer_norms.png/pdf`: layer gradient RMS and actual optimizer weight-update norm relative to the weight norm. For untied models, the plot shows the median/range across copies of each block position; every individual layer is preserved in `diagnostics.csv`.

Plots are generated after successful completion. To print saved test accuracy (whole-grid exact accuracy):

```bash
python3 scripts/results.py
```

To plot a paused run inside an existing GPU allocation:

```bash
export STUDY_ROOT="$PWD"
source configs/cluster.env.sh
bash scripts/container.sh scripts/repro_plot.py --run results/EXACT_RUN_DIRECTORY
```

You can also plot copied results on any machine with NumPy and Matplotlib. Local CPU tests cover depth matching, gradients, argument mapping, checkpoint/resume and compilation-wrapper behavior. Actual CUDA compilation, memory fit and runtime must be validated by the supplied GH200 profiling job.
