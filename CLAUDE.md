# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A grab-bag (`misc`) of two unrelated workstreams. Identify which one a change belongs to before touching anything:

1. **Airgap installer toolchain** (the bulk of the repo). Bash scripts that build offline-installable bundles on a WSL Ubuntu 24.04 box and deploy them on an airgapped Ubuntu 24.04 server with 8× NVIDIA B300 GPUs (Blackwell Ultra, sm_103 / sm_100) + 4th-gen NVSwitch + ConnectX-7/8 + DOCA-OFED. Target is a clean vendor-prepped server.
2. **FEA → HDF5 → GNN data tooling** (Python). `warpage_to_hdf5.py`, `visualize_mesh.py`, `visualize_raster.py`. The HDF5 layout is specified in [DATASET_FORMAT.md](DATASET_FORMAT.md) — treat that doc as the source of truth when changing dataset code.

The two streams share no code. Don't refactor one to look like the other.

## Airgap installer architecture

The installer is split into **two independent bundles**, run in strict sequence on the target. The split is load-bearing: the NVIDIA bundle holds driver+CUDA+FM+NCCL behind `apt-mark hold` so the userland bundle can never accidentally upgrade them.

| Phase | Gather (on WSL, internet) | Preflight (target) | Install (target) | Verify (target) |
|-------|---------------------------|--------------------|------------------|-----------------|
| NVIDIA stack — R580 LTS driver, CUDA 13.0, FabricManager, NVLSM, NCCL +cuda13.0, DCGM | [gather-nvidia.sh](gather-nvidia.sh) → `~/nvidia-airgap-bundle-ubuntu24.04.bin` | [pre-install-nvidia.sh](pre-install-nvidia.sh) | [install-nvidia.sh](install-nvidia.sh) (**reboot mandatory**) | [test-nvidia.sh](test-nvidia.sh) |
| Userland — apt pkgs, VS Code/Chrome/Firefox/Node/Bun/Opencode, Python venvs, llama.cpp, xfce4/xrdp | [gather-all.sh](gather-all.sh) → `~/all-airgap-bundle-ubuntu24.04.bin` | [pre-install-check.sh](pre-install-check.sh) | [install-all.sh](install-all.sh) | [test-all.sh](test-all.sh) |

Canonical target sequence (also documented at the top of [install-all.sh:11-17](install-all.sh#L11-L17)):

**6 mandatory reboots by default** — one per failure surface. The user has bricked this
exact target three times in prior bring-up attempts; verification ergonomics
beat install speed here. Each `install-all.sh` invocation auto-skips
already-`.ok` steps and resumes at the next pending one. `checkpoint_reboot`
(defined in [install-all-steps.sh](install-all-steps.sh)) fires
`exit 75` at the end of steps 06, 08, 09 (if desktop), 15, and 17; the
launcher catches it and prints a "REBOOT REQUIRED" banner. The invocation
after each reboot consumes `/var/lib/install-all/last-checkpoint` and verifies
that the previous phase actually stuck before later steps run. Set
`SKIP_CHECKPOINTS=1` to bypass — NOT recommended on this target.

```
# Phase 1 — NVIDIA stack (1 reboot)
sudo bash pre-install-nvidia.sh                       # strict preflight (no /etc writes)
sudo bash install-nvidia.sh                           # driver + FM + NVLSM + CUDA-min
sudo reboot                                           # REBOOT #1 — loads nvidia.ko
sudo bash test-nvidia.sh                              # fabric Completed? peermem loaded? kmod-from-running-kernel?

# Phase 2 — Userland readiness
sudo bash pre-install-check.sh                        # strict base-OS gate vs userland bundle

# Phase 3 — Userland install (5 checkpoint reboots when INSTALL_DESKTOP=1)
sudo bash install-all.sh                              # 01-06; exit 75 after apt userland
sudo reboot                                           # REBOOT #2 — confirm nvidia survived apt
sudo bash test-nvidia.sh && bash test-all.sh --phase userland
sudo bash install-all.sh                              # 07-08; exit 75 after needrestart
sudo reboot                                           # REBOOT #3 — confirm daemons clean from boot
sudo bash test-all.sh --phase apps
sudo bash install-all.sh                              # 09 only (if INSTALL_DESKTOP=1); exit 75
sudo reboot                                           # REBOOT #4 — confirm lightdm/xrdp boot
sudo bash test-all.sh --phase desktop
sudo bash install-all.sh                              # 10-15; exit 75 after sysctl tuning
sudo reboot                                           # REBOOT #5 — confirm tuning persists
sudo bash install-all.sh                              # 16-17; exit 75 at final-status
sudo reboot                                           # REBOOT #6 — final cold-boot sanity
sudo bash install-all.sh                              # consumes final post-reboot marker

# Phase 4 — Final verify
sudo bash test-nvidia.sh && bash test-all.sh
gpu-health-check
```

[INSTALL_INVENTORY.md](INSTALL_INVENTORY.md) is the authoritative manifest of every package, venv, systemd unit, config file, sysctl, and limit each phase touches. **Edit it whenever you change what the installers install** — it's not a generated doc, it's the spec.

### Script families and their conventions

- **`gather-*.sh`** — run on internet-connected WSL Ubuntu 24.04 as a normal user. Stage everything under `~/GPU_server_downloads*` then pack into a single `.bin` (tar.gz) with a `.sha256` sidecar. Re-runnable; resume by re-running. Do NOT add commands that require the airgapped target environment here. `gather-all.sh` ALSO copies `install-all-steps.sh` into the bundle and next to it so the target has both the launcher and the step library after extraction.
- **`pre-install-*.sh`** — read-only readiness gates on the target. Three severity tiers, kubeadm-style: **RED** blocks (exit 1), **YELLOW** warns, **GREEN** is inventory. Support `--ignore=R20,Y03` and `--json`. They must NOT modify the system.
- **`install-nvidia.sh`** — single script, `set -Eeuo pipefail`, re-exec under sudo if not root, write a timestamped transcript log in `$SCRIPT_DIR`. Owns the entire NVIDIA stack: open driver, FabricManager, NVLSM readiness (`ib_umad` autoload; Fabric Manager owns the NVLSM daemon), NCCL (skipped by default), the **minimal** CUDA toolkit (NOT `cuda-toolkit-13-0`), `/etc/profile.d/cuda.sh`, `/etc/ld.so.conf.d/cuda-system.conf`, the `apt-mark hold` set, the file:// nvidia bundle repo + `99-nvidia-prefer-bundle` apt pin. **Everything CUDA-/driver-adjacent lives here; install-all.sh and install-all-steps.sh must not touch any of it.**
- **`install-all.sh`** — thin launcher (~260 lines). Sources `install-all-steps.sh` and dispatches each step as a function call inside a subshell (`( step_NN_name )`). Supports `--list` / `--run NN` / `--from NN` / `--rerun NN[,NN]` / `--force` / `--skip-preflight`. Honors the resume marker (`/var/lib/install-all-prepped/stage1.done`) by skipping steps 03–05 on re-entry, except in `--run NN` mode (explicit re-run wins). Aggregates per-step `.ok` / `.failed` markers + warnings/errors into a final summary.
- **`install-all-steps.sh`** — single ~1660-line library, sourced by `install-all.sh`. Contains: (1) shared helpers (`log`/`warn`/`die`/`step`/`init_step`/`mark_step_ok`/`checkpoint_reboot`/`locate_bundle`/`source_bundle_metadata`/`_apt_install*`/`_wheelhouse_*`); (2) the `ALL_STEPS=(01-preflight … 17-final-status)` ordered array; (3) 17 step functions named `step_NN_name` (e.g. `step_14_llamacpp_build`). Each step function:
  - calls `init_step "NN-name"` (sets `STEP_LOG`, registers ERR/EXIT traps, redirects stdout/stderr through tee — all scoped to the subshell the launcher invokes it in)
  - calls `mark_step_ok` at the end (writes `/var/lib/install-all/steps/NN-name.ok`)
  - per-step log at `/var/log/install-all/<RUN_ID>/NN-name.log`
  - is invocable via `sudo bash install-all.sh --run 14` (no standalone-script path anymore)
  - steps 06 / 08 / 09 / 15 / 17 call `checkpoint_reboot` after `mark_step_ok` and exit 75 (reboot requested — launcher catches and prints "REBOOT REQUIRED" banner)
- **No `install-all.d/` directory.** The split-file layout was collapsed into the single `install-all-steps.sh` to ease FTP one-by-one transfer to the airgapped target. Do NOT recreate `install-all.d/` — edit `install-all-steps.sh` in place; each step function is delimited by a clear `# STEP NN: name` banner inside the file.
- **`test-*.sh`** — post-install verification. Support `--json`. Exit 0 only if every required check passes. Use the `record name STATUS detail` helper pattern (`PASS|FAIL|MISSING|SKIP`).

### Hard invariants — don't break these

- **R580 LTS driver, CUDA 13.0, NCCL `+cuda13.0` suffix.** NCCL packages must be pinned with the exact `+cuda13.0` suffix to guard against `+cuda13.2` drift (see [install-nvidia.sh](install-nvidia.sh) and gather-nvidia.sh comments).
- **Open kernel modules are mandatory** on Blackwell — use `nvidia-driver-580-open`, never the proprietary flavor.
- **`apt-mark hold`** on everything `nvidia-*`/`cuda-*`/`libnvidia-*`/`libnccl*`/`nvlsm`/`datacenter-gpu-manager-*`. This is install-nvidia.sh's responsibility. install-all.sh / install-all-steps.sh only adds holds on system runtime libs (libstdc++6, libgcc-s1, libgomp1, libc6) to prevent downgrades; it must never touch the NVIDIA stack.
- **CUDA minimal package set** — install-nvidia.sh deliberately AVOIDS `cuda-toolkit-13-0` (the ~3 GB metapackage). The actual install set is: `cuda-nvcc-13-0`, `cuda-cudart-13-0`, `cuda-cudart-dev-13-0`, `cuda-cccl-13-0`, `libcublas-13-0`, `libcublas-dev-13-0`, `libnvjitlink-13-0`, `cuda-compat-13-0`. `gather-nvidia.sh` must list the same set. `test-nvidia.sh` checks each individually. The metapackage check is GONE — referencing it from any script is a regression.
- **CUDA arch list `100-real;103-real`** for B300 (sm_103 Blackwell Ultra) + B200 (sm_100). `-real` strips PTX because the hardware is fixed. Used when building llama.cpp; default in `install-all-steps.sh`.
- **llama.cpp cmake flags (2026-05 baseline)**: `-DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=100-real;103-real -DLLAMA_OPENSSL=ON -DLLAMA_BUILD_UI=OFF`. **`-DLLAMA_CURL=ON` is deprecated** (llama.cpp #18922, Jan 2026); the OpenSSL path requires `libssl-dev` (present in `gather-all.sh`).
- **Python 3.12** + **torch 2.11.0+cu130** + PyG cu130 wheels from `data.pyg.org/whl/torch-2.11.0+cu130.html`. Pass `torch==2.11.0` to pip (NOT `torch==2.11.0+cu130` — the `+cu130` suffix is internal to the wheel filename). Don't bump these without updating both gather scripts, both inventory sections, and the cu130 wheel index references.
- **Both venvs pin torch to the same `${TORCH_VER_TRAINING}+cu130`.** `gather-all.sh` explicitly downloads `torch==2.11.0` from `https://download.pytorch.org/whl/cu130` into BOTH `wheels/inference/` and `wheels/training/` *before* any transitive resolver runs. This is the real invariant — *not* "inference is CPU-only" (an earlier note claimed multi-GPU NCCL ABI skew between separate venvs and cited #15525/#20862/#28283; those three are actually unrelated PRs about `reflection_pad2d`, `jit.trace` forward-hooks, and clang-tidy fixes). The real NCCL pitfall (PyTorch #112285, #122571) is **within a single venv** when two packages pull mismatched `nvidia-nccl-cu*` wheels; separate venvs are separate processes and each dlopen their own `libnccl.so.2`. Inference uses sentence-transformers/transformers GPU-capable; bulk generation still routes to llama.cpp's HTTP server (`step_14_llamacpp_build`). **vLLM remains excluded** — its torch/NCCL pinning is too aggressive and historically diverges from the training venv inside the same install. If you bump `TORCH_VER_TRAINING`, the inference download in section 6 of `gather-all.sh` must move with it; otherwise pip silently resolves torch from PyPI default (currently 2.12.0) and breaks PyG `+pt211cu130` ABI alignment.
- **Install prefixes default to `/scratch/`** (no `$HOME` dependency). `INFERENCE_PREFIX`, `TRAINING_PREFIX`, `JUPYTER_PREFIX`, `LLAMA_PREFIX` all branch off `SCRATCH_ROOT`.
- **DOCA-OFED is vendor-installed** — never bundle it; `pre-install-nvidia.sh` only verifies `ofed_info -s` shows DOCA 3.2+.
- **Bundle variant marker** — `meta/target.env` in the bundle carries `BUNDLE_VARIANT=prepped`. `install-all.sh` refuses to run a bare-metal bundle on a prepped server (and vice versa). Don't remove the gate.
- **Airgap mandate** — no script in `install-all.sh` / `install-all-steps.sh` / `install-nvidia.sh` may fetch from the internet. The only internet-using scripts are `gather-*.sh`. `grep -RnE 'curl|wget|pip install$|--index-url' install-all.sh install-all-steps.sh install-nvidia.sh` should produce no live calls; the airgap-vs-upstream comparison table lives in `INSTALL_INVENTORY.md`.

### Common dev commands (this repo, on WSL or Windows host)

```bash
bash -n install-all.sh install-all-steps.sh # syntax check the launcher + the step library
bash -n install-nvidia.sh test-nvidia.sh pre-install-nvidia.sh
bash -n gather-all.sh gather-nvidia.sh pre-install-check.sh test-all.sh
bash install-all.sh --list                  # show step status (no root needed)
sudo bash install-all.sh --run 14           # run a single step
sudo bash install-all.sh --rerun 14         # delete its .ok marker and re-run
sudo bash install-all.sh --from 11          # resume from step 11, skipping already-.ok steps
sudo bash install-all.sh --from 11 --force  # resume from step 11, re-run all regardless of .ok
sudo bash install-all.sh --skip-preflight   # skip 01-preflight's pre-install-check.sh call
SKIP_CHECKPOINTS=1 sudo bash install-all.sh # bypass all checkpoint_reboot exit 75s (not recommended on target)
sed -n '2,32p' install-all.sh               # -h/--help dumps the script header
INSTALL_DESKTOP=0 bash gather-all.sh        # headless variant (no xfce4/xrdp)
INCLUDE_JUPYTER=0 bash gather-all.sh        # skip JupyterLab wheels
bash test-all.sh --phase userland --json    # verify one install phase, machine-readable output
# valid --phase values: all (default), nvidia, userland, apps, desktop, venvs, tuning
```

There is no test suite for the shell scripts — validation happens on the target via `test-nvidia.sh` / `test-all.sh --json`. When changing behavior, also update the matching `pre-install-*.sh` check and the relevant section in `INSTALL_INVENTORY.md`.

### `legacy/Ubuntu_offline_setup/`

Older single-phase variants (gather-all.sh, install-all.sh, install-phase1.sh, install-phase2.sh, pre-reboot.sh, test-all.sh). Kept for reference. **Do not edit** unless explicitly resurrecting that flow — current development is on the two-bundle scripts at the repo root.

### `install-llamacpp.sh` / `gather-llamacpp.sh`

Standalone Ubuntu 22.04 llama.cpp installer. Predates the unified bundles, still works but is out of the main flow. Userland `install-all.sh` is the modern path for llama.cpp on this hardware.

## FEA → HDF5 toolchain

- [warpage_to_hdf5.py](warpage_to_hdf5.py) — converts ANSYS APDL `.inp` mesh + `.txt` warpage rasters into a single HDF5 dataset that conforms to [DATASET_FORMAT.md](DATASET_FORMAT.md). Uses `ProcessPoolExecutor`; respect `--workers`.
- [visualize_mesh.py](visualize_mesh.py) — auto-discovers the single `.inp` and `.h5` in a directory and renders mesh (colored by part) and dataset (colored by z-displacement, feature index 5).
- [visualize_raster.py](visualize_raster.py) — tab-separated raster grid viewer with `--nodata 9999.0` (auto-switches to `Agg` when no display).

Dataset feature layout is fixed at 8 channels: `[x, y, z, x_disp, y_disp, z_disp, stress, part_no]`. Sample IDs are 1-indexed string keys under `data/`. **If you change the schema, update DATASET_FORMAT.md in the same commit** — downstream GNN training code in sibling Huni projects reads it.

## Repo hygiene notes

- `requirements-all-projects.txt` is the union of dependencies across Huni's sibling projects (SimulGenVAE, SimulGenVAE2D, PEMTRON_warpage, LLM_API, LLM_API_fast, Gmail_exporter). Excluded entries are listed in the file header — preserve those exclusions (Windows-only / Py2 / huge dev tools / CUDA-compile-only packages).
- Files like `20250905192955@B550657650203.txt`, `installalllogs`, `a`, `codex_tmp_nvidia_check/`, `NVIDIA Corporation/umdlogs/` are stray data captures — don't treat them as source.
- [git-auto-push.sh](git-auto-push.sh) commits with a timestamp message and `git pull --rebase` before pushing. It's a convenience script, not the recommended commit flow when Claude is making changes.
