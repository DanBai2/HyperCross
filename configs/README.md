# Benchmark configurations

Store portable benchmark configuration templates here. Configuration files must use dataset keys and relative artifact references; they must not embed machine-specific absolute data paths.

## Training fast lane

`fast_lane/` contains hash-bound stage templates for task 13.9. Load them with
`benchmark.fast_lane.load_fast_lane_template` before allocating a run. The
loader rejects changes to the audited environment, CUDA device, process count,
precision, seed set, resource limits, checkpoint policy, or output roots.

All templates require:

- the `TCFM` environment and audited Environment Lock
  `a381dbcd2b3aaf8723b92ce4ed7f44c95d039ff46c04e8d343b82c88eb36a275`;
- `cuda:0` on `NVIDIA GeForce RTX 5080`, one visible CUDA device, and one process;
- deterministic float32 execution;
- a recovery boundary every epoch containing model, optimizer, scheduler, and
  stage state;
- selected, last, and recovery checkpoint retention;
- workspace-relative writes under the stage's `runs_v2/` root or `artifacts/`;
- `src/` and legacy `runs/` as read-only inputs; and
- resume only from the same stage's workspace-relative `runs_v2/` tree. `F:`
  paths and parent traversal are rejected before state restoration.

| Stage | Seeds | Hard epoch/step/time limits | Output root |
|---|---|---|---|
| smoke | 42 | 2 / 128 / 15 minutes | `runs_v2/smoke` |
| baseline | 42 | 300 / 500,000 / 48 hours | `runs_v2/baselines` |
| screening | 42 | 80 / 100,000 / 12 hours | `runs_v2/screening` |
| strict | 7, 42, 123 | 400 / 1,000,000 / 72 hours | `runs_v2/strict` |
| official | 42 | 400 / 1,000,000 / 72 hours | `runs_v2/official` |

Every stage also has a 16,000,000,000-byte hard peak-memory limit and advisory
limits of 500,000,000 parameters and 8,000,000,000 input bytes. These values
are part of each template hash. Changing one requires a new template hash and
new Frozen Run Identity; it is not an in-place resume.

## PowerShell CLI

`smoke-job.example.json` documents the portable job-request shape. Replace both
all-zero hash placeholders with hashes from completed dataset and protocol
manifests before using it to launch work. Job configs contain dataset keys and
workspace-relative references only. They must not contain machine-local dataset
paths.

Validate and freeze the request without writing files or starting training:

```powershell
python -m benchmark.cli plan `
  --template "configs\fast_lane\smoke.json" `
  --job-config "configs\fast_lane\smoke-job.example.json" `
  --workspace-root "G:\Datasets\HyperCross" `
  --device "cuda:0" `
  --process-count 1 `
  --precision "float32" `
  --determinism "deterministic" `
  --output-root "runs_v2/smoke"
```

Use the same arguments with `smoke --dry-run` to validate the bounded smoke
entry point. Omit `--dry-run` and add a concrete package runner such as
`--runner-module benchmark.runners.hyperleaf_smoke` only after that runner is
installed. The CLI launches Python with an argument vector and `shell=False`,
so paths containing spaces do not require shell-specific escaping beyond normal
PowerShell quoting.

`run` and `resume` use the same frozen request. `resume` additionally requires a
workspace-relative `--recovery-source` below the selected stage output root.
`aggregate`, `report`, and `audit` accept portable JSON requests and support
`--dry-run` or shell-free dispatch to their package runner modules.

Before a non-dry launch, the CLI checks the active `CONDA_DEFAULT_ENV`, captures
the live Environment Lock, requires the audited TCFM lock and RTX 5080 identity,
then atomically persists the Frozen Run Identity. Absolute paths in job payloads,
all `F:` paths, parent traversal, and references targeting legacy `src/` or
`runs/` are rejected before launch.

## Remediation precision templates

`remediation/v1/` is a separate immutable namespace for task 24.1. It does not
replace or modify any `fast_lane/` template. Load these templates with
`benchmark.remediation.load_remediation_template` and provide a
`RemediationPlanningContext` before freezing a job.

The two v1 profiles are:

- `deterministic-float32.json`: deterministic framework settings, float32, and
  autocast disabled;
- `performance-bf16.json`: performance framework settings and CUDA autocast
  with `bfloat16` operations under a `mixed_bfloat16` precision identity.

Both profiles freeze seeds `7/42/123`, one process on audited TCFM/RTX 5080,
the strict 400-epoch/1,000,000-step/72-hour budget, one-epoch checkpoint
boundaries, actual-bytes-read input accounting, and all four recovery states.
Each profile has a distinct canonical template `config_hash`, output root,
artifact root, and recovery root. Every frozen job embeds that template hash,
precision, autocast policy, and gradient-accumulation policy in its own
`FrozenRunConfig`. Resume planning requires exact equality of every
`Frozen_Run_Identity` field and only accepts recovery paths inside that job's
precision-specific namespace.
