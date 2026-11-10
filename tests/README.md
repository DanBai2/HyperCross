# Layered pytest suites

Pytest assigns a primary marker from each test file's first directory below `tests/`. The supported markers are:

| Marker | Purpose | Ordinary run |
|---|---|---|
| `unit` | Isolated deterministic behavior | included |
| `property` | Hypothesis properties; the CI profile runs at least 100 examples | included |
| `integration` | Bounded multi-component behavior | included |
| `regression` | Previously fixed behavior | included |
| `smoke` | Bounded viability checks that do not consume formal evaluation budget | included when CPU-only |
| `audit` | Evidence, provenance, leakage, and artifact integrity | included when CPU-only |
| `gpu` | Audited TCFM CUDA and RTX 5080 required | excluded |
| `slow` | Runtime/resource-heavy check | excluded |
| `strict` | Formal strict-protocol or complete-training check | excluded; implies `slow` |

Files under `tests/unit`, `tests/property`, `tests/integration`, `tests/regression`, `tests/smoke`, `tests/audit`, and `tests/gpu` receive the matching marker automatically. A test outside those directories must declare a primary marker explicitly. Add `@pytest.mark.gpu`, `@pytest.mark.slow`, or `@pytest.mark.strict` for additional execution constraints.

The ordinary command excludes `gpu`, `slow`, and `strict`, so it cannot start complete long training:

```powershell
python -m pytest
```

Use the PowerShell wrapper from the repository root for named suites:

```powershell
& ".\scripts\Invoke-Pytest.ps1" -Suite Unit
& ".\scripts\Invoke-Pytest.ps1" -Suite Property
& ".\scripts\Invoke-Pytest.ps1" -Suite Smoke
& ".\scripts\Invoke-Pytest.ps1" -Suite Audit
& ".\scripts\Invoke-Pytest.ps1" -Suite Collection
```

GPU smoke runs separately in TCFM and fails before test execution unless the active dependency versions, CUDA visibility, and RTX 5080 identity match `artifacts/audit/training_fast_lane_audit.json`:

```powershell
& ".\scripts\Invoke-Pytest.ps1" -Suite GpuSmoke
```

A direct GPU command is also supported. Without `--require-audited-environment`, mismatched GPU tests skip instead of failing:

```powershell
conda run --no-capture-output -n TCFM python -m pytest -m "gpu and smoke" --require-audited-environment
```

These commands invoke pytest once and do not use watch mode or shell command chaining.
