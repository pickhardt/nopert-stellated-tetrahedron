# Reproducibility artifacts

This repository contains the complete computer-assisted certificate for the paper, together with
the scripts that **generate** it and the independent scripts that **replay-verify** it. A single
top-level script, `assembly_verify.py`, consumes the four verified inputs and reports the overall
verdict.

Everything here is reproducible from source. You have two ways to get the certificate manifests
(a few GB), which are not stored in this repo:

- **Download the pre-built manifests** (fastest — verify in ~1.5 h):
  ```bash
  aws s3 sync --no-sign-request s3://nopert/manifests/ code/
  ```
  (public-read bucket, `us-east-1`; also browsable at `https://nopert.s3.amazonaws.com/`).
- **Regenerate them from the drivers** (fully from source; see [`REPRODUCE.md`](REPRODUCE.md)).

Either way, the verifiers then re-check them from disk.

For your convenience, I am hosting the certificate manifests on S3 for the forseable future,
to make it easy to download without regenerating. However, since it costs money
for ongoing storage as well as transfer costs, I expect to remove them from S3 at
some point. From then on, you will have to regenerate them yourself.

> **Verdict reproduced by this artifact:** `ASSEMBLY VERDICT: COMPLETE (all inputs verified,
> interfaces gap-free)`.

## The proof in one paragraph

Non-Rupertness of $\mathcal{P}_{11/20}$ is reduced (Theorem GA) to
four finite, certified inputs:

| Input | What it certifies | Generator | Verifier | Manifest |
|-------|-------------------|-----------|----------|----------|
| **IN-1a (SB-box)** | box-anchored near-orbit exclusion, $\delta\in[5.12\text{e-}3,\,5\text{e-}2]$ | `sbbox_parallel.py` / `sbbox_prove.py` | `sbbox_verify_tiling.py` | `sbbox_ext/` |
| **IN-1b (SF)** | second-order stress certificate | `sf_make_witness.py` | `verify_sf.py` | `sf_shards/` |
| **IN-2 (depth grid)** | $d_{\mathrm{lo}}>0$ off the critical windows + wall layer | `depthgrid_parallel.py`, `wallfix_sweep.py` | `verify_wallfix.py` | `dg_full/`, `wallfix_out/` |
| **IN-3 (far / det-dual)** | far-deviation exclusion incl. the low-$\delta$ wall band | `rust/dk_prover` (sweep) + `recover_lowdelta.py`, `recover_fix.py` | `dk_verify.py`, `verify_recover.py` | `dk_full/`, `recover_out/` |
| **IN-4 (assembly)** | ties IN-1..IN-3 + interface inequalities | — | `assembly_verify.py` | `reports/assembly_report.json` |

Every verifier re-checks its manifest from disk alone — coverage (nothing missing/extra), exact
dyadic tiling, zero unclosed cells, and a re-execution of the stored per-cell certificate — and shares
only the interval-arithmetic core (`fast_interval` / `tm2` / `dk_kernel`) with the generator, not the
generator's search/selection logic.

The (SB-box) $\delta$-continuum is certified by a direct interval-$\delta$ sweep
(`ivd_extension_sweep.py`: 10,560 cells, 0 fails over $[1.28\times10^{-3},5\times10^{-2}]\times[0,\pi]$),
and the one analytic constant of the local theorem, the order-4 Taylor-tail constant $C_4 \le 53 < 60$,
is proved by its own exact-interval certificate (`c4_taylor.py`); `assembly_verify.py` checks both.
(The earlier flag-acceleration bound $A_\delta \le 8$ via `adelta_run_all.py` and the point-$\delta$
manifests remain as an independent cross-check, no longer load-bearing.) See [`REPRODUCE.md`](REPRODUCE.md).

This artifact contains only the code and certificates.

## Layout

```
code/     all Python modules (flat; imports are cwd/self-relative) + Cython sources + build_ext.py
rust/     the DK det-dual prover (dk_prover, Rust) that produces the IN-3 annulus sweep
reports/  the small PASS reports checked in as evidence (verify_*_report.json, assembly_report.json,
          and the per-shard .done stat summaries)
```

The manifest directories (`dg_full/`, `wallfix_out/`, `dk_full/`, `recover_out/`, `sf_shards/`,
`sbbox_ext/`) are **produced by the drivers** — see [`REPRODUCE.md`](REPRODUCE.md) for the exact
commands and the expected cell counts at each step.

## Quick start (download + verify)

```bash
pip install -r requirements.txt
cd code
aws s3 sync --no-sign-request s3://nopert/manifests/ .   # ~2 GB of certificate manifests
python verify_recover.py --full --jobs 8    # IN-3 low-δ recovery layer  → VERDICT PASS
python verify_wallfix.py --full --jobs 8    # IN-2 wall layer            → PASS
python verify_sf.py                          # IN-1b SF                   → VERDICT PASS
python sbbox_verify_tiling.py                # IN-1a SB-box               → PASS
python assembly_verify.py                    # ties it together → ASSEMBLY VERDICT: COMPLETE
```

To regenerate a manifest from scratch instead of downloading it, follow [`REPRODUCE.md`](REPRODUCE.md).
