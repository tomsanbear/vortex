# tomsanbear/vortex — fork bill of materials

The **incubation trunk** for Vortex changes born in QuiltDB's production use — a streaming-ingest storage engine that writes Vortex files continuously (many structurally-identical fragments per column) and scans them through DataFusion. Changes land on `quiltdb-develop`, prove themselves downstream, then graduate to focused `fix/*` / `feat/*` branches and upstream PRs. This file is the single inventory: every carried change, its upstream status, and our confidence in it.

**Base**: `quiltdb-develop` is a **linear rebase stack** (no merge commits) on upstream `vortex-data/vortex` `develop`, currently `6ae2a8b0e` (2026-08-06). Validation at the rebase: `cargo nextest` 3742/3742 across vortex-{array,compressor,btrblocks,file,io,layout} plus 277/277 vortex-datafusion, workspace build clean, `cargo +nightly fmt --all` a no-op, `clippy --all-targets --all-features` clean.

**Sync protocol**: fetch upstream → safety branch `quiltdb-develop-pre-sync-<date>` → `rerere` on → `git merge-tree` dry-run + `git cherry` triage → rebase with a `cargo check` at every non-trivial stop so each commit compiles → `git range-diff` against the safety branch → full test/fmt/clippy gates → `git push --force-with-lease fork quiltdb-develop`. Commits that upstream has absorbed (verbatim or rewritten) resolve to `--ours` and drop; record them below under [Merged upstream](#merged-upstream) or [Closed with receipts](#closed-with-receipts).

**Status**: `Incubating` (on the stack, not yet PR'd) · `Submitted (#PR)` · `Merged (#PR)` · `Rejected (#PR)` · `Refuted` (we measured it and killed it ourselves) · `Superseded` (upstream solved it another way) · `Wanted` (need identified, nothing written).

**Confidence**: `proven` (shipping downstream with receipts) · `tested` (unit/bench coverage, not in production) · `experimental` · `n/a`.

**Maintenance protocol** (humans and agents): add an entry when a change lands on `quiltdb-develop`; move it when a PR opens, merges, or closes; refresh commit hashes after every rebase (they all change). Never delete Rejected/Refuted/Superseded entries — their receipts are what stop the idea being re-litigated. One entry per logical change, not per commit. Check [Upstream coordination](#upstream-coordination) before opening any PR.

## Contents

- [Upstream coordination](#upstream-coordination) — active upstream work that overlaps with carried changes; read before PRing
- [In flight](#in-flight) — open upstream PRs
- [Incubating on `quiltdb-develop`](#incubating-on-quiltdb-develop) — landed on the stack, not yet PR'd
- [Incubating on side branches](#incubating-on-side-branches) — larger work not yet on the stack
- [Merged upstream](#merged-upstream) — this fork's landed history
- [Closed with receipts](#closed-with-receipts) — superseded and dropped, kept so nobody retries them blind
- [Wanted](#wanted) · [Branch hygiene](#branch-hygiene)

## Upstream coordination

Verified at the 2026-08-06 rebase. Check here before opening a PR — several carried changes sit on actively-moving upstream surfaces.

- **The compressor is under active upstream rework** (tracking issue #7697): #8745 reorganized `vortex-compressor` into `compressor/` + `scheme/` modules ("I'd like to make some improvements to the compressor logic"), and #8667 hardcoded constant detection into the cascade ahead of scheme selection. Our freeze/observer fast-path family lives exactly on this surface — talk to the maintainer (connortsui20) before PRing it, and expect further churn at each sync.
- **The DataFusion integration is under active upstream rework**: `VortexSource` improvements (#8718, #9167), configurable `ExpressionConvertor` (#9185), session-time source configuration (#8575). Our dynamic-filter pushdown family touches the same opener/convertor code — a PR should open with a design conversation, and each sync will conflict here (it did in 2026-08).
- **Upstream converges on our fixes independently** — twice now a carried decimal fix was reimplemented upstream within weeks (see [Closed with receipts](#closed-with-receipts)). For small correctness fixes, PR early rather than carrying; the carry costs conflict resolution at every sync and the upstream version usually lands anyway.
- Upstream removed `Cargo.lock` (#8394): builds float within semver ranges (arrow pinned `58.3` resolves 58.4.0, datafusion `54` resolves 54.1.0 as of 2026-08-06). `cargo tree`, not the repo, answers "what version am I on".

## In flight

None. Two prior PRs merged (see [Merged upstream](#merged-upstream)); nothing is currently open.

## Incubating on `quiltdb-develop`

Stack order, oldest first. Hashes are post-2026-08-06-rebase and change at every sync.

### Zero-copy adoption of single-chunk object-store stream bytes
`vortex-io` · `08a4967b0`, ablation example `fdaf0b611` · Incubating · tested

Opt-in zero-copy adoption of the bytes an object store returns for a single-chunk read, instead of copying into a fresh aligned buffer. The `read_at` ablation example exists to measure the copy-vs-adopt tradeoff per workload before turning it on.

### Compressor scheme-winner fast path: freeze hint + winner observer
`vortex-compressor` / `vortex-btrblocks` · `fc7c11b47`, `787fcf89f`, `a11ad4b46`, `3b10601ea` · Incubating · **proven — kills the per-fragment IntDict distinct-counter recompute in streaming ingest**

The heart of the fork. A streaming writer that observes a column's winning scheme on one fragment has no reason to re-run scheme selection — including the `HashMap<NativeValue<T>, u32>` distinct counter IntegerStats builds whenever an IntDict candidate is in scope — on every structurally-identical fragment after it. Four pieces: `CompressorContext::with_frozen_scheme(id)` dispatches straight to a cached winner, skipping the eligible-schemes filter, the merged stats-options fold, and the two-pass `choose_best_scheme`; `CascadingCompressor::compress_with_ctx` (plus the BtrBlocks pass-through) lets callers supply the context; `with_winner_observer` reports the winner (after selection or freeze, before compress, so a failing compress doesn't suppress the observation) — the input for the next fragment's freeze hint; and the freeze gate still consults the frozen scheme's own `expected_compression_ratio` — `Skip` and `Deferred::Callback` fall through to normal selection (the shape-fragility fence: BitPacking on signed-negative input, FoR on wrap-overflow spans, and the constant family's `is_constant` callback, which if bypassed would corrupt every row to the first scalar), while `Ratio`/`AlwaysUse`/`Deferred::Sample` proceed. A stale hint never produces wrong output; worst case is one wasted lookup plus one estimate call.

Ported onto upstream's #8745 module reorg at the 2026-08 sync (`compressor/cascade.rs` + `scheme/ctx.rs`). Behaviour note from the same sync: upstream's built-in constant detection (#8667) short-circuits ahead of scheme selection, so constant chunks never reach selection and the observer does not fire for them — the observer reports *scheme* wins only. **Do not PR without a maintainer conversation** — see [Upstream coordination](#upstream-coordination).

### Per-field compressor plugin overrides
`vortex-file` · `15407d888` · Incubating · tested

`WriteStrategyBuilder::with_field_compressors(HashMap<FieldPath, Arc<dyn CompressorPlugin>>)`: `build()` constructs a per-field leaf chain identical in shape to the default fallback but with the leaf `CompressingStrategy` swapped to the override. This is how a per-column winner cache (previous entry) gets installed without duplicating the whole strategy chain. Explicit `with_field_writer` entries take precedence for the same path.

### `with_compressing_stats` narrows per-chunk stat compute
`vortex-file` · `2ff1b1827` · Incubating · tested

The leaf `CompressingStrategy` pre-computes `Stat::all()` per chunk; callers whose scheme set never reads `IsSorted`/`IsStrictSorted`/`UncompressedSizeInBytes`/`Sum`/`NaNCount` can drop those per-fragment scans. Shared by the fallback chain and every per-field override.

### BtrBlocks estimate gates read the array stats cache directly
`vortex-btrblocks` · `67db30b4d` (bitpacking), `67d02c63b` (FoR), `d30acd743` (zigzag) · Incubating · proven (part of the streaming-ingest freeze-path work)

The bitpacking/FoR/zigzag `expected_compression_ratio` gates need only min/max, but read them through `data.integer_stats`, triggering the full `IntegerStats` compute (including the dict distinct counter) — pure waste on the freeze fast path. They now read `compute_min`/`compute_max` from the array's stats cache, populated by `CompressingStrategy`'s `compute_all` before any estimate runs, so it is O(1) on a hit.

### FoR skips signed spans that would wrap-overflow
`vortex-btrblocks` · `7c27c1aaa` · Incubating · tested

Correctness fence: for signed inputs whose `max - min` exceeds the signed positive range, `FoR.encode`'s `wrapping_sub` produces biased values that read as negative, tripping `bitpack_encode`'s negative-integer guard in the unconditional BitPacking call inside `FoRScheme::compress`. Compression ratio in this regime would have been 1.0 anyway, so refusing costs nothing.

### DataFusion dynamic join-filter pushdown into the Vortex scan
`vortex-datafusion` / `vortex-layout` · `609c5e323`, `d41cf6e30`, `63ace6398`, `54e86b1d3`, `d1484fc6f`, guard test `635b4dbd4` · Incubating · **proven — probe bytes 4.00 MB → 1.91 MB, surviving rows 1.00 M → 8.19 K on the clustered-probe benchmark**

DataFusion materializes a dynamic filter on a hash join's probe side once the build completes: `col >= min AND col <= max` plus an `InList` for small builds or an unconvertible `hash_lookup` for large ones. Vortex previously dropped these at the in-scan boundary (upstream issue #4034 territory). Small builds: snapshot the dynamic at the `FileOpener`, adapter-rewrite to the file schema, and route the convertible `InList` into the in-scan filter. Large builds: route the min/max bounds in as a **prune-only** filter — `FilterExpr` gains a per-conjunct `ConjunctEval` (`PruneAndFilter`/`PruneOnly` via `with_prune_only`), so the bounds drive zone pruning via stats but skip the per-row pass (a non-selective bound evaluated per row is pure decode cost — the measured regression of earlier attempts). `ScanBuilder::with_some_prune_filter` threads it through `RepeatedScan`. Hardening: `VORTEX_DYNAMIC_INSCAN=0` kill-switch (the dynamic still feeds the file-level `FilePruner`); the InList conversion is nullability-total (degrade, never abort) and gated on literal-only elements, with `NotExpr` conversion; a guard test pins that the TopK threshold stays a file-level prune. Dynamic filters remain `PushedDown::No`, so the join re-filters and results are unchanged.

Re-ported onto upstream's `BoundExpression` scan world at the 2026-08 sync. **Coordinate before PRing** — the opener/convertor surface is moving upstream (see [Upstream coordination](#upstream-coordination)).

### Pushed-filter evaluation errors surface as Arrow errors, not read failures
`vortex-datafusion` · `9a056c4c7`, `4dedf2908`, `069a1aacb` · Incubating · tested

An arithmetic overflow or divide-by-zero raised evaluating a pushed-down predicate used to surface as "Failed to read Vortex file: <path>" — a query-evaluation error mislabelled as a corrupt file. `evaluation_arrow_error` peels `Shared`/`Context` wrappers (the `MaskFuture` driving filter evaluation is `Shared`, so the structural `ArrowError` sits one or two levels down and must be rebuilt by hand — `ArrowError` is not `Clone`) and re-raises compute variants as `DataFusionError::ArrowError`; genuine read errors keep the read-file context, minus the object location in the message.

### Multipart upload lifecycle: lazy start, abort on failure
`vortex-io` (+ datafusion sink, python, jni call sites) · `bbbd80ab7`, `197d2620f` · Incubating · tested

Two halves of the orphaned-upload leak. Lazy multipart: objects that never split do a single PUT instead of initiating a multipart upload. Abort: `ObjectStoreWrite::abort()` is the error-path counterpart to shutdown (aborts an in-flight upload, no-op if never split or already completed); every production call site aborts on the write/shutdown error path, best-effort, primary error wins. Without it a failed encode left initiated-but-never-completed uploads: billable orphaned parts on S3/GCS until a lifecycle rule reaps them, uncommitted blocks on Azure for 7 days. `Drop` is the forgot-to-call backstop and only *reports* (vortex-io has no ambient runtime to spawn an abort onto); a cancelled task's orphan is the recorded residual — reaping stays the operator's lifecycle-rule job.

### `Handle::find` auto-selects `WasmRuntime` on wasm32
`vortex-io` · `cd88ec660` · Incubating · tested

Straightforward upstream-PR candidate.

### Build-graph slimming feature gates
`vortex-io` / `vortex` / workspace · `e5778e896` (`profiling-labels`), `5c396dcf1` (`smol-runtime`), `c54906ab5` (datafusion-expr `sql`) · Incubating · tested

Three independent dependency-graph cuts: `custom-labels` (a C++ compile + bindgen, dragging clang-sys into every consumer) moves behind an off-by-default `profiling-labels` feature with the two in-repo profiling consumers opted in; smol becomes opt-out via a default-on `smol-runtime` feature so a tokio-only consumer can shed the second async runtime with `default-features = false`; `datafusion-expr` drops its default `sql` feature. Good PR candidates individually.

### Decimal expr-cast rescale regression test
`vortex-array` · `8b1aff3b2` · Incubating (test-only) · n/a

The end-to-end pin for the decimal-rescale behaviour whose implementation upstream now owns (see [Closed with receipts](#closed-with-receipts)): the expr cast path — what vortex-datafusion's pushed-down filter uses — must rescale mantissas (`(10,2) → (20,4)` multiplies by 100), not relabel the dtype. Upstream's own tests cover the scalar and kernel layers but not this expr-level path.

### Stack glue
`workspace` · `1bcb71e07` (fmt/clippy style pass) · n/a

Formatting and lint conformance for the carried patches; not a logical change. Folds into neighbouring entries' conflicts at each sync.

## Incubating on side branches

### Pretrained FSST compressor scheme
`vortex-btrblocks` · `feat/fsst-pretrained-compressor` (41 commits over the old base), variant `feat/scheme-arc-storage` (42) · Incubating · proven (downstream streaming ingest)

`FSSTSchemeWithPretrained` — reuse a caller-trained FSST symbol table instead of retraining per array. This is the downstream consumer of the whole freeze/probe family on the trunk: retraining FSST per chunk per string column attributed ~20% of active CPU on a 20 M-row, 10-UTF8-column, ~1000-fragment ingest before the probe fix (#8406) and scheme-arc plumbing landed. Two branch variants predate the 2026-08 rebase and need a re-port onto the reorganized compressor before the next move; the `Arc`-storage variant is the one to keep.

## Merged upstream

Newest first.

- **#8406** `fix(layout/dict): probe with the configured compressor instead of a hardcoded default` (merged 2026-07-13). The dict-eligibility probe ran a stock `BtrBlocksCompressor::default()`, silently retraining FSST per chunk for callers with pretrained codecs. Upstream then refined it further — the probe is now an explicit `DictStrategy::new` parameter with a `WriteStrategyBuilder::with_probe_compressor` override defaulting to the configured stats compressor — which absorbed our follow-up fix too (probe with the full compressor, not the IntDict-excluded one).
- **#8369** `fix[file]: read the one-row pruning result in can_prune` (merged 2026-06-11). Composite falsification trees stopped constant-folding after `ScalarFnConstantRule` was removed, so `can_prune` silently stopped pruning every and/or/eq predicate. Upstream's later #8345 stats-falsification migration rewrote the mechanism and kept our regression test (`test_can_prune_composite_predicates`, upgraded to the `VortexResult` idiom).

## Closed with receipts

Kept so nobody retries (or re-carries) them blind.

### Decimal rescale-on-cast (array kernel + scalar path)
Superseded · receipt: verified at the 2026-08-06 rebase

We carried `DecimalValue::rescale`/`rescale_i256` with round-half-away-from-zero, used from both the array cast kernel and `DecimalScalar::cast` (which previously relabelled the dtype and left the mantissa untouched — a pushed-down `cast(col) > lit` then matched nothing). Upstream independently implemented the same semantics (`DecimalValue::cast_decimal`, a rescaling array kernel, and tests `cast_different_scale_rescales` / `cast_lower_scale_requires_exact_rescale` plus lossy-fail and primitive-to-decimal coverage, which is *stricter* than ours — narrowing that loses precision errors instead of rounding). Both carries dropped at the rebase; only the expr-level regression test survives on the trunk. Lesson recorded in [Upstream coordination](#upstream-coordination): PR small correctness fixes early instead of carrying.

### `decimal_to_arrow` carries precision/scale
Superseded · receipt: verified at the 2026-08-06 rebase

`Decimal{128,256}Array::new_scalar` stamps Arrow's default `(38,10)`/`(76,10)`, mis-scaling every stat datum a consumer read back (a DECIMAL(40,2) stat rescaled 678.90 to near-zero and over-pruned `amount > 200.00` to empty). Upstream reimplemented width-selection-by-precision with `with_precision_and_scale` and its own regression test (`decimal_scalar_to_arrow_preserves_precision_and_scale`). Dropped at the rebase, including the then-orphaned `DecimalScalar::decimal_dtype()` accessor.

### Probe dict-eligibility with the full compressor
Superseded · receipt: upstream refinement of #8406

Our follow-up to #8406 (the IntDict-excluded data compressor could never detect an integer column as dict-eligible) is structurally guaranteed by upstream's current shape: the builder's probe defaults to the stats compressor, which keeps IntDict. Dropped at the rebase.

## Wanted

Currently empty.

## Branch hygiene

- `quiltdb-develop-pre-sync-2026-08-06` — safety branch at the pre-rebase tip; delete once the rebased stack has survived downstream use.
- `fix/can-prune-one-row-result`, `fix/dict-strategy-probe-compressor` — PR branches for #8369/#8406, both merged; verify against upstream and delete.
- `zero-copy-object-store-read` — early 3-commit slice of the trunk's first entries; superseded by the trunk, delete.
- `fix-c-on-quiltdb-pin-pre-slim-backup` — pre-slim backup of a QuiltDB pin, pre-rebase vintage; delete once QuiltDB has repinned onto the rebased stack.
- `feat/fsst-pretrained-compressor` / `feat/scheme-arc-storage` — active side branches (see above); consolidate to one variant at the re-port.
- The `fork` remote also hosts `ad/*` branches belonging to another contributor's CUDA/GPU line of work — out of scope for this inventory.
