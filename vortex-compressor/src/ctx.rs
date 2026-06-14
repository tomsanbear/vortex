// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright the Vortex contributors

//! Compression context for recursive compression.

use std::fmt;
use std::sync::Arc;

use vortex_error::VortexExpect;

use crate::compressor::ROOT_SCHEME_ID;
use crate::scheme::SchemeId;
use crate::stats::GenerateStatsOptions;

/// Caller-supplied hook invoked when [`choose_and_compress`] determines a
/// winning scheme at this compression site.
///
/// Fires after selection (or short-circuit via a frozen-scheme hint) but
/// before the scheme's `compress` runs, so a failing compress does not
/// suppress the observation. The argument is the winner's
/// [`SchemeId`].
///
/// Useful for downstream consumers that want to build a per-column or
/// per-fragment cache of winners (the natural input to a future
/// [`with_frozen_scheme`](CompressorContext::with_frozen_scheme) hint
/// supplied on the next compression of the same column).
///
/// [`choose_and_compress`]: crate::compressor::CascadingCompressor
pub type WinnerObserver = Arc<dyn Fn(SchemeId) + Send + Sync>;

// TODO(connor): Why is this 3??? This doesn't seem smart or adaptive.
/// Maximum cascade depth for compression.
pub const MAX_CASCADE: usize = 3;

/// Context passed through recursive compression calls.
///
/// Tracks the cascade history (which schemes and child indices have been applied in the current
/// chain) so the compressor can enforce exclusion rules and prevent cycles.
#[derive(Clone)]
pub struct CompressorContext {
    /// Whether we're compressing a sample (for ratio estimation).
    is_sample: bool,

    /// Remaining cascade depth allowed.
    allowed_cascading: usize,

    /// Merged stats options from all eligible schemes at this compression site.
    merged_stats_options: GenerateStatsOptions,

    // TODO(connor): Replace this with an `im::Vector`
    /// The cascade chain: `(scheme_id, child_index)` pairs from root to current depth.
    /// Used for self-exclusion, push rules ([`descendant_exclusions`]), and pull rules
    /// ([`ancestor_exclusions`]).
    ///
    /// [`descendant_exclusions`]: crate::scheme::Scheme::descendant_exclusions
    /// [`ancestor_exclusions`]: crate::scheme::Scheme::ancestor_exclusions
    cascade_history: Vec<(SchemeId, usize)>,

    /// Caller-supplied scheme winner for this compression site, if known
    /// upfront.
    ///
    /// When `Some(id)` and a scheme with that id is registered and
    /// applicable to the canonical input, the compressor skips scheme
    /// selection: the merged stats-options fold and the two-pass
    /// [`choose_best_scheme`](crate::compressor::CascadingCompressor)
    /// dispatch are bypassed, and the chosen scheme's compress runs
    /// directly. Stats are derived from THAT scheme's own
    /// [`Scheme::stats_options`](crate::scheme::Scheme::stats_options),
    /// so any per-scheme stats that selection would have computed only
    /// to discard are never generated.
    ///
    /// On mismatch (id not registered, or the scheme refuses the
    /// canonical type, or the cascade's exclusion rules apply at this
    /// position), the cascade falls through to its normal selection
    /// path — safe in the face of a stale hint.
    ///
    /// The hint applies to the current level only. Children produced by
    /// the chosen scheme's compress go through normal selection unless a
    /// fresh hint is set on the descended context.
    frozen_scheme: Option<SchemeId>,

    /// Caller-supplied observer notified of the winning scheme at this
    /// compression site. Dropped on descent so the observer fires only
    /// at the level the caller installed it at.
    winner_observer: Option<WinnerObserver>,
}

impl fmt::Debug for CompressorContext {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        // `WinnerObserver` is `Arc<dyn Fn(SchemeId)>` and does not impl
        // `Debug`; elide it with a presence marker so the rest of the
        // context still renders for diagnostics.
        f.debug_struct("CompressorContext")
            .field("is_sample", &self.is_sample)
            .field("allowed_cascading", &self.allowed_cascading)
            .field("merged_stats_options", &self.merged_stats_options)
            .field("cascade_history", &self.cascade_history)
            .field("frozen_scheme", &self.frozen_scheme)
            .field(
                "winner_observer",
                &format_args!(
                    "{}",
                    if self.winner_observer.is_some() {
                        "<installed>"
                    } else {
                        "<none>"
                    }
                ),
            )
            .finish()
    }
}

impl CompressorContext {
    /// Creates a new `CompressorContext` with default state.
    ///
    /// External callers that want to attach scheme-selection hints
    /// (e.g. via [`Self::with_frozen_scheme`] or
    /// [`Self::with_winner_observer`]) before handing the context to
    /// [`CascadingCompressor::compress_with_ctx`](crate::CascadingCompressor::compress_with_ctx)
    /// start from this constructor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            is_sample: false,
            allowed_cascading: MAX_CASCADE,
            merged_stats_options: GenerateStatsOptions::default(),
            cascade_history: Vec::new(),
            frozen_scheme: None,
            winner_observer: None,
        }
    }
}

#[cfg(test)]
impl Default for CompressorContext {
    fn default() -> Self {
        Self::new()
    }
}

impl CompressorContext {
    /// Whether this context is for sample compression (ratio estimation).
    pub fn is_sample(&self) -> bool {
        self.is_sample
    }

    /// Returns the merged stats generation options for this compression site.
    pub fn merged_stats_options(&self) -> GenerateStatsOptions {
        self.merged_stats_options
    }

    /// Returns the cascade chain of `(scheme_id, child_index)` pairs.
    pub fn cascade_history(&self) -> &[(SchemeId, usize)] {
        &self.cascade_history
    }

    /// Returns a display wrapper for the current cascade ancestry.
    pub(crate) fn cascade_path(&self) -> impl fmt::Display + '_ {
        CascadePath(&self.cascade_history)
    }

    /// Returns the current cascade ancestry depth.
    pub(crate) fn cascade_depth(&self) -> usize {
        self.cascade_history.len()
    }

    /// Whether cascading is exhausted (no further cascade levels allowed).
    ///
    /// This should only be used in the implementation of a [`Scheme`](crate::scheme::Scheme) if the
    /// scheme knows that it's child _must_ be compressed for it to make any sense being chosen.
    pub fn finished_cascading(&self) -> bool {
        self.allowed_cascading == 0
    }

    /// Returns a context that disallows further cascading.
    pub fn as_leaf(mut self) -> Self {
        self.allowed_cascading = 0;
        self
    }

    /// Returns a context with the given stats options.
    pub(super) fn with_merged_stats_options(mut self, opts: GenerateStatsOptions) -> Self {
        self.merged_stats_options = opts;
        self
    }

    /// Returns a context marked as sample compression.
    pub(super) fn with_sampling(mut self) -> Self {
        self.is_sample = true;
        self
    }

    /// Descends one level in the cascade, recording the current scheme and which child is
    /// being compressed.
    ///
    /// The `child_index` identifies which child of the scheme is being compressed (e.g. for
    /// Dict: values=0, codes=1).
    ///
    /// Any caller-supplied scheme winner is dropped on descent. The hint
    /// applies at one cascade level only; carrying it to children would
    /// force the same id to win at every depth, which is wrong for
    /// almost every layout. Callers that want to fix a descendant's
    /// scheme too must set it explicitly via [`with_frozen_scheme`] on
    /// the returned context.
    ///
    /// The winner observer is dropped here for the same reason: it
    /// applies only at the level the caller installed it at; firing it
    /// for every descendant would conflate per-column winners with
    /// per-(child of winner) cascading winners.
    pub(super) fn descend_with_scheme(mut self, id: SchemeId, child_index: usize) -> Self {
        self.allowed_cascading = self
            .allowed_cascading
            .checked_sub(1)
            .vortex_expect("cannot descend: cascade depth exhausted");
        self.cascade_history.push((id, child_index));
        self.frozen_scheme = None;
        self.winner_observer = None;
        self
    }

    /// Returns the caller-supplied scheme winner for this compression
    /// site, if any.
    pub fn frozen_scheme(&self) -> Option<SchemeId> {
        self.frozen_scheme
    }

    /// Sets a scheme winner for this compression site. When the chosen
    /// scheme is registered and applicable to the canonical input, the
    /// cascade bypasses its selection pass and dispatches directly to
    /// it. Otherwise the cascade falls through to normal selection.
    ///
    /// Intended for callers that have observed the winning scheme on a
    /// prior, structurally-equivalent array (e.g. consecutive fragments
    /// of the same column in a streaming writer) and want to avoid the
    /// per-call selection + stats-merge work.
    #[must_use]
    pub fn with_frozen_scheme(mut self, scheme_id: SchemeId) -> Self {
        self.frozen_scheme = Some(scheme_id);
        self
    }

    /// Returns the caller-supplied winner observer, if any.
    pub fn winner_observer(&self) -> Option<&WinnerObserver> {
        self.winner_observer.as_ref()
    }

    /// Sets an observer that fires when the cascade determines a winner
    /// at this compression site (whether via the
    /// [`with_frozen_scheme`](Self::with_frozen_scheme) fast path or
    /// via normal selection). The observer is dropped on cascade
    /// descent so it fires only at the level the caller installed it.
    #[must_use]
    pub fn with_winner_observer(mut self, observer: WinnerObserver) -> Self {
        self.winner_observer = Some(observer);
        self
    }
}

/// Display wrapper for a cascade ancestry path.
struct CascadePath<'a>(&'a [(SchemeId, usize)]);

impl fmt::Display for CascadePath<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("root");
        }

        for (index, (scheme_id, child_index)) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str(" > ")?;
            }

            if *scheme_id == ROOT_SCHEME_ID {
                write!(f, "root[{child_index}]")?;
            } else {
                write!(f, "{scheme_id}[{child_index}]")?;
            }
        }

        Ok(())
    }
}
