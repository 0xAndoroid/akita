//! Spongefish backend selected by Akita's transcript feature.

/// Sponge backend selected by the active transcript feature.
///
/// Exactly one transcript backend feature must be active in the complete PCS graph.
#[cfg(all(feature = "transcript-blake2b", not(feature = "blake2-inline")))]
pub type TranscriptSponge = spongefish::instantiations::Blake2b512;

/// Sponge backend selected by the active transcript feature, over the inline hasher.
#[cfg(all(feature = "transcript-blake2b", feature = "blake2-inline"))]
pub type TranscriptSponge = spongefish::instantiations::hash::Hash<
    jolt_inlines_blake2::digest_adapter::Blake2b<jolt_inlines_blake2::digest_adapter::U64>,
>;

/// Sponge backend selected by the active transcript feature.
#[cfg(feature = "transcript-keccak")]
pub type TranscriptSponge = spongefish::instantiations::Keccak;
