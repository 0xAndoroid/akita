//! Selected-row verifier catalog views.
//!
//! A view carries the complete catalog's row identities but the parameters of
//! only the rows a verifier needs, in a bincode encoding a guest decodes
//! without JSON parsing. Loading it repeats the JSON catalog's semantic audit
//! for every carried row and recomputes the complete catalog digest, so a view
//! binds the same transcript identity as the catalog it was cut from.

use super::{
    catalog_digest, policy_digest, AkitaError, CatalogCoverage, PlannerPolicy,
    ScheduleCatalogArtifactRowV1, SparseChallengeConfig, ValidatedScheduleCatalog,
    AKITA_INSTANCE_DESCRIPTOR_VERSION, MAX_TRUSTED_CATALOG_ROWS,
    MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES,
};
use akita_types::{OpeningScheduleSelection, ScheduleRowDigest};
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

const VERIFIER_MAGIC: [u8; 8] = *b"AKSCVFY1";

#[derive(Serialize, Deserialize)]
struct BinaryVerifierCatalog {
    magic: [u8; 8],
    protocol_epoch: u32,
    policy_digest: [u8; 32],
    family_name: String,
    row_digests: Vec<[u8; 32]>,
    rows: Vec<ScheduleCatalogArtifactRowV1>,
}

impl ValidatedScheduleCatalog {
    /// Encode a verifier view retaining only the requested rows while
    /// preserving the complete catalog's transcript identity. Omitted rows
    /// remain opaque digest commitments. The view serves verification with an
    /// existing setup; it cannot size a new setup or be exported as a complete
    /// JSON catalog.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] for an empty, oversized, or
    /// unresolvable selection, or an oversized encoding.
    pub fn to_verifier_view(
        &self,
        selections: &[OpeningScheduleSelection],
    ) -> Result<Vec<u8>, AkitaError> {
        if selections.is_empty() || selections.len() > MAX_TRUSTED_CATALOG_ROWS {
            return Err(AkitaError::InvalidSetup(
                "verifier catalog selection is outside row limits".into(),
            ));
        }
        let selected = selections
            .iter()
            .map(|selection| selection.row_digest)
            .collect::<BTreeSet<_>>();
        let rows = selected
            .into_iter()
            .map(|row_digest| {
                self.resolve_selection(OpeningScheduleSelection { row_digest })
                    .map(|row| ScheduleCatalogArtifactRowV1 {
                        schedule: row.schedule().clone(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let row_digests = match &self.coverage {
            CatalogCoverage::Complete => self
                .rows()
                .map(|row| *row.selection().row_digest.as_bytes())
                .collect(),
            CatalogCoverage::Selected { row_digests } => row_digests.clone(),
        };
        let artifact = BinaryVerifierCatalog {
            magic: VERIFIER_MAGIC,
            protocol_epoch: AKITA_INSTANCE_DESCRIPTOR_VERSION,
            policy_digest: self.policy_digest,
            family_name: self.family_name.clone(),
            row_digests,
            rows,
        };
        let bytes = bincode::serde::encode_to_vec(&artifact, bincode::config::standard()).map_err(
            |error| AkitaError::InvalidSetup(format!("cannot encode verifier catalog: {error}")),
        )?;
        if bytes.len() > MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES {
            return Err(AkitaError::InvalidSetup(
                "verifier catalog exceeds byte limit".into(),
            ));
        }
        Ok(bytes)
    }

    /// Load a verifier view supplied by the application's trusted setup path.
    ///
    /// Checks format, configuration binding, every carried row's semantic
    /// invariants, and that each carried row is a member of the catalog
    /// commitment. Omitted identities do not resolve to parameters. The caller
    /// owns provenance.
    ///
    /// # Errors
    ///
    /// Returns [`AkitaError::InvalidSetup`] for a malformed view, a
    /// configuration mismatch, or a carried row outside the commitment.
    pub fn from_verifier_view(
        bytes: &[u8],
        expected_family_name: &str,
        policy: &PlannerPolicy,
        ring_challenge_config: impl Fn(usize) -> Result<SparseChallengeConfig, AkitaError>,
    ) -> Result<Self, AkitaError> {
        if bytes.len() > MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES {
            return Err(AkitaError::InvalidSetup(
                "verifier catalog exceeds byte limit".into(),
            ));
        }
        let (artifact, consumed): (BinaryVerifierCatalog, usize) =
            bincode::serde::decode_from_slice(
                bytes,
                bincode::config::standard().with_limit::<MAX_TRUSTED_SCHEDULE_ARTIFACT_BYTES>(),
            )
            .map_err(|error| {
                AkitaError::InvalidSetup(format!("invalid verifier catalog: {error}"))
            })?;
        if consumed != bytes.len()
            || artifact.magic != VERIFIER_MAGIC
            || artifact.protocol_epoch != AKITA_INSTANCE_DESCRIPTOR_VERSION
        {
            return Err(AkitaError::InvalidSetup(
                "unsupported verifier catalog format or trailing bytes".into(),
            ));
        }
        if artifact.family_name != expected_family_name
            || artifact.policy_digest != policy_digest(policy)
        {
            return Err(AkitaError::InvalidSetup(
                "verifier catalog does not match runtime config".into(),
            ));
        }
        if artifact.row_digests.is_empty()
            || artifact.row_digests.len() > MAX_TRUSTED_CATALOG_ROWS
            || artifact.rows.is_empty()
            || artifact.rows.len() > artifact.row_digests.len()
            || artifact
                .row_digests
                .windows(2)
                .any(|pair| pair[0] >= pair[1])
        {
            return Err(AkitaError::InvalidSetup(
                "verifier catalog identities are not a bounded canonical list".into(),
            ));
        }
        let rows = artifact
            .rows
            .into_iter()
            .map(ScheduleCatalogArtifactRowV1::into_profile_and_schedule)
            .collect::<Result<Vec<_>, _>>()?;
        // The JSON catalog's semantic audit owns the carried rows. No supplied
        // identity can replace the digest computed from a row's validated contents.
        let mut catalog = Self::try_new(artifact.family_name, rows, policy, ring_challenge_config)?;
        for row in catalog.rows() {
            if artifact
                .row_digests
                .binary_search(row.selection().row_digest.as_bytes())
                .is_err()
            {
                return Err(AkitaError::InvalidSetup(
                    "selected row is absent from the catalog commitment".into(),
                ));
            }
        }
        catalog.catalog_digest = catalog_digest(
            &catalog.family_name,
            catalog.policy_digest,
            artifact
                .row_digests
                .iter()
                .copied()
                .map(ScheduleRowDigest::from_bytes),
        );
        catalog.coverage = CatalogCoverage::Selected {
            row_digests: artifact.row_digests,
        };
        Ok(catalog)
    }
}
