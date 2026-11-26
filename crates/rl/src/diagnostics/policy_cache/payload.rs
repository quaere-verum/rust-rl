use super::{CacheEntry, CacheLimits, CacheLoadTiming, hash_bytes, read_limited};
use crate::{
    Result,
    core::ensure,
    diagnostics::{
        codec,
        policy_slice::GridEligibility,
        policy_surface::{
            EvaluatedPolicyBatch, POLICY_SURFACE_VERSION, PolicySurface, SurfaceBatch,
        },
        schema::RunManifest,
        storage::IndexEntry,
        visualization::EnvironmentSemantics,
    },
};
use sha2::{Digest, Sha256};
use std::{
    fs::OpenOptions,
    io::{BufWriter, Write},
    path::Path,
};

const MAGIC: &[u8; 8] = b"RLPOLC01";
struct Writer {
    file: BufWriter<std::fs::File>,
    hash: Sha256,
    count: u64,
    limit: u64,
}
impl Writer {
    fn put(&mut self, bytes: &[u8]) -> Result<()> {
        ensure(
            bytes.len() as u64 <= self.limit.saturating_sub(self.count),
            "encoded policy frame exceeds byte limit",
        )?;
        self.file.write_all(bytes)?;
        self.hash.update(bytes);
        self.count += bytes.len() as u64;
        Ok(())
    }
    fn number(&mut self, value: usize) -> Result<()> {
        self.put(&(value as u64).to_le_bytes())
    }
}
pub(super) fn write(
    path: &Path,
    surface: &PolicySurface,
    limits: CacheLimits,
) -> Result<(u64, String)> {
    let mut writer = Writer {
        file: BufWriter::with_capacity(
            64 * 1024,
            OpenOptions::new().write(true).create_new(true).open(path)?,
        ),
        hash: Sha256::new(),
        count: 0,
        limit: limits.max_frame_bytes as u64,
    };
    writer.put(MAGIC)?;
    writer.number(surface.batches.len())?;
    for batch in &surface.batches {
        writer.number(batch.first_row)?;
        writer.number(batch.rows)?;
        for chunk in [&batch.evaluated.probes, &batch.evaluated.policy] {
            let bytes = codec::encode(
                chunk,
                limits.max_chunk_bytes,
                limits.max_metadata_bytes,
                true,
            )?;
            writer.number(bytes.len())?;
            writer.put(&bytes)?;
        }
        writer.number(batch.projected.len())?;
        for value in &batch.projected {
            writer.put(&value.to_le_bytes())?;
        }
    }
    writer.file.flush()?;
    writer.file.get_ref().sync_all()?;
    Ok((writer.count, format!("{:x}", writer.hash.finalize())))
}
struct Reader<'a> {
    bytes: &'a [u8],
    offset: usize,
}
impl<'a> Reader<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        ensure(
            count <= self.bytes.len().saturating_sub(self.offset),
            "truncated policy frame",
        )?;
        let value = &self.bytes[self.offset..self.offset + count];
        self.offset += count;
        Ok(value)
    }
    fn number(&mut self) -> Result<usize> {
        usize::try_from(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
        .map_err(|_| crate::Error::Shape("policy frame dimension overflow".into()))
    }
}
pub(super) fn load(
    path: &Path,
    entry: &CacheEntry,
    manifest: &RunManifest,
    limits: CacheLimits,
    started: std::time::Instant,
) -> Result<(PolicySurface, CacheLoadTiming)> {
    ensure(
        entry.retained_bytes <= limits.max_surface_bytes,
        "policy surface exceeds retained read limit",
    )?;
    let bytes = read_limited(path, limits.max_frame_bytes)?;
    ensure(
        hash_bytes(&bytes) == entry.payload_sha256,
        "policy cache payload checksum mismatch",
    )?;
    let read_micros = started.elapsed().as_micros() as u64;
    let decoding = std::time::Instant::now();
    let mut reader = Reader {
        bytes: &bytes,
        offset: 0,
    };
    ensure(
        reader.take(8)? == MAGIC && reader.number()? == entry.batches,
        "incompatible policy cache payload header",
    )?;
    ensure(
        entry
            .batches
            .saturating_mul(std::mem::size_of::<SurfaceBatch>())
            <= limits.max_surface_bytes,
        "policy batch metadata exceeds retained read limit",
    )?;
    let mut surface = PolicySurface {
        version: POLICY_SURFACE_VERSION,
        spec: entry.slice.clone(),
        action_space: manifest.action_space.clone(),
        batches: Vec::with_capacity(entry.batches),
        rows: entry.rows,
    };
    let mut retained = surface.owned_bytes();
    for _ in 0..entry.batches {
        let first_row = reader.number()?;
        let rows = reader.number()?;
        let mut decode = || -> Result<_> {
            let len = reader.number()?;
            ensure(
                len <= limits.max_chunk_bytes,
                "encoded cache chunk exceeds read limit",
            )?;
            let chunk = codec::decode_if_within_charge(
                reader.take(len)?,
                limits.max_chunk_bytes,
                limits.max_metadata_bytes,
                limits.max_surface_bytes.saturating_sub(retained),
            )?
            .ok_or_else(|| {
                crate::Error::Config("decoded cache chunk exceeds retained read limit".into())
            })?;
            retained = retained.saturating_add(chunk.retained_charge());
            Ok(chunk)
        };
        let probes = decode()?;
        let policy = decode()?;
        let count = reader.number()?;
        ensure(
            rows > 0
                && rows.checked_mul(surface.action_count()) == Some(count)
                && count
                    .checked_mul(8)
                    .is_some_and(|n| n <= limits.max_surface_bytes.saturating_sub(retained)),
            "invalid policy projection dimensions or retained byte limit",
        )?;
        let projected = reader
            .take(count * 8)?
            .chunks_exact(8)
            .map(|v| f64::from_le_bytes(v.try_into().expect("eight bytes")))
            .collect::<Vec<_>>();
        let batch = SurfaceBatch {
            first_row,
            evaluated: EvaluatedPolicyBatch { probes, policy },
            projected,
            rows,
        };
        retained = retained
            .saturating_add(batch.projected.capacity().saturating_mul(8))
            .saturating_add(std::mem::size_of::<SurfaceBatch>());
        ensure(
            retained <= limits.max_surface_bytes,
            "decoded policy frame exceeds retained byte limit",
        )?;
        surface.batches.push(batch);
    }
    ensure(
        reader.offset == bytes.len(),
        "unreferenced trailing policy cache bytes",
    )?;
    validate(&surface, &entry.source, manifest, limits.max_surface_bytes)?;
    Ok((
        surface,
        CacheLoadTiming {
            read_micros,
            decode_micros: decoding.elapsed().as_micros() as u64,
        },
    ))
}
pub(super) fn validate(
    surface: &PolicySurface,
    source: &IndexEntry,
    manifest: &RunManifest,
    limit: usize,
) -> Result<()> {
    let config = surface
        .spec()
        .to_config(EnvironmentSemantics::new(manifest))?;
    ensure(
        surface.version() == POLICY_SURFACE_VERSION
            && config.grid_eligibility() == GridEligibility::ContinuousFeedforward
            && config.len()? == surface.rows()
            && surface.action_space() == &manifest.action_space
            && surface.owned_bytes() <= limit,
        "invalid policy surface contract or retained limit",
    )?;
    let mut first = 0usize;
    for batch in &surface.batches {
        let checked = batch.evaluated.checked(surface.action_space())?;
        ensure(
            batch.first_row == first
                && batch.rows > 0
                && batch.rows == checked.rows()
                && batch.rows.checked_mul(surface.action_count()) == Some(batch.projected.len())
                && batch.projected.iter().all(|v| v.is_finite()),
            "invalid policy surface batch or projections",
        )?;
        for chunk in [&batch.evaluated.probes, &batch.evaluated.policy] {
            let identity = &chunk.context.identity;
            ensure(
                identity.run_id == manifest.run_id
                    && identity.model_role == source.model_role
                    && identity.model_role == surface.spec().model_role
                    && identity.learner_update_revision == source.revision
                    && identity.checkpoint_id == source.checkpoint_id,
                "policy surface checkpoint provenance differs",
            )?;
        }
        first = first
            .checked_add(batch.rows)
            .ok_or_else(|| crate::Error::Shape("policy row count overflow".into()))?;
        ensure(first <= surface.rows(), "policy batch exceeds surface rows")?;
    }
    ensure(
        first == surface.rows(),
        "incomplete policy numerical surface",
    )?;
    for row in 0..surface.rows() {
        ensure(
            surface.input(row)? == config.observation_at(row)?,
            "cached input differs from exact slice grid",
        )?;
    }
    Ok(())
}
