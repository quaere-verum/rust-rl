//! Exact, bounded, prepare-owned decode witnesses. Every load still reads.
use super::*;
use std::{mem::size_of, sync::Weak};

const CACHE_BYTES: usize = 1024 * 1024;
const HEADER_BYTES: usize = 2 * 1024;
const SLOTS: usize = 5;

struct Token {
    root: PathBuf,
    manifest: Weak<RunManifest>,
    limits: ReadLimits,
    entry: IndexEntry,
}
impl Token {
    fn matches(&self, source: &ChunkSource) -> bool {
        self.root == source.root
            && self.manifest.ptr_eq(&Arc::downgrade(&source.manifest))
            && self.limits == source.limits
            && self.entry == source.entry
    }
    fn charge(source: &ChunkSource) -> usize {
        // Clones cannot need more capacity than the source strings/path.
        // Pre-evict with those actual capacities; store the clone's actual
        // capacities below. Charge the weak origin allocation too; it retains
        // no manifest-owned strings/trees.
        size_of::<Self>()
            + size_of::<RunManifest>()
            + 2 * size_of::<usize>()
            + source.root.capacity()
            + source.entry.file.capacity()
            + source.entry.model_role.capacity()
            + [
                &source.entry.probe_set,
                &source.entry.trajectory,
                &source.entry.checkpoint_id,
            ]
            .into_iter()
            .flatten()
            .map(String::capacity)
            .sum::<usize>()
    }
    fn new(source: &ChunkSource) -> Self {
        Self {
            root: source.root.clone(),
            manifest: Arc::downgrade(&source.manifest),
            limits: source.limits.clone(),
            entry: source.entry.clone(),
        }
    }
    fn owned_charge(&self) -> usize {
        size_of::<Self>()
            + size_of::<RunManifest>()
            + 2 * size_of::<usize>()
            + self.root.capacity()
            + self.entry.file.capacity()
            + self.entry.model_role.capacity()
            + [
                &self.entry.probe_set,
                &self.entry.trajectory,
                &self.entry.checkpoint_id,
            ]
            .into_iter()
            .flatten()
            .map(String::capacity)
            .sum::<usize>()
    }
}
struct Witness {
    source: Token,
    encoded: Vec<u8>,
    chunk: Arc<Chunk>,
    decoded_charge: usize,
    owned_charge: usize,
}

/// An optional exact-byte decode cache for one bounded operation. It owns at
/// most 1 MiB, including numeric/encoded capacities and conservative metadata.
/// Hits still perform the ordinary retention, admission, file/page and CRC
/// checks. This cache saves decoding, never reads or logical caller charges.
///
/// Keep this owner inside the caller's decoder scratch lease. An unrecognized
/// or larger-than-2-KiB header clears it before ordinary decoding; inability to
/// retain a witness changes neither successful results nor errors/admission.
pub struct DecodedChunkCache {
    // Empty slots precede live witnesses ordered from least to most recent.
    slots: [Option<Witness>; SLOTS],
    decodes: u64,
}
impl Default for DecodedChunkCache {
    fn default() -> Self {
        Self {
            slots: std::array::from_fn(|_| None),
            decodes: 0,
        }
    }
}
impl DecodedChunkCache {
    /// Decoder invocations, including refused/invalid decoded payloads.
    pub fn decodes(&self) -> u64 {
        self.decodes
    }
    /// Same admission/error order as `ChunkSource::load_if_within_charge`.
    pub fn load_if_within_charge(
        &mut self,
        source: &ChunkSource,
        limit: usize,
    ) -> Result<Option<Arc<Chunk>>> {
        let Some(bytes) =
            read_chunk_admitted(&source.root, &source.limits, &source.entry, Some(limit))?
        else {
            return Ok(None);
        };
        if let Some(index) = self.slots.iter().rposition(|slot| {
            slot.as_ref()
                .is_some_and(|witness| witness.source.matches(source) && witness.encoded == bytes)
        }) {
            let witness = self.slots[index].as_ref().expect("matched witness");
            let chunk = (witness.decoded_charge <= limit).then(|| witness.chunk.clone());
            // Exact validated reads touch recency even on decoded admission
            // refusal. Moving slots creates no additional retained witness.
            self.slots[index..].rotate_left(1);
            return Ok(chunk);
        }
        let header = small_header(&bytes);
        if header.is_none() {
            self.clear();
        }
        self.decodes += 1;
        let Some(chunk) = decode_chunk_admitted(
            &bytes,
            &source.manifest,
            &source.limits,
            &source.entry,
            Some(limit),
        )?
        else {
            return Ok(None);
        };
        let chunk = Arc::new(chunk);
        if let Some(metadata) = header.and_then(|header| retained_metadata(&chunk, header)) {
            // Decoder-created arrays have exact numeric allocation lengths.
            // Still count actual capacities rather than relying on that fact.
            let numeric = chunk
                .arrays
                .values()
                .map(|a| match &a.values {
                    Values::F32(v) => v.capacity() * size_of::<f32>(),
                    Values::F64(v) => v.capacity() * size_of::<f64>(),
                    Values::U64(v) => v.capacity() * size_of::<u64>(),
                    Values::U8(v) => v.capacity(),
                })
                .sum::<usize>();
            let charge = size_of::<Witness>()
                + Token::charge(source)
                + bytes.capacity()
                + numeric
                + metadata;
            if charge <= CACHE_BYTES - size_of::<Self>() {
                // Drop old ownership before cloning a token or transferring
                // decoder scratch into a witness; no old/new cache overlap.
                for index in 0..SLOTS {
                    if self.owned_bytes() + charge <= CACHE_BYTES {
                        break;
                    }
                    self.slots[index] = None;
                }
                // If all slots were occupied, retire the oldest; otherwise
                // this is already empty. Keep holes before surviving entries.
                self.slots[0] = None;
                self.slots.rotate_left(1);
                let token = Token::new(source);
                // Confirm the exact clone capacities underlying pre-insertion
                // eviction, and record actual token ownership in the slot.
                let actual_charge = charge - Token::charge(source) + token.owned_charge();
                debug_assert!(actual_charge <= charge);
                self.slots[SLOTS - 1] = Some(Witness {
                    source: token,
                    encoded: bytes,
                    decoded_charge: chunk.retained_charge(),
                    chunk: chunk.clone(),
                    owned_charge: actual_charge,
                });
            }
        }
        Ok(Some(chunk))
    }
    fn clear(&mut self) {
        for slot in &mut self.slots {
            *slot = None;
        }
    }
    fn owned_bytes(&self) -> usize {
        size_of::<Self>()
            + self
                .slots
                .iter()
                .flatten()
                .map(|w| w.owned_charge)
                .sum::<usize>()
    }
}

fn retained_metadata(chunk: &Chunk, header: usize) -> Option<usize> {
    // These records contain strings and flat maps, never arbitrary JSON or
    // nested Vecs. Typed decoding creates <=8 shape capacity per descriptor.
    // 2048 per entry covers even a one-entry x64 BTreeMap node (11 key/value
    // slots and 12 edges) plus shape capacity; 4H covers all decoded strings.
    // The fixed term includes Chunk, Record, Arc and tree roots. Unsupported
    // kinds remain fully decoded/validated, but are not retained here.
    let maps = match &chunk.record {
        Record::Metrics(record) => record.values.len() + record.unavailable.len(),
        Record::ProbeSet(record) => record.histories.len(),
        Record::Policy(_) | Record::Critic(_) | Record::Empirical(_) => 0,
        _ => return None,
    };
    Some(4096 + 4 * header + 2048 * (maps + chunk.arrays.len()))
}
fn small_header(bytes: &[u8]) -> Option<usize> {
    // For the pinned x64 serde/tree layouts, 256H + 64 KiB bounds header,
    // parser, shapes and serialization scratch, including arbitrary JSON
    // configuration on malformed/mismatched sources. H<=2 KiB keeps this at
    // 576 KiB. Larger inputs lose witnesses before the ordinary parser runs.
    if bytes.len() < 12 || &bytes[..8] != b"RLDIAG01" {
        return None;
    }
    let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
    (len <= HEADER_BYTES && len <= bytes.len() - 12).then_some(len)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::action::ActionSpace;

    #[test]
    fn task3a2_recency_promotes_hits_and_preserves_order_after_byte_eviction() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("recency");
        let manifest = RunManifest::new(
            "recency",
            "fixture",
            "fixture",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap();
        let mut writer = Recorder::create(
            &root,
            manifest,
            StorageConfig {
                compress: false,
                ..Default::default()
            },
        )
        .unwrap();
        // Small slot-pressure keys, followed by A/B/C, large L, D, E for the
        // independent byte-pressure trace below. All sources remain eligible.
        let sizes: Vec<_> = std::iter::repeat_n(0, SLOTS + 1)
            .chain([
                96 * 1024,
                96 * 1024,
                96 * 1024,
                320 * 1024,
                64 * 1024,
                96 * 1024,
            ])
            .collect();
        for (revision, &bytes) in sizes.iter().enumerate() {
            let arrays = if bytes == 0 {
                BTreeMap::new()
            } else {
                BTreeMap::from([(
                    "bytes".into(),
                    Array::u8(vec![bytes], vec![42; bytes]).unwrap(),
                )])
            };
            writer
                .append(&Chunk {
                    context: RecordContext::new("recency", revision as u64),
                    record: Record::Metrics(MetricRecord {
                        values: BTreeMap::new(),
                        unavailable: BTreeMap::new(),
                    }),
                    arrays,
                })
                .unwrap();
        }
        writer.finish().unwrap();
        drop(writer);
        let recording = Recording::open(&root, ReadLimits::default()).unwrap();
        let sources: Vec<_> = (0..sizes.len())
            .map(|i| recording.source(i).unwrap())
            .collect();
        let load = |cache: &mut DecodedChunkCache, index: usize| {
            cache
                .load_if_within_charge(&sources[index], 4 * 1024 * 1024)
                .unwrap()
                .unwrap()
        };
        let mut cache = DecodedChunkCache::default();
        let held: Vec<_> = (0..SLOTS).map(|i| load(&mut cache, i)).collect();
        assert!(Arc::ptr_eq(&held[0], &load(&mut cache, 0)));
        // A validated byte-identical refusal also marks the witness recent.
        assert!(
            cache
                .load_if_within_charge(&sources[1], held[1].retained_charge() - 1)
                .unwrap()
                .is_none()
        );
        drop(load(&mut cache, SLOTS));
        assert_eq!(cache.decodes(), (SLOTS + 1) as u64);
        assert!(
            Arc::ptr_eq(&held[0], &load(&mut cache, 0)),
            "a recent hit must survive insertion"
        );
        assert!(
            Arc::ptr_eq(&held[1], &load(&mut cache, 1)),
            "a validated refusal must preserve recency"
        );
        assert!(
            !Arc::ptr_eq(&held[2], &load(&mut cache, 2)),
            "the least recent key must be evicted"
        );
        assert_eq!(cache.decodes(), (SLOTS + 2) as u64);

        let start = SLOTS + 1;
        let mut cache = DecodedChunkCache::default();
        let a = load(&mut cache, start);
        let b = load(&mut cache, start + 1);
        let bw = Arc::downgrade(&b);
        drop(b);
        let c = load(&mut cache, start + 2);
        let cw = Arc::downgrade(&c);
        drop(c);
        assert!(Arc::ptr_eq(&a, &load(&mut cache, start)));
        let large = load(&mut cache, start + 3);
        let lw = Arc::downgrade(&large);
        drop(large);
        assert!(
            bw.upgrade().is_none() && cw.upgrade().is_none(),
            "byte pressure must evict B then C before retaining L"
        );
        assert!(cache.owned_bytes() <= CACHE_BYTES);
        let d = load(&mut cache, start + 4);
        assert!(Arc::ptr_eq(&a, &load(&mut cache, start)));
        drop(load(&mut cache, start + 5));
        assert!(
            lw.upgrade().is_none(),
            "holes must not disturb L/D/A recency ordering"
        );
        assert!(Arc::ptr_eq(&a, &load(&mut cache, start)));
        assert!(Arc::ptr_eq(&d, &load(&mut cache, start + 4)));
        assert_eq!(cache.decodes(), 6);
        assert!(cache.owned_bytes() <= CACHE_BYTES);
        let aw = Arc::downgrade(&a);
        let dw = Arc::downgrade(&d);
        drop(a);
        drop(d);
        drop(cache);
        assert!(aw.upgrade().is_none() && dw.upgrade().is_none());
    }

    #[test]
    fn limits_origin_header_errors_and_drop_do_not_retain_incompatible_owners() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path().join("run");
        let manifest = RunManifest::new(
            "cache",
            "fixture",
            "fixture",
            1,
            ActionSpace::Discrete { count: 2 },
            0.9,
        )
        .unwrap();
        let mut writer = Recorder::create(&root, manifest, StorageConfig::default()).unwrap();
        writer
            .append(&Chunk {
                context: RecordContext::new("cache", 0),
                record: Record::Metrics(MetricRecord {
                    values: BTreeMap::new(),
                    unavailable: BTreeMap::new(),
                }),
                arrays: BTreeMap::new(),
            })
            .unwrap();
        writer.finish().unwrap();
        drop(writer);
        let recording = Recording::open(&root, ReadLimits::default()).unwrap();
        let source = recording.source(0).unwrap();
        let origin = Arc::downgrade(&source.manifest);
        let mut cache = DecodedChunkCache::default();
        let first = cache.load_if_within_charge(&source, 4096).unwrap().unwrap();
        let mut narrow = source.clone();
        narrow.limits.max_metadata_bytes -= 1;
        let second = cache.load_if_within_charge(&narrow, 4096).unwrap().unwrap();
        assert!(!Arc::ptr_eq(&first, &second));
        assert_eq!(cache.decodes(), 2);
        assert!(cache.owned_bytes() <= CACHE_BYTES);
        let file = root.join(&source.entry.file);
        let original = fs::read(&file).unwrap();
        for oversized in [false, true] {
            let mut malformed = original.clone();
            if oversized {
                malformed[8..12].copy_from_slice(&((HEADER_BYTES + 1) as u32).to_le_bytes());
            } else {
                malformed[0] ^= 1;
            }
            fs::write(&file, &malformed).unwrap();
            let mut bad = source.clone();
            bad.entry.checksum = crc32fast::hash(&malformed);
            assert_eq!(
                bad.load_if_within_charge(4096).unwrap_err().to_string(),
                cache
                    .load_if_within_charge(&bad, 4096)
                    .unwrap_err()
                    .to_string()
            );
            assert_eq!(cache.owned_bytes(), size_of::<DecodedChunkCache>());
            fs::write(&file, &original).unwrap();
            drop(cache.load_if_within_charge(&source, 4096).unwrap().unwrap());
        }
        let weak = Arc::downgrade(&cache.slots.iter().flatten().next().unwrap().chunk);
        drop(first);
        drop(second);
        drop(narrow);
        drop(source);
        drop(recording);
        assert!(
            origin.upgrade().is_none(),
            "cache must not keep an arbitrary manifest alive"
        );
        assert!(weak.upgrade().is_some());
        drop(cache);
        assert!(weak.upgrade().is_none());
    }
}
