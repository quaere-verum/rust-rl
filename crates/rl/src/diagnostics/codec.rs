use super::schema::*;
use crate::{Result, core::ensure};
use flate2::{Compression, read::DeflateDecoder, write::DeflateEncoder};
use serde::{Deserialize, Serialize};
use std::{
    collections::BTreeMap,
    io::{Read, Write},
};

const MAGIC: &[u8; 8] = b"RLDIAG01";
#[derive(Serialize, Deserialize)]
struct Descriptor {
    dtype: DType,
    shape: Vec<usize>,
    offset: usize,
    bytes: usize,
}
#[derive(Serialize, Deserialize)]
struct Header {
    schema_version: u32,
    context: RecordContext,
    record: Record,
    arrays: BTreeMap<String, Descriptor>,
    raw_bytes: usize,
    compressed: bool,
}
#[derive(Serialize)]
struct BorrowedHeader<'a> {
    schema_version: u32,
    context: &'a RecordContext,
    record: &'a Record,
    arrays: BTreeMap<String, Descriptor>,
    raw_bytes: usize,
    compressed: bool,
}

pub(super) fn bounded_json(value: &impl Serialize, limit: usize) -> Result<Vec<u8>> {
    struct Limited {
        bytes: Vec<u8>,
        limit: usize,
    }
    impl Write for Limited {
        fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
            if bytes.len() > self.limit.saturating_sub(self.bytes.len()) {
                return Err(std::io::Error::other(
                    "diagnostic metadata exceeds configured limit",
                ));
            }
            let needed = self.bytes.len() + bytes.len();
            if needed > self.bytes.capacity() {
                let target = needed
                    .max(self.bytes.capacity().saturating_mul(2))
                    .min(self.limit);
                self.bytes
                    .try_reserve_exact(target - self.bytes.len())
                    .map_err(std::io::Error::other)?;
            }
            self.bytes.extend_from_slice(bytes);
            Ok(bytes.len())
        }
        fn flush(&mut self) -> std::io::Result<()> {
            Ok(())
        }
    }
    let mut writer = Limited {
        bytes: vec![],
        limit,
    };
    serde_json::to_writer(&mut writer, value)?;
    Ok(writer.bytes)
}
pub(super) fn encode(
    chunk: &Chunk,
    max_bytes: usize,
    max_header: usize,
    compress: bool,
) -> Result<Vec<u8>> {
    let raw_limit = if matches!(chunk.record, Record::Typed(_)) {
        max_bytes.min(super::typed::MAX_TYPED_BYTES)
    } else {
        max_bytes
    };
    ensure(
        chunk.array_bytes() <= raw_limit,
        "diagnostic arrays exceed chunk budget",
    )?;
    let mut raw = Vec::with_capacity(chunk.array_bytes());
    let mut arrays = BTreeMap::new();
    for (name, array) in &chunk.arrays {
        let offset = raw.len();
        match &array.values {
            Values::F32(v) => {
                for value in v {
                    raw.extend_from_slice(&value.to_le_bytes());
                }
            }
            Values::F64(v) => {
                for value in v {
                    raw.extend_from_slice(&value.to_le_bytes());
                }
            }
            Values::U64(v) => {
                for value in v {
                    raw.extend_from_slice(&value.to_le_bytes());
                }
            }
            Values::U8(v) => raw.extend_from_slice(v),
        }
        arrays.insert(
            name.clone(),
            Descriptor {
                dtype: array.dtype(),
                shape: array.shape().into(),
                offset,
                bytes: array.byte_len(),
            },
        );
    }
    let raw_bytes = raw.len();
    let mut compressed = false;
    if compress && raw.len() >= 1024 {
        let mut encoder = DeflateEncoder::new(Vec::new(), Compression::fast());
        encoder.write_all(&raw)?;
        let encoded = encoder.finish()?;
        if encoded.len() + 32 < raw.len() {
            raw = encoded;
            compressed = true;
        }
    }
    let header = bounded_json(
        &BorrowedHeader {
            schema_version: SCHEMA_VERSION,
            context: &chunk.context,
            record: &chunk.record,
            arrays,
            raw_bytes,
            compressed,
        },
        max_header,
    )?;
    let total = 12usize
        .checked_add(header.len())
        .and_then(|n| n.checked_add(raw.len()))
        .ok_or_else(|| std::io::Error::other("diagnostic chunk overflow"))?;
    ensure(
        total <= max_bytes,
        "diagnostic encoded chunk exceeds chunk budget",
    )?;
    let mut bytes = Vec::with_capacity(total);
    bytes.extend_from_slice(MAGIC);
    bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
    bytes.extend_from_slice(&header);
    bytes.extend_from_slice(&raw);
    Ok(bytes)
}
pub(super) fn decode(bytes: &[u8], max_bytes: usize, max_header: usize) -> Result<Chunk> {
    decode_versioned(bytes, max_bytes, max_header, SCHEMA_VERSION)
}
fn decode_versioned(
    bytes: &[u8],
    max_bytes: usize,
    max_header: usize,
    max_version: u32,
) -> Result<Chunk> {
    Ok(
        decode_admitted(bytes, max_bytes, max_header, max_version, None)?
            .expect("unlimited admission"),
    )
}
pub(super) fn decode_if_within_charge(
    bytes: &[u8],
    max_bytes: usize,
    max_header: usize,
    charge_limit: usize,
) -> Result<Option<Chunk>> {
    decode_admitted(
        bytes,
        max_bytes,
        max_header,
        SCHEMA_VERSION,
        Some(charge_limit),
    )
}
fn validate_render_descriptors(header: &Header) -> Result<()> {
    let Record::Trajectory(record) = &header.record else {
        return Ok(());
    };
    let Some(track) = &record.render else {
        return Ok(());
    };
    let limits = super::render::RenderLimits::default();
    ensure(
        header.schema_version >= 4 && track.version == super::render::RENDER_VERSION,
        "unsupported render track/schema version",
    )?;
    let get = |key: &str| {
        header
            .arrays
            .get(key)
            .ok_or_else(|| crate::Error::Shape("missing render descriptor".into()))
    };
    let observations = get(&record.observations)?;
    let rewards = get(&record.rewards)?;
    let offsets = get(&track.offsets)?;
    let data = get(&track.data)?;
    ensure(
        observations.shape.len() == 2 && !rewards.shape.is_empty(),
        "invalid render trajectory descriptor rank",
    )?;
    let frames = observations.shape[0];
    ensure(
        frames > 0
            && frames <= limits.max_frames
            && rewards.shape[0].checked_add(1) == Some(frames)
            && offsets.dtype == DType::U64
            && offsets.shape.len() == 1
            && frames.checked_add(1) == offsets.shape.first().copied()
            && offsets.shape[0].checked_mul(8) == Some(offsets.bytes)
            && data.dtype == DType::U8
            && data.shape.len() == 1
            && data.shape[0] == data.bytes
            && data.bytes >= frames
            && frames
                .checked_mul(limits.max_scene_bytes)
                .is_some_and(|n| data.bytes <= n)
            && data
                .bytes
                .checked_add(offsets.bytes)
                .is_some_and(|n| n <= limits.max_trajectory_bytes),
        "invalid render descriptor or render budget exceeded",
    )
}

// A successful early EOF is a decoded-size mismatch. Preserve actual reader
// errors, notably flate2's UnexpectedEof("incomplete deflate stream").
fn read_decoded_exact(reader: &mut impl Read, mut out: &mut [u8]) -> Result<()> {
    while !out.is_empty() {
        match reader.read(out) {
            Ok(0) => ensure(false, "diagnostic decompression size mismatch")?,
            Ok(n) => out = &mut out[n..],
            Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
            Err(error) => return Err(error.into()),
        }
    }
    Ok(())
}

fn read_numeric<T, const N: usize>(
    reader: &mut impl Read,
    count: usize,
    convert: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>> {
    let mut values = Vec::with_capacity(count);
    let mut scratch = [0u8; 8192];
    while values.len() < count {
        let bytes = (count - values.len()).min(scratch.len() / N) * N;
        read_decoded_exact(reader, &mut scratch[..bytes])?;
        values.extend(
            scratch[..bytes]
                .chunks_exact(N)
                .map(|bytes| convert(bytes.try_into().expect("complete numeric value"))),
        );
    }
    Ok(values)
}

fn read_values(reader: &mut impl Read, dtype: DType, count: usize) -> Result<Values> {
    Ok(match dtype {
        DType::F32 => Values::F32(read_numeric(reader, count, f32::from_le_bytes)?),
        DType::F64 => Values::F64(read_numeric(reader, count, f64::from_le_bytes)?),
        DType::U64 => Values::U64(read_numeric(reader, count, u64::from_le_bytes)?),
        DType::U8 => {
            let mut values = vec![0; count];
            read_decoded_exact(reader, &mut values)?;
            Values::U8(values)
        }
    })
}

fn decode_admitted(
    bytes: &[u8],
    max_bytes: usize,
    max_header: usize,
    max_version: u32,
    charge_limit: Option<usize>,
) -> Result<Option<Chunk>> {
    ensure(
        bytes.len() >= 12 && &bytes[..8] == MAGIC,
        "invalid diagnostic chunk header",
    )?;
    let header_len = u32::from_le_bytes(bytes[8..12].try_into().expect("four bytes")) as usize;
    ensure(
        header_len <= max_header && header_len <= bytes.len() - 12,
        "invalid diagnostic header length",
    )?;
    let header: Header = serde_json::from_slice(&bytes[12..12 + header_len])?;
    ensure(
        (1..=max_version).contains(&header.schema_version),
        "unsupported diagnostic chunk version",
    )?;
    ensure(
        header.raw_bytes <= max_bytes,
        "diagnostic decoded chunk exceeds read limit",
    )?;
    if matches!(header.record, Record::Typed(_)) {
        ensure(
            header.schema_version >= 5 && header.raw_bytes <= super::typed::MAX_TYPED_BYTES,
            "unsupported typed chunk or typed payload exceeds its byte limit",
        )?;
    }
    validate_render_descriptors(&header)?;
    // Validate the partition before decompression or array allocation, including
    // admission-refused chunks. No descriptor can amplify the retained payload.
    let mut extent = 0usize;
    for descriptor in header.arrays.values() {
        let count = elements(&descriptor.shape)?;
        ensure(
            count.checked_mul(descriptor.dtype.bytes()) == Some(descriptor.bytes)
                && descriptor.offset == extent
                && descriptor.bytes <= header.raw_bytes.saturating_sub(extent),
            "invalid diagnostic array extent",
        )?;
        extent = extent
            .checked_add(descriptor.bytes)
            .ok_or_else(|| std::io::Error::other("diagnostic extent overflow"))?;
    }
    ensure(
        extent == header.raw_bytes,
        "diagnostic payload has unreferenced trailing bytes",
    )?;
    // This admission precedes Deflate and every numeric array allocation. The
    // encoded input and bounded conversion/Deflate scratch are separate from
    // the retained ownership charge.
    if charge_limit.is_some_and(|limit| {
        super::schema::retained_chunk_charge(
            &header.context,
            &header.record,
            header.raw_bytes,
            header.arrays.len(),
        ) > limit
    }) {
        return Ok(None);
    }
    let body = &bytes[12 + header_len..];
    let mut decoder = if header.compressed {
        Some(DeflateDecoder::new(body).take(header.raw_bytes as u64 + 1))
    } else {
        ensure(
            body.len() == header.raw_bytes,
            "diagnostic payload length mismatch",
        )?;
        None
    };
    let mut arrays = BTreeMap::new();
    // The entire immutable descriptor partition was checked before admission.
    // Conversion introduces no semantic checks before the completion probe.
    for (name, descriptor) in header.arrays {
        let count = elements(&descriptor.shape)?;
        let values = if let Some(reader) = &mut decoder {
            read_values(reader, descriptor.dtype, count)?
        } else {
            let data = &body[descriptor.offset..descriptor.offset + descriptor.bytes];
            match descriptor.dtype {
                DType::F32 => Values::F32(
                    data.chunks_exact(4)
                        .map(|b| f32::from_le_bytes(b.try_into().expect("four bytes")))
                        .collect(),
                ),
                DType::F64 => Values::F64(
                    data.chunks_exact(8)
                        .map(|b| f64::from_le_bytes(b.try_into().expect("eight bytes")))
                        .collect(),
                ),
                DType::U64 => Values::U64(
                    data.chunks_exact(8)
                        .map(|b| u64::from_le_bytes(b.try_into().expect("eight bytes")))
                        .collect(),
                ),
                DType::U8 => Values::U8(data.to_vec()),
            }
        };
        arrays.insert(name, Array::checked(descriptor.shape, values)?);
    }
    if let Some(reader) = &mut decoder {
        // Check size and the Deflate ending before any final chunk validation,
        // even for empty payloads. Stop after at most one extra decoded byte.
        let mut extra = [0u8; 1];
        loop {
            match reader.read(&mut extra) {
                Ok(n) => {
                    ensure(n == 0, "diagnostic decompression size mismatch")?;
                    break;
                }
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => continue,
                Err(error) => return Err(error.into()),
            }
        }
    }
    let chunk = Chunk {
        context: header.context,
        record: header.record,
        arrays,
    };
    chunk.validate()?;
    Ok(Some(chunk))
}

#[cfg(test)]
mod bounded_decode_tests {
    use super::*;

    const LIMIT: usize = 1024 * 1024;
    fn chunk() -> Chunk {
        Chunk {
            context: RecordContext::new("bounded-decode", 0),
            record: Record::Activations(ActivationRecord {
                network: "policy".into(),
                layers: BTreeMap::from([("hidden".into(), "f32".into())]),
                ..Default::default()
            }),
            arrays: BTreeMap::from([
                ("empty".into(), Array::f64(vec![2, 0], vec![]).unwrap()),
                (
                    "f32".into(),
                    Array::f32(
                        vec![1, 2051],
                        (0..2051)
                            .map(|i| [0., -0., f32::MIN_POSITIVE, -13.25][i % 4])
                            .collect(),
                    )
                    .unwrap(),
                ),
                (
                    "f64".into(),
                    Array::f64(
                        vec![1027],
                        (0..1027)
                            .map(|i| [0., -0., f64::MIN_POSITIVE, 7.125][i % 4])
                            .collect(),
                    )
                    .unwrap(),
                ),
                (
                    "u64".into(),
                    Array::u64(vec![1025], (0..1025).map(|i| u64::MAX - i).collect()).unwrap(),
                ),
                (
                    "u8".into(),
                    Array::u8(vec![8193], (0..8193).map(|i| (i % 251) as u8).collect()).unwrap(),
                ),
            ]),
        }
    }
    fn parts(bytes: &[u8]) -> (serde_json::Value, &[u8]) {
        let length = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        (
            serde_json::from_slice(&bytes[12..12 + length]).unwrap(),
            &bytes[12 + length..],
        )
    }
    fn encoded(header: &serde_json::Value, body: &[u8], compressed: bool) -> Vec<u8> {
        let mut header = header.clone();
        header["compressed"] = compressed.into();
        let json = serde_json::to_vec(&header).unwrap();
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&json);
        if compressed {
            let mut encoder = DeflateEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(body).unwrap();
            out.extend(encoder.finish().unwrap());
        } else {
            out.extend_from_slice(body);
        }
        out
    }
    fn error(bytes: &[u8]) -> String {
        decode(bytes, LIMIT, LIMIT)
            .expect_err("invalid fixture must be rejected")
            .to_string()
    }
    #[test]
    fn mixed_dtypes_cross_conversion_blocks_and_keep_all_bits_and_metadata() {
        let chunk = chunk();
        for compressed in [false, true] {
            let bytes = encode(&chunk, LIMIT, LIMIT, compressed).unwrap();
            assert_eq!(parts(&bytes).0["compressed"], compressed);
            let loaded = decode(&bytes, LIMIT, LIMIT).unwrap();
            assert_eq!(
                serde_json::to_value((&loaded.context, &loaded.record)).unwrap(),
                serde_json::to_value((&chunk.context, &chunk.record)).unwrap()
            );
            assert_eq!(loaded.arrays.len(), chunk.arrays.len());
            for (name, array) in &chunk.arrays {
                let actual = &loaded.arrays[name];
                assert_eq!(actual.dtype(), array.dtype());
                assert_eq!(actual.shape(), array.shape());
                match (&actual.values, &array.values) {
                    (Values::F32(a), Values::F32(b)) => assert_eq!(
                        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    ),
                    (Values::F64(a), Values::F64(b)) => assert_eq!(
                        a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
                    ),
                    (Values::U64(a), Values::U64(b)) => assert_eq!(a, b),
                    (Values::U8(a), Values::U8(b)) => assert_eq!(a, b),
                    _ => panic!("dtype changed"),
                }
            }
        }
    }
    #[test]
    fn zero_payload_and_empty_arrays_still_check_compressed_completion() {
        for empty_arrays in [false, true] {
            let mut chunk = chunk();
            chunk.record = Record::Metrics(MetricRecord {
                values: BTreeMap::new(),
                unavailable: BTreeMap::new(),
            });
            chunk
                .arrays
                .retain(|name, _| empty_arrays && name == "empty");
            let bytes = encode(&chunk, LIMIT, LIMIT, false).unwrap();
            let (header, raw) = parts(&bytes);
            assert!(raw.is_empty());
            for compressed in [false, true] {
                assert_eq!(
                    decode(&encoded(&header, raw, compressed), LIMIT, LIMIT)
                        .unwrap()
                        .arrays
                        .len(),
                    usize::from(empty_arrays)
                );
                assert!(
                    error(&encoded(&header, &[0], compressed)).contains(if compressed {
                        "decompression size mismatch"
                    } else {
                        "payload length mismatch"
                    })
                );
            }
            let mut incomplete = encoded(&header, raw, true);
            incomplete.pop();
            assert!(
                matches!(decode(&incomplete, LIMIT, LIMIT), Err(crate::Error::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof && e.to_string() == "incomplete deflate stream")
            );
        }
    }
    #[test]
    fn short_partial_values_expanded_and_incomplete_streams_keep_error_distinction() {
        let bytes = encode(&chunk(), LIMIT, LIMIT, false).unwrap();
        let (header, raw) = parts(&bytes);
        // Includes a partial f32 and a partial u64, in addition to a final u8.
        for length in [1, 8203, 16421, raw.len() - 1] {
            assert!(
                error(&encoded(&header, &raw[..length], true))
                    .contains("decompression size mismatch")
            );
        }
        let mut expanded = raw.to_vec();
        expanded.push(0);
        assert!(error(&encoded(&header, &expanded, true)).contains("decompression size mismatch"));
        assert!(error(&encoded(&header, &expanded, false)).contains("payload length mismatch"));
        let mut incomplete = encoded(&header, raw, true);
        incomplete.truncate(incomplete.len() - 8);
        assert!(
            matches!(decode(&incomplete, LIMIT, LIMIT), Err(crate::Error::Io(e)) if e.kind() == std::io::ErrorKind::UnexpectedEof && e.to_string() == "incomplete deflate stream")
        );
        let mut trailing = encoded(&header, raw, true);
        trailing.extend_from_slice(&[255, 255, 0, 42]);
        assert_eq!(
            decode(&trailing, LIMIT, LIMIT).unwrap().array_bytes(),
            raw.len()
        );
    }
    #[test]
    fn all_descriptors_are_preflighted_before_inflation_or_charge_refusal() {
        let bytes = encode(&chunk(), LIMIT, LIMIT, false).unwrap();
        let (header, _) = parts(&bytes);
        for (variant, expected) in [
            (0, "version"),
            (1, "extent"),
            (2, "rank"),
            (3, "overflow"),
            (4, "unreferenced"),
            (5, "extent"),
        ] {
            let mut bad = header.clone();
            match variant {
                0 => bad["schema_version"] = 999.into(),
                1 => bad["arrays"]["u8"]["offset"] = 1.into(),
                2 => bad["arrays"]["u8"]["shape"] = serde_json::json!([]),
                3 => bad["arrays"]["u8"]["shape"] = serde_json::json!([usize::MAX, 2]),
                4 => bad["raw_bytes"] = (header["raw_bytes"].as_u64().unwrap() + 1).into(),
                _ => bad["arrays"]["u64"]["bytes"] = 7.into(),
            }
            let mut invalid = encoded(&bad, &[], true);
            invalid.pop(); // incomplete Deflate must never mask descriptor errors
            let err = decode_if_within_charge(&invalid, LIMIT, LIMIT, 0)
                .unwrap_err()
                .to_string();
            assert!(err.contains(expected), "{variant}: {err}");
        }
        // Valid descriptors with a broken stream must be refused before inflation.
        let mut invalid = encoded(&header, &[], true);
        invalid.truncate(invalid.len() - 1);
        assert!(
            decode_if_within_charge(&invalid, LIMIT, LIMIT, 0)
                .unwrap()
                .is_none()
        );
        assert!(decode(&invalid, LIMIT, LIMIT).is_err());
    }
    #[test]
    fn payload_completion_precedes_final_schema_and_nonfinite_validation() {
        let bytes = encode(&chunk(), LIMIT, LIMIT, false).unwrap();
        let (header, raw) = parts(&bytes);
        for nonfinite in [false, true] {
            let mut header = header.clone();
            let mut raw = raw.to_vec();
            if nonfinite {
                header["record"] = serde_json::to_value(Record::Policy(PolicyRecord {
                    actions: "f32".into(),
                    parameters: PolicyParameters::Deterministic,
                    entropy: None,
                    entropy_method: None,
                    masks: None,
                    stochastic: false,
                }))
                .unwrap();
                raw[..4].copy_from_slice(&f32::NAN.to_le_bytes());
            } else {
                header["record"]["data"]["layers"]["hidden"] = "missing".into();
            }
            let valid_stream_error = error(&encoded(&header, &raw, true));
            assert!(
                valid_stream_error.contains(if nonfinite {
                    "action shape/dtype/values"
                } else {
                    "missing array"
                }),
                "{valid_stream_error}"
            );
            raw.push(0);
            assert!(error(&encoded(&header, &raw, true)).contains("decompression size mismatch"));
            raw.truncate(raw.len() - 2);
            assert!(error(&encoded(&header, &raw, true)).contains("decompression size mismatch"));
        }
    }
    #[test]
    fn conversion_retries_interrupted_partial_reads_and_keeps_actual_io_errors() {
        struct Partial<'a> {
            bytes: &'a [u8],
            interrupt: bool,
            error: bool,
        }
        impl Read for Partial<'_> {
            fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
                if std::mem::take(&mut self.interrupt) {
                    return Err(std::io::ErrorKind::Interrupted.into());
                }
                if self.bytes.is_empty() && self.error {
                    return Err(std::io::Error::new(
                        std::io::ErrorKind::UnexpectedEof,
                        "controlled decoder failure",
                    ));
                }
                let length = out.len().min(3);
                self.bytes.read(&mut out[..length])
            }
        }
        let raw = [u64::MAX.to_le_bytes(), 7u64.to_le_bytes()].concat();
        let mut reader = Partial {
            bytes: &raw,
            interrupt: true,
            error: false,
        };
        let Values::U64(values) = read_values(&mut reader, DType::U64, 2).unwrap() else {
            panic!()
        };
        assert_eq!(values, [u64::MAX, 7]);
        for error in [false, true] {
            let mut reader = Partial {
                bytes: &raw[..15],
                interrupt: true,
                error,
            };
            let actual = read_values(&mut reader, DType::U64, 2).unwrap_err();
            if error {
                assert!(
                    matches!(actual, crate::Error::Io(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && e.to_string() == "controlled decoder failure")
                );
            } else {
                assert!(actual.to_string().contains("decompression size mismatch"));
            }
        }
    }
}

#[cfg(test)]
mod lineage_version_tests {
    use super::*;
    #[test]
    fn bounded_admission_keeps_schema_and_array_extent_validation() {
        let chunk = Chunk {
            context: RecordContext::new("bounded", 0),
            record: Record::Metrics(MetricRecord {
                values: BTreeMap::new(),
                unavailable: BTreeMap::new(),
            }),
            arrays: BTreeMap::from([("a".into(), Array::u8(vec![2], vec![1, 2]).unwrap())]),
        };
        let bytes = encode(&chunk, 4096, 4096, false).unwrap();
        let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let header: serde_json::Value = serde_json::from_slice(&bytes[12..12 + len]).unwrap();
        for schema_error in [false, true] {
            let mut invalid = header.clone();
            if schema_error {
                invalid["schema_version"] = 999.into();
            } else {
                invalid["arrays"]["a"]["offset"] = 1.into();
            }
            let json = serde_json::to_vec(&invalid).unwrap();
            let mut encoded = MAGIC.to_vec();
            encoded.extend_from_slice(&(json.len() as u32).to_le_bytes());
            encoded.extend(json);
            encoded.extend_from_slice(&bytes[12 + len..]);
            let error = decode_if_within_charge(&encoded, 4096, 4096, 4096)
                .unwrap_err()
                .to_string();
            assert!(error.contains(if schema_error { "version" } else { "extent" }));
        }
        let loaded = decode_if_within_charge(&bytes, 4096, 4096, 4096)
            .unwrap()
            .unwrap();
        assert_eq!(loaded.retained_charge(), chunk.retained_charge());
        assert!(
            decode_if_within_charge(&bytes, 4096, 4096, chunk.retained_charge() - 1)
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn legacy_missing_lineage_decodes_but_schema_two_reader_rejects_new_semantics() {
        let chunk = Chunk {
            context: RecordContext::new("test", 0),
            record: Record::Activations(ActivationRecord {
                network: "policy".into(),
                layers: BTreeMap::from([("h".into(), "h".into())]),
                inputs: ActivationInputs::Observations,
                ..Default::default()
            }),
            arrays: BTreeMap::from([("h".into(), Array::f32(vec![2, 1], vec![1., 2.]).unwrap())]),
        };
        let bytes = encode(&chunk, 4096, 4096, false).unwrap();
        assert!(
            decode_versioned(&bytes, 4096, 4096, 2)
                .unwrap_err()
                .to_string()
                .contains("version")
        );
        assert!(matches!(
            decode(&bytes, 4096, 4096).unwrap().record,
            Record::Activations(ActivationRecord {
                inputs: ActivationInputs::Observations,
                ..
            })
        ));
        let length = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut header: serde_json::Value =
            serde_json::from_slice(&bytes[12..12 + length]).unwrap();
        header["record"]["data"]
            .as_object_mut()
            .unwrap()
            .remove("inputs");
        for version in [1, 2] {
            header["schema_version"] = version.into();
            let json = serde_json::to_vec(&header).unwrap();
            let mut old = MAGIC.to_vec();
            old.extend_from_slice(&(json.len() as u32).to_le_bytes());
            old.extend(json);
            old.extend_from_slice(&bytes[12 + length..]);
            assert!(matches!(
                decode(&old, 4096, 4096).unwrap().record,
                Record::Activations(ActivationRecord {
                    inputs: ActivationInputs::Unknown,
                    ..
                })
            ));
        }
    }
}

#[cfg(test)]
use crate as recording_rl;
#[cfg(test)]
#[path = "../../tests/support/recording_env.rs"]
mod recording_env;

#[cfg(test)]
mod render_codec_tests {
    use super::recording_env::{NAME, RecordingEnv};
    use super::*;
    use crate::{
        core::env::Env,
        diagnostics::{
            evaluation::{EvaluationConfig, evaluate},
            render::*,
        },
        train::trainer::RandomActor,
    };
    fn trajectory(rendered: bool) -> Chunk {
        let actor = RandomActor {
            space: RecordingEnv::new(2, 0).action_space().clone(),
        };
        let result = evaluate(
            |seed| Ok(RecordingEnv::new(2, seed)),
            &actor,
            &EvaluationConfig {
                seeds: vec![1],
                trajectories: 1,
                max_steps: 3,
                ..Default::default()
            },
        )
        .unwrap();
        let mut chunk = result
            .trajectory_record(0, RecordContext::new("render-codec", 0), "episode", NAME)
            .unwrap();
        if rendered {
            let scene = RenderScene {
                version: RENDER_VERSION,
                bounds: RenderBounds {
                    min: [0., 0.],
                    max: [1., 1.],
                },
                background: [0; 4],
                items: vec![],
            };
            let bytes = serde_json::to_vec(&scene).unwrap();
            chunk.arrays.insert(
                "render/data".into(),
                Array::u8(vec![bytes.len() * 3], bytes.repeat(3)).unwrap(),
            );
            chunk.arrays.insert(
                "render/offsets".into(),
                Array::u64(
                    vec![4],
                    vec![
                        0,
                        bytes.len() as u64,
                        2 * bytes.len() as u64,
                        3 * bytes.len() as u64,
                    ],
                )
                .unwrap(),
            );
            let Record::Trajectory(record) = &mut chunk.record else {
                panic!()
            };
            record.render = Some(RenderTrack {
                version: RENDER_VERSION,
                data: "render/data".into(),
                offsets: "render/offsets".into(),
            });
        }
        chunk
    }
    fn rewrite(
        bytes: &[u8],
        edit: impl FnOnce(&mut serde_json::Value),
        body: Option<&[u8]>,
    ) -> Vec<u8> {
        let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let mut header: serde_json::Value = serde_json::from_slice(&bytes[12..12 + len]).unwrap();
        edit(&mut header);
        let json = serde_json::to_vec(&header).unwrap();
        let mut out = MAGIC.to_vec();
        out.extend_from_slice(&(json.len() as u32).to_le_bytes());
        out.extend_from_slice(&json);
        out.extend_from_slice(body.unwrap_or(&bytes[12 + len..]));
        out
    }
    #[test]
    fn render_descriptors_rejected_before_deflate_or_admission() {
        let bytes = encode(&trajectory(true), 1024 * 1024, 65536, false).unwrap();
        for variant in 0..5 {
            let invalid = rewrite(
                &bytes,
                |h| {
                    h["compressed"] = true.into();
                    match variant {
                        0 => h["arrays"]["render/data"]["bytes"] = (9 * 1024 * 1024).into(),
                        1 => h["arrays"]["render/offsets"]["shape"] = serde_json::json!([5000]),
                        2 => h["record"]["data"]["render"]["version"] = 999.into(),
                        3 => h["arrays"]["render/data"]["dtype"] = "F64".into(),
                        _ => {
                            h["arrays"]["observations"]["shape"] =
                                serde_json::json!([usize::MAX, 1])
                        }
                    }
                },
                Some(&[255, 255]),
            );
            // The deliberately invalid compressed body proves rejection precedes inflation.
            let error = decode(&invalid, 64 * 1024 * 1024, 65536)
                .unwrap_err()
                .to_string();
            assert!(error.contains("render"), "{variant}: {error}");
            assert!(decode_if_within_charge(&invalid, 64 * 1024 * 1024, 65536, 0).is_err());
        }
    }
    #[test]
    fn malformed_scene_validation_follows_compressed_payload_completion() {
        let mut chunk = trajectory(true);
        let Values::U8(data) = &mut chunk.arrays.get_mut("render/data").unwrap().values else {
            panic!()
        };
        data[0] = b'!';
        let bytes = encode(&chunk, 1024 * 1024, 65536, false).unwrap();
        let len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let raw = &bytes[12 + len..];
        for variant in 0..4 {
            let mut payload = raw.to_vec();
            if variant == 1 {
                payload.push(0);
            }
            if variant == 2 {
                payload.pop();
            }
            let mut encoder = DeflateEncoder::new(Vec::new(), Compression::fast());
            encoder.write_all(&payload).unwrap();
            let mut body = encoder.finish().unwrap();
            if variant == 3 {
                body.truncate(body.len() - 8);
            }
            let compressed = rewrite(&bytes, |h| h["compressed"] = true.into(), Some(&body));
            let error = decode(&compressed, 1024 * 1024, 65536).unwrap_err();
            match variant {
                0 => assert!(!error.to_string().contains("decompression")),
                1 | 2 => assert!(error.to_string().contains("decompression size mismatch")),
                _ => assert!(
                    matches!(error, crate::Error::Io(e) if e.kind() == std::io::ErrorKind::UnexpectedEof && e.to_string() == "incomplete deflate stream")
                ),
            }
        }
    }
    #[test]
    fn near_budget_polygon_track_validation_cost() {
        let limits = RenderLimits::default();
        let scene = RenderScene {
            version: RENDER_VERSION,
            bounds: RenderBounds {
                min: [0., 0.],
                max: [2048., 4_200_000.],
            },
            background: [0; 4],
            items: vec![RenderItem {
                layer: 0,
                primitive: RenderPrimitive::Polygon {
                    points: (0..2048).map(|i| [i as f64, (i * i) as f64]).collect(),
                    fill: Some([255; 4]),
                    stroke: None,
                },
            }],
        };
        scene.validate(&limits).unwrap();
        let bytes = serde_json::to_vec(&scene).unwrap();
        assert!(bytes.len() <= limits.max_scene_bytes);
        let frames = ((limits.max_trajectory_bytes - 8) / (bytes.len() + 8)).min(limits.max_frames);
        let steps = frames - 1;
        let mut chunk = trajectory(false);
        let Record::Trajectory(record) = &mut chunk.record else {
            panic!()
        };
        chunk.arrays.insert(
            record.observations.clone(),
            Array::f32(vec![frames, 1], vec![0.; frames]).unwrap(),
        );
        chunk.arrays.insert(
            record.actions.clone(),
            Array::f32(vec![steps, 1], vec![0.; steps]).unwrap(),
        );
        chunk.arrays.insert(
            record.rewards.clone(),
            Array::f32(vec![steps], vec![0.; steps]).unwrap(),
        );
        let mut flags = vec![0; steps];
        flags[steps - 1] = 1;
        chunk.arrays.insert(
            record.terminated.clone(),
            Array::u8(vec![steps], flags).unwrap(),
        );
        chunk.arrays.insert(
            record.truncated.clone(),
            Array::u8(vec![steps], vec![0; steps]).unwrap(),
        );
        chunk.arrays.insert(
            "render/data".into(),
            Array::u8(vec![frames * bytes.len()], bytes.repeat(frames)).unwrap(),
        );
        chunk.arrays.insert(
            "render/offsets".into(),
            Array::u64(
                vec![frames + 1],
                (0..=frames).map(|i| (i * bytes.len()) as u64).collect(),
            )
            .unwrap(),
        );
        record.render = Some(RenderTrack {
            version: RENDER_VERSION,
            data: "render/data".into(),
            offsets: "render/offsets".into(),
        });
        let started = std::time::Instant::now();
        chunk.validate().unwrap();
        eprintln!(
            "render bounds audit: {} frames x 2048 polygon points, {} scene bytes, {} track bytes, validation {:?}",
            frames,
            bytes.len(),
            frames * bytes.len() + (frames + 1) * 8,
            started.elapsed()
        );
    }
    #[test]
    fn original_schema_three_traffic_trajectory_remains_readable() {
        // Original prior-writer bytes: see adjacent fixture provenance/SHA-256.
        // Do not regenerate the chunk or rewrite its header in this test.
        let bytes = include_bytes!("../../tests/fixtures/traffic_schema3_trajectory.rld");
        assert_eq!(bytes.len(), 4446);
        assert_eq!(&bytes[..8], MAGIC);
        let header_len = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let header: serde_json::Value =
            serde_json::from_slice(&bytes[12..12 + header_len]).unwrap();
        assert_eq!(header["schema_version"], 3);
        assert!(header["record"]["data"].get("render").is_none());

        let chunk = decode(bytes, 64 * 1024, 16 * 1024).unwrap();
        assert_eq!(chunk.context.identity.run_id, "traffic-ddqn-seed7");
        assert_eq!(chunk.context.identity.learner_update_revision, 0);
        assert_eq!(chunk.context.identity.model_role, "reference-never_switch");
        let Record::Trajectory(record) = &chunk.record else {
            panic!("expected original trajectory");
        };
        assert!(record.render.is_none());
        assert_eq!(record.id, "never_switch-k0-seed30000");
        assert_eq!(record.environment, "traffic");
        assert_eq!(record.evaluation.continuation_policy, "never_switch");
        assert_eq!(
            record.evaluation.conditioning,
            Conditioning::InitialScenario
        );
        assert_eq!(record.evaluation.discount, 0.99_f32 as f64);
        assert!(record.evaluation.complete_returns);
        assert!(!record.evaluation.stochastic_policy);

        assert_eq!(chunk.arrays[&record.observations].shape(), &[65, 5]);
        assert_eq!(chunk.arrays[&record.actions].shape(), &[64]);
        assert_eq!(chunk.arrays[&record.rewards].shape(), &[64]);
        assert!(
            chunk.arrays[&record.actions]
                .as_u64()
                .unwrap()
                .iter()
                .all(|&a| a == 0)
        );
        let masks = &chunk.arrays[record.action_masks.as_ref().unwrap()];
        assert_eq!(masks.shape(), &[65, 2]);
        let masks = masks.as_u8().unwrap();
        assert!(masks[..128].iter().all(|&v| v == 1));
        assert_eq!(&masks[128..], &[0, 0]);
        let terminated = chunk.arrays[&record.terminated].as_u8().unwrap();
        assert_eq!(terminated.len(), 64);
        assert!(terminated[..63].iter().all(|&v| v == 0));
        assert_eq!(terminated[63], 1);
        assert!(
            chunk.arrays[&record.truncated]
                .as_u8()
                .unwrap()
                .iter()
                .all(|&v| v == 0)
        );
        let times = chunk.arrays[record.simulation_times.as_ref().unwrap()]
            .as_f64()
            .unwrap();
        assert_eq!(times.len(), 65);
        assert_eq!((times[0], times[64]), (0., 64.));
        assert!(
            scene_at(&chunk, 64, &RenderLimits::default())
                .unwrap()
                .is_none()
        );
    }
    #[test]
    fn legacy_trajectory_versions_without_render_are_readable() {
        let bytes = encode(&trajectory(false), 1024 * 1024, 65536, false).unwrap();
        for version in 1..=3 {
            let old = rewrite(
                &bytes,
                |h| {
                    h["schema_version"] = version.into();
                    h["record"]["data"]
                        .as_object_mut()
                        .unwrap()
                        .remove("render");
                },
                None,
            );
            let decoded = decode(&old, 1024 * 1024, 65536).unwrap();
            assert!(matches!(
                decoded.record,
                Record::Trajectory(TrajectoryRecord { render: None, .. })
            ));
        }
        assert!(
            decode_versioned(
                &encode(&trajectory(true), 1024 * 1024, 65536, false).unwrap(),
                1024 * 1024,
                65536,
                3
            )
            .is_err()
        );
    }
}
