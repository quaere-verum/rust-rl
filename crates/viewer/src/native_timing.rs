//! Opt-in, capped native capture evidence. Never owns recording or model data.
use std::{
    fs::{File, OpenOptions},
    io::Write,
    path::Path,
    sync::{Arc, Mutex},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Clone, Default)]
pub(crate) struct NativeTiming(Option<Arc<Trace>>);
struct Trace {
    origin: Instant,
    sink: Mutex<Sink>,
}
struct Sink {
    file: File,
    cap: u64,
    written: u64,
    summaries: u64,
    summary_dropped: u64,
    dropped: u64,
    failed: bool,
}
impl NativeTiming {
    pub(crate) fn from_env() -> std::io::Result<Self> {
        std::env::var_os("RL_VIEWER_NATIVE_TIMING").map_or(Ok(Self::default()), |path| {
            Self::open(Path::new(&path), 100_000)
        })
    }
    pub(crate) fn open(path: &Path, cap: u64) -> std::io::Result<Self> {
        let origin = Instant::now();
        let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
        writeln!(
            file,
            "{}",
            serde_json::json!({"event":"origin", "us":0, "pid":std::process::id(), "unix_us":SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_micros(), "cap":cap})
        )?;
        Ok(Self(Some(Arc::new(Trace {
            origin,
            sink: Mutex::new(Sink {
                file,
                cap,
                written: 0,
                summaries: 0,
                summary_dropped: 0,
                dropped: 0,
                failed: false,
            }),
        }))))
    }
    pub(crate) fn enabled(&self) -> bool {
        self.0.is_some()
    }
    pub(crate) fn event(
        &self,
        name: &'static str,
        id: u64,
        details: impl FnOnce() -> serde_json::Value,
    ) {
        let Some(trace) = &self.0 else {
            return;
        };
        let mut sink = trace.sink.lock().unwrap_or_else(|e| e.into_inner());
        let summary = matches!(
            name,
            "screenshot_request" | "screenshot_callback" | "screenshot_written" | "close_request"
        );
        // Capture and close summaries have a separate bounded reserve that
        // cannot be consumed by per-frame/read detail.
        if (if summary {
            sink.summaries >= 4096
        } else {
            sink.written >= sink.cap
        }) || sink.failed
        {
            if summary {
                sink.summary_dropped += 1;
                return;
            }
            if sink.dropped == 0 && !sink.failed {
                let marker = serde_json::json!({"event":"overflow", "us":trace.origin.elapsed().as_micros(),"cap":sink.cap});
                if let Err(error) = writeln!(sink.file, "{marker}") {
                    sink.failed = true;
                    eprintln!("Native timing overflow marker failed: {error}");
                }
            }
            sink.dropped += 1;
            return;
        }
        let event = serde_json::json!({"event":name, "id":id, "us":trace.origin.elapsed().as_micros(), "details":details()});
        if let Err(error) = writeln!(sink.file, "{event}") {
            sink.failed = true;
            eprintln!("Native timing write failed: {error}");
        }
        if summary {
            sink.summaries += 1;
        } else {
            sink.written += 1;
        }
    }
    pub(crate) fn summary(&self) {
        let Some(trace) = &self.0 else {
            return;
        };
        let mut sink = trace.sink.lock().unwrap_or_else(|e| e.into_inner());
        let summary = serde_json::json!({"event":"summary", "us":trace.origin.elapsed().as_micros(), "written":sink.written, "dropped":sink.dropped, "lifecycle_written":sink.summaries,"lifecycle_dropped":sink.summary_dropped,"lifecycle_final_written":false,"io_failed":sink.failed});
        if let Err(error) = writeln!(sink.file, "{summary}") {
            eprintln!("Native timing summary failed: {error}");
        }
    }
    pub(crate) fn span(&self, name: &'static str, id: u64) -> Span {
        self.event(name, id, || serde_json::json!({"boundary":"begin"}));
        Span {
            timing: self.clone(),
            name,
            id,
        }
    }
}
pub(crate) struct Span {
    timing: NativeTiming,
    name: &'static str,
    id: u64,
}
impl Drop for Span {
    fn drop(&mut self) {
        self.timing
            .event(self.name, self.id, || serde_json::json!({"boundary":"end"}));
    }
}
impl Drop for Trace {
    fn drop(&mut self) {
        let sink = self.sink.get_mut().unwrap_or_else(|e| e.into_inner());
        let summary = serde_json::json!({"event":"summary", "us":self.origin.elapsed().as_micros(), "written":sink.written, "dropped":sink.dropped, "lifecycle_written":sink.summaries,"lifecycle_dropped":sink.summary_dropped,"lifecycle_final_written":false,"io_failed":sink.failed});
        if let Err(error) = writeln!(sink.file, "{summary}") {
            eprintln!("Native timing summary failed: {error}");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn capture_and_close_summaries_survive_exhausted_detail_budget() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("timing.jsonl");
        {
            let timing = NativeTiming::open(&path, 0).unwrap();
            timing.event("detail", 1, || serde_json::json!({}));
            for event in [
                "screenshot_request",
                "screenshot_callback",
                "screenshot_written",
                "close_request",
            ] {
                timing.event(event, 1, || serde_json::json!({"ok":true}));
            }
            timing.summary();
        }
        let values: Vec<serde_json::Value> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(values.len(), 8);
        assert_eq!(values[1]["event"], "overflow");
        assert_eq!(values[2]["event"], "screenshot_request");
        assert_eq!(values[3]["event"], "screenshot_callback");
        assert_eq!(values[4]["event"], "screenshot_written");
        assert_eq!(values[5]["event"], "close_request");
        for summary in &values[6..] {
            assert_eq!(summary["event"], "summary");
            assert_eq!(summary["written"], 0);
            assert_eq!(summary["dropped"], 1);
            assert_eq!(summary["lifecycle_written"], 4);
            assert_eq!(summary["lifecycle_dropped"], 0);
            assert_eq!(summary["lifecycle_final_written"], false);
        }
    }
    #[test]
    fn capture_and_close_summary_reserve_is_bounded_and_separate_from_details() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("timing.jsonl");
        {
            let timing = NativeTiming::open(&path, 1).unwrap();
            let events = [
                "screenshot_request",
                "screenshot_callback",
                "screenshot_written",
                "close_request",
            ];
            for id in 0..4096 {
                timing.event(events[id as usize % events.len()], id, || {
                    serde_json::json!({})
                });
            }
            for event in events {
                timing.event(event, 4096, || panic!("capped trace evaluated payload"));
            }
            timing.event("detail", 1, || serde_json::json!({"ok":true}));
            timing.event("detail", 2, || panic!("capped trace evaluated payload"));
        }
        let values: Vec<serde_json::Value> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(values.len(), 4100);
        assert_eq!(values[4097]["event"], "detail");
        assert_eq!(values[4098]["event"], "overflow");
        assert_eq!(values[4099]["written"], 1);
        assert_eq!(values[4099]["dropped"], 1);
        assert_eq!(values[4099]["lifecycle_written"], 4096);
        assert_eq!(values[4099]["lifecycle_dropped"], 4);
        assert_eq!(values[4099]["lifecycle_final_written"], false);
    }
    #[test]
    fn disabled_timing_does_not_evaluate_payload_or_touch_files() {
        NativeTiming::default().event("disabled", 1, || panic!("disabled trace allocated payload"));
    }
    #[test]
    fn bounded_stream_retains_order_and_reports_dropped_events() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("timing.jsonl");
        {
            let timing = NativeTiming::open(&path, 2).unwrap();
            for id in 1..=5 {
                timing.event("read", id, || serde_json::json!({"sequence": id + 10}));
            }
        }
        let lines: Vec<serde_json::Value> = std::fs::read_to_string(path)
            .unwrap()
            .lines()
            .map(|line| serde_json::from_str(line).unwrap())
            .collect();
        assert_eq!(lines.len(), 5); // origin, two events, overflow marker, final summary
        assert_eq!(lines[1]["id"], 1);
        assert_eq!(lines[2]["id"], 2);
        assert_eq!(lines[2]["details"]["sequence"], 12);
        assert_eq!(lines[3]["event"], "overflow");
        assert_eq!(lines[4]["dropped"], 3);
        assert!(lines[2]["us"].as_u64() >= lines[1]["us"].as_u64());
    }
}
