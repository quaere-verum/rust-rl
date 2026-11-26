//! Bounded, non-blocking directory enumeration for the recording chooser.
use super::egui;
use std::{
    path::PathBuf,
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
        mpsc,
    },
};

pub(super) struct DirectoryListing {
    pub path: String,
    pub folders: Vec<PathBuf>,
    pub notice: Option<String>,
}

pub(super) struct DirectoryBrowser {
    requests: mpsc::SyncSender<(String, u64)>,
    results: mpsc::Receiver<(u64, DirectoryListing)>,
    submitted: Option<String>,
    desired: Option<String>,
    generation: Arc<AtomicU64>,
}

impl DirectoryBrowser {
    pub fn new(ctx: egui::Context) -> std::io::Result<Self> {
        let (requests, input) = mpsc::sync_channel::<(String, u64)>(1);
        let (output, results) = mpsc::sync_channel::<(u64, DirectoryListing)>(1);
        let generation = Arc::new(AtomicU64::new(0));
        let worker_generation = generation.clone();
        std::thread::Builder::new()
            .name("rl-directory-browser".into())
            .spawn(move || {
                while let Ok((path, request_generation)) = input.recv() {
                    let cancelled =
                        || worker_generation.load(Ordering::Acquire) != request_generation;
                    if let Some(listing) = scan(path, &cancelled)
                        && !cancelled()
                        && output.send((request_generation, listing)).is_err()
                    {
                        break;
                    }
                    ctx.request_repaint();
                }
            })?;
        Ok(Self {
            requests,
            results,
            submitted: None,
            desired: None,
            generation,
        })
    }

    pub fn cancel(&mut self) {
        if self.desired.take().is_some() {
            self.generation.fetch_add(1, Ordering::AcqRel);
            self.submitted = None;
        }
    }

    pub fn poll(&mut self, path: &str, ctx: &egui::Context) -> Option<DirectoryListing> {
        if self.desired.as_deref() != Some(path) {
            self.generation.fetch_add(1, Ordering::AcqRel);
            self.desired = Some(path.to_owned());
            self.submitted = None;
        }
        let mut latest = None;
        while let Ok((generation, listing)) = self.results.try_recv() {
            if generation == self.generation.load(Ordering::Acquire) && listing.path == path {
                latest = Some(listing);
            }
        }
        if self.submitted.as_deref() != Some(path) {
            match self
                .requests
                .try_send((path.to_owned(), self.generation.load(Ordering::Acquire)))
            {
                Ok(()) => self.submitted = Some(path.to_owned()),
                Err(mpsc::TrySendError::Full(_)) => {
                    ctx.request_repaint_after(std::time::Duration::from_millis(50))
                }
                Err(mpsc::TrySendError::Disconnected(_)) => {
                    return Some(DirectoryListing {
                        path: path.into(),
                        folders: vec![],
                        notice: Some("Directory browser unavailable".into()),
                    });
                }
            }
        }
        latest
    }
}

fn scan(path: String, cancelled: &impl Fn() -> bool) -> Option<DirectoryListing> {
    if cancelled() {
        return None;
    }
    let mut listing = DirectoryListing {
        path: path.clone(),
        folders: Vec::new(),
        notice: None,
    };
    match std::fs::read_dir(&path) {
        Ok(entries) => {
            for (index, entry) in entries.take(513).enumerate() {
                if cancelled() {
                    return None;
                }
                if index == 512 {
                    listing.notice = Some("Showing folders among the first 512 entries; enter a path to open another.".into());
                    break;
                }
                if let Ok(entry) = entry {
                    let path = entry.path();
                    if path.is_dir() {
                        listing.folders.push(path);
                    }
                }
            }
            listing.folders.sort();
        }
        Err(error) => listing.notice = Some(error.to_string()),
    }
    if cancelled() { None } else { Some(listing) }
}

#[cfg(test)]
#[test]
fn scan_bounds_entries_before_testing_for_directories() {
    let directory = tempfile::tempdir().unwrap();
    for index in 0..520 {
        std::fs::write(directory.path().join(format!("entry-{index}")), []).unwrap();
    }
    let listing = scan(directory.path().display().to_string(), &|| false).unwrap();
    assert!(listing.folders.is_empty());
    assert!(
        listing
            .notice
            .as_deref()
            .is_some_and(|notice| notice.contains("512"))
    );
}

#[cfg(test)]
#[test]
fn cancelled_scan_does_not_publish_a_listing() {
    let directory = tempfile::tempdir().unwrap();
    assert!(scan(directory.path().display().to_string(), &|| true).is_none());
}

#[cfg(test)]
#[test]
fn newer_directory_request_supersedes_old_results() {
    let first = tempfile::tempdir().unwrap();
    let second = tempfile::tempdir().unwrap();
    std::fs::create_dir(second.path().join("new-folder")).unwrap();
    let ctx = egui::Context::default();
    let mut browser = DirectoryBrowser::new(ctx.clone()).unwrap();
    browser.poll(&first.path().display().to_string(), &ctx);
    let target = second.path().display().to_string();
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
    loop {
        if let Some(listing) = browser.poll(&target, &ctx) {
            assert_eq!(listing.path, target);
            assert_eq!(listing.folders, [second.path().join("new-folder")]);
            break;
        }
        assert!(
            std::time::Instant::now() < deadline,
            "new directory did not load"
        );
        std::thread::sleep(std::time::Duration::from_millis(2));
    }
}
