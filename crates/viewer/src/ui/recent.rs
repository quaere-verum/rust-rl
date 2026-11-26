//! Small, bounded list of successfully opened recording directories.
use std::{
    io::Read,
    path::{Path, PathBuf},
};

#[derive(Default)]
pub(super) struct RecentRecordings {
    paths: Vec<PathBuf>,
    storage: Option<PathBuf>,
}

impl RecentRecordings {
    pub fn load() -> Self {
        let directory = std::env::var_os("APPDATA")
            .or_else(|| std::env::var_os("XDG_CONFIG_HOME"))
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")));
        let storage = directory.map(|root| root.join("rust-rl").join("viewer-recent.json"));
        let paths = storage
            .as_ref()
            .and_then(|path| {
                let mut bytes = Vec::new();
                std::fs::File::open(path)
                    .ok()?
                    .take(64 * 1024 + 1)
                    .read_to_end(&mut bytes)
                    .ok()?;
                Some(bytes)
            })
            .filter(|bytes| bytes.len() <= 64 * 1024)
            .and_then(|bytes| serde_json::from_slice::<Vec<PathBuf>>(&bytes).ok())
            .unwrap_or_default()
            .into_iter()
            .take(8)
            .collect();
        Self { paths, storage }
    }

    pub fn paths(&self) -> &[PathBuf] {
        &self.paths
    }

    pub fn remember(&mut self, path: &Path) {
        self.paths.retain(|existing| existing != path);
        self.paths.insert(0, path.to_owned());
        self.paths.truncate(8);
        if let Some(storage) = &self.storage
            && let Some(parent) = storage.parent()
            && std::fs::create_dir_all(parent).is_ok()
            && let Ok(data) = serde_json::to_vec(&self.paths)
        {
            let _ = std::fs::write(storage, data);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn remembers_a_bounded_ordered_list_and_persists_it() {
        let directory = tempfile::tempdir().unwrap();
        let file = directory.path().join("settings").join("recent.json");
        let mut recent = RecentRecordings {
            paths: Vec::new(),
            storage: Some(file.clone()),
        };
        for index in 0..10 {
            recent.remember(Path::new(&format!("run-{index}")));
        }
        recent.remember(Path::new("run-5"));
        assert_eq!(recent.paths().len(), 8);
        assert_eq!(recent.paths()[0], Path::new("run-5"));
        assert_eq!(
            recent
                .paths()
                .iter()
                .filter(|path| *path == Path::new("run-5"))
                .count(),
            1
        );
        let saved: Vec<PathBuf> = serde_json::from_slice(&std::fs::read(file).unwrap()).unwrap();
        assert_eq!(saved, recent.paths());
    }
}
