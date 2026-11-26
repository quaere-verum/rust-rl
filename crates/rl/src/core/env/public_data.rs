use serde::{Deserialize, Serialize, de::DeserializeOwned};
use sha2::{Digest, Sha256};
use std::{
    any::Any,
    borrow::Cow,
    fmt,
    sync::{Arc, OnceLock},
};

/// Optional exact public facts for a domain policy alongside its dense tensor.
/// The learner sees the numerical projection. Live reads borrow the original
/// type. Checkpoints store the data as a generic JSON value and the adapter
/// validates its version and contents when importing the corresponding type.
#[derive(Clone)]
pub struct PublicData {
    inner: Arc<dyn Erased>,
    identity: Arc<OnceLock<[u8; 32]>>,
    retained_bytes: usize,
}

trait Erased: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn eq_value(&self, other: &dyn Erased) -> Option<bool>;
    fn serialize(&self) -> serde_json::Result<serde_json::Value>;
}

impl<T: PartialEq + Serialize + Send + Sync + 'static> Erased for T {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn eq_value(&self, other: &dyn Erased) -> Option<bool> {
        other
            .as_any()
            .downcast_ref::<T>()
            .map(|other| self == other)
    }

    fn serialize(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::to_value(self)
    }
}

impl PublicData {
    pub fn new<T: PartialEq + Serialize + Send + Sync + 'static>(
        value: T,
        retained_bytes: usize,
    ) -> Self {
        Self {
            inner: Arc::new(value),
            identity: Arc::default(),
            retained_bytes,
        }
    }

    pub fn read<T: Clone + DeserializeOwned + 'static>(&self) -> crate::Result<Cow<'_, T>> {
        if let Some(value) = self.inner.as_any().downcast_ref::<T>() {
            Ok(Cow::Borrowed(value))
        } else {
            let value = Erased::serialize(self.inner.as_ref())?;
            Ok(Cow::Owned(serde_json::from_value(value)?))
        }
    }

    pub(crate) fn retained_bytes(&self) -> usize {
        self.retained_bytes
    }

    pub(crate) fn identity(&self) -> [u8; 32] {
        *self.identity.get_or_init(|| {
            let mut hash = Sha256::new();
            match Erased::serialize(self.inner.as_ref())
                .and_then(|value| serde_json::to_vec(&value))
            {
                Ok(encoded) => {
                    hash.update([1]);
                    hash.update(encoded);
                }
                Err(_) => hash.update([0]),
            }
            hash.finalize().into()
        })
    }
}

impl fmt::Debug for PublicData {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("PublicData")
            .field("retained_bytes", &self.retained_bytes)
            .finish_non_exhaustive()
    }
}

impl PartialEq for PublicData {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.inner, &other.inner)
            || self
                .inner
                .eq_value(other.inner.as_ref())
                .unwrap_or_else(|| {
                    match (
                        Erased::serialize(self.inner.as_ref()),
                        Erased::serialize(other.inner.as_ref()),
                    ) {
                        (Ok(left), Ok(right)) => left == right,
                        _ => false,
                    }
                })
    }
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct Wire {
    value: serde_json::Value,
    retained_bytes: usize,
}

impl Serialize for PublicData {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        Wire {
            value: Erased::serialize(self.inner.as_ref()).map_err(serde::ser::Error::custom)?,
            retained_bytes: self.retained_bytes,
        }
        .serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for PublicData {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = Wire::deserialize(deserializer)?;
        Ok(Self::new(wire.value, wire.retained_bytes))
    }
}
