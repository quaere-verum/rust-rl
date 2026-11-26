//! Bounded metadata-only desired raw reads, collected for one UI pass.
use crate::data::CacheKey;
use rl::{Result, diagnostics::storage::ChunkSource};
use std::collections::BTreeMap;

pub const MAX_DESIRED_READS: usize = 32;

pub(crate) struct DesiredRead {
    pub key: CacheKey,
    pub source: ChunkSource,
    pub max_charge: Option<usize>,
    order: u64,
}

#[derive(Default)]
pub(crate) struct DesiredWork {
    items: Vec<DesiredRead>,
    previous_order: BTreeMap<CacheKey, u64>,
    next_order: u64,
    pub generation: u64,
}
impl DesiredWork {
    pub fn begin_pass(&mut self) {
        self.previous_order.clear();
        self.previous_order
            .extend(self.items.iter().map(|item| (item.key.clone(), item.order)));
        self.items.clear();
        self.generation = self.generation.wrapping_add(1);
    }
    pub fn clear(&mut self) {
        self.items.clear();
        self.previous_order.clear();
        self.generation = self.generation.wrapping_add(1);
    }
    pub fn len(&self) -> usize {
        self.items.len()
    }
    pub fn offer(
        &mut self,
        key: CacheKey,
        source: impl FnOnce() -> Result<ChunkSource>,
    ) -> Result<()> {
        self.offer_admitted(key, source, None)
    }
    pub fn offer_admitted(
        &mut self,
        key: CacheKey,
        source: impl FnOnce() -> Result<ChunkSource>,
        max_charge: Option<usize>,
    ) -> Result<()> {
        if let Some(existing) = self.items.iter_mut().find(|item| item.key == key) {
            existing.max_charge = match (existing.max_charge, max_charge) {
                (Some(a), Some(b)) => Some(a.min(b)),
                (a, b) => a.or(b),
            };
            return Ok(());
        }
        let order = self.previous_order.get(&key).copied().unwrap_or_else(|| {
            let order = self.next_order;
            self.next_order = self.next_order.wrapping_add(1);
            order
        });
        let replace = if self.items.len() < MAX_DESIRED_READS {
            None
        } else if let Some((position, worst)) = self
            .items
            .iter()
            .enumerate()
            .max_by_key(|(_, item)| item.order)
            && order < worst.order
        {
            Some(position)
        } else {
            return Ok(());
        };
        // Build manifest/address metadata only for admitted unique work.
        let incoming = DesiredRead {
            key,
            source: source()?,
            max_charge,
            order,
        };
        if let Some(position) = replace {
            self.items[position] = incoming;
        } else {
            self.items.push(incoming);
        }
        Ok(())
    }
    pub fn pop_next(&mut self) -> Option<DesiredRead> {
        let position = self
            .items
            .iter()
            .enumerate()
            .min_by_key(|(_, item)| item.order)?
            .0;
        Some(self.items.remove(position))
    }
}
