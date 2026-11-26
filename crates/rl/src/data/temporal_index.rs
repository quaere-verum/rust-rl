//! Private weighted episode order. AVL balancing bounds lookup independently of IDs.
use super::EpisodeId;

#[derive(Debug, Clone, Default)]
pub(super) struct EpisodeIndex {
    root: Option<Box<Node>>,
}

#[derive(Debug, Clone)]
struct Node {
    id: EpisodeId,
    weight: usize,
    total: usize,
    height: usize,
    left: Option<Box<Node>>,
    right: Option<Box<Node>>,
}

impl Node {
    fn total(node: &Option<Box<Self>>) -> usize {
        node.as_ref().map_or(0, |n| n.total)
    }
    fn height(node: &Option<Box<Self>>) -> usize {
        node.as_ref().map_or(0, |n| n.height)
    }
    fn refresh(&mut self) {
        // The store checks its new total before changing any indexed weight.
        self.total = Self::total(&self.left) + self.weight + Self::total(&self.right);
        self.height = 1 + Self::height(&self.left).max(Self::height(&self.right));
    }
    fn rotate_left(mut self: Box<Self>) -> Box<Self> {
        let mut next = self.right.take().expect("right-heavy AVL node");
        self.right = next.left.take();
        self.refresh();
        next.left = Some(self);
        next.refresh();
        next
    }
    fn rotate_right(mut self: Box<Self>) -> Box<Self> {
        let mut next = self.left.take().expect("left-heavy AVL node");
        self.left = next.right.take();
        self.refresh();
        next.right = Some(self);
        next.refresh();
        next
    }
    fn balance(mut self: Box<Self>) -> Box<Self> {
        self.refresh();
        if Self::height(&self.left) > Self::height(&self.right) + 1 {
            let left = self.left.as_ref().expect("left-heavy AVL node");
            if Self::height(&left.right) > Self::height(&left.left) {
                self.left = self.left.take().map(Self::rotate_left);
            }
            self.rotate_right()
        } else if Self::height(&self.right) > Self::height(&self.left) + 1 {
            let right = self.right.as_ref().expect("right-heavy AVL node");
            if Self::height(&right.left) > Self::height(&right.right) {
                self.right = self.right.take().map(Self::rotate_right);
            }
            self.rotate_left()
        } else {
            self
        }
    }
    fn set(node: Option<Box<Self>>, id: EpisodeId, weight: usize) -> Box<Self> {
        let Some(mut node) = node else {
            return Box::new(Self {
                id,
                weight,
                total: weight,
                height: 1,
                left: None,
                right: None,
            });
        };
        match id.cmp(&node.id) {
            std::cmp::Ordering::Less => node.left = Some(Self::set(node.left.take(), id, weight)),
            std::cmp::Ordering::Greater => {
                node.right = Some(Self::set(node.right.take(), id, weight))
            }
            std::cmp::Ordering::Equal => node.weight = weight,
        }
        node.balance()
    }
    fn take_min(mut node: Box<Self>) -> (Box<Self>, Option<Box<Self>>) {
        let Some(left) = node.left.take() else {
            let right = node.right.take();
            return (node, right);
        };
        let (first, left) = Self::take_min(left);
        node.left = left;
        (first, Some(node.balance()))
    }
    fn remove(node: Option<Box<Self>>, id: EpisodeId) -> Option<Box<Self>> {
        let mut node = node?;
        match id.cmp(&node.id) {
            std::cmp::Ordering::Less => node.left = Self::remove(node.left.take(), id),
            std::cmp::Ordering::Greater => node.right = Self::remove(node.right.take(), id),
            std::cmp::Ordering::Equal => {
                return match (node.left.take(), node.right.take()) {
                    (left, None) => left,
                    (None, right) => right,
                    (left, Some(right)) => {
                        let (mut next, right) = Self::take_min(right);
                        next.left = left;
                        next.right = right;
                        Some(next.balance())
                    }
                };
            }
        }
        Some(node.balance())
    }
    fn audit(&self, low: Option<EpisodeId>, high: Option<EpisodeId>) -> Option<(usize, usize)> {
        if low.is_some_and(|id| id >= self.id) || high.is_some_and(|id| id <= self.id) {
            return None;
        }
        let (left_count, left_total) = if let Some(left) = &self.left {
            left.audit(low, Some(self.id))?
        } else {
            (0, 0)
        };
        let (right_count, right_total) = if let Some(right) = &self.right {
            right.audit(Some(self.id), high)?
        } else {
            (0, 0)
        };
        let left_height = Self::height(&self.left);
        let right_height = Self::height(&self.right);
        let total = left_total
            .checked_add(self.weight)?
            .checked_add(right_total)?;
        (self.total == total
            && self.height == 1 + left_height.max(right_height)
            && left_height.abs_diff(right_height) <= 1)
            .then_some((left_count.checked_add(right_count)?.checked_add(1)?, total))
    }
}

impl EpisodeIndex {
    pub fn total(&self) -> usize {
        Node::total(&self.root)
    }
    pub fn set(&mut self, id: EpisodeId, weight: usize) {
        self.root = Some(Node::set(self.root.take(), id, weight));
    }
    pub fn remove(&mut self, id: EpisodeId) {
        self.root = Node::remove(self.root.take(), id);
    }
    pub fn weight(&self, id: EpisodeId) -> Option<usize> {
        let mut next = self.root.as_ref();
        while let Some(node) = next {
            match id.cmp(&node.id) {
                std::cmp::Ordering::Less => next = node.left.as_ref(),
                std::cmp::Ordering::Greater => next = node.right.as_ref(),
                std::cmp::Ordering::Equal => return Some(node.weight),
            }
        }
        None
    }
    pub fn before(&self, id: EpisodeId) -> usize {
        let mut total = 0;
        let mut next = self.root.as_ref();
        while let Some(node) = next {
            if id <= node.id {
                next = node.left.as_ref();
            } else {
                total += Node::total(&node.left) + node.weight;
                next = node.right.as_ref();
            }
        }
        total
    }
    pub fn select(&self, mut rank: usize) -> Option<(EpisodeId, usize)> {
        let mut next = self.root.as_ref();
        while let Some(node) = next {
            let preceding = Node::total(&node.left);
            if rank < preceding {
                next = node.left.as_ref();
            } else if rank - preceding < node.weight {
                return Some((node.id, rank - preceding));
            } else {
                rank -= preceding + node.weight;
                next = node.right.as_ref();
            }
        }
        None
    }
    pub fn validate(&self, count: usize, total: usize) -> bool {
        self.root
            .as_ref()
            .map_or(Some((0, 0)), |n| n.audit(None, None))
            == Some((count, total))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weighted_order_remains_balanced_through_monotone_growth_updates_and_removals() {
        let mut index = EpisodeIndex::default();
        for id in 0..20_000 {
            index.set(id, id as usize % 4);
        }
        assert!(index.validate(20_000, 30_000));
        assert!(index.root.as_ref().unwrap().height <= 22);
        let mut rank = 0;
        for id in 0..20_000 {
            assert_eq!(index.before(id), rank);
            for local in 0..id as usize % 4 {
                assert_eq!(index.select(rank), Some((id, local)));
                rank += 1;
            }
        }
        assert_eq!(index.select(rank), None);
        for id in (0..20_000).step_by(2) {
            index.remove(id);
        }
        for id in (1..20_000).step_by(2) {
            index.set(id, 2);
        }
        assert!(index.validate(10_000, 20_000));
        assert!(index.root.as_ref().unwrap().height <= 21);
        for rank in 0..20_000 {
            assert_eq!(
                index.select(rank),
                Some((2 * (rank / 2) as u64 + 1, rank % 2))
            );
        }
        for id in (1..20_000).rev().step_by(2) {
            index.remove(id);
        }
        assert!(index.validate(0, 0));
    }

    #[test]
    fn comprehensive_audit_detects_private_weight_height_and_order_corruption() {
        let mut index = EpisodeIndex::default();
        for id in 1..=7 {
            index.set(id, 1);
        }
        for corruption in 0..3 {
            let mut bad = index.clone();
            let root = bad.root.as_mut().unwrap();
            match corruption {
                0 => root.total += 1,
                1 => root.height += 1,
                _ => root.left.as_mut().unwrap().id = root.id,
            }
            assert!(!bad.validate(7, 7));
        }
    }
}
