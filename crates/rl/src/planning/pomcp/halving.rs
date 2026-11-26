use super::{Node, stable_max};

pub(super) struct SequentialHalving {
    candidates: Vec<usize>,
    remaining: usize,
    round_remaining: usize,
    next: usize,
}

impl SequentialHalving {
    pub(super) fn new(actions: usize, simulations: u32) -> Self {
        let mut halving = Self {
            candidates: (0..actions).collect(),
            remaining: simulations as usize,
            round_remaining: 0,
            next: 0,
        };
        halving.start_round();
        halving
    }

    fn start_round(&mut self) {
        let count = self.candidates.len();
        // Give the final comparison all remaining samples. Earlier rounds
        // allocate whole passes over the active set when the budget permits.
        self.round_remaining = if count <= 2 {
            self.remaining
        } else {
            let rounds = (usize::BITS - (count - 1).leading_zeros()) as usize;
            let passes = (self.remaining / rounds / count).max(1);
            (passes * count).min(self.remaining)
        };
        self.next = 0;
    }

    pub(super) fn select(&self) -> usize {
        self.candidates[self.next]
    }

    pub(super) fn complete<A, O>(&mut self, root: &Node<A, O>) {
        self.remaining -= 1;
        self.round_remaining -= 1;
        self.next = (self.next + 1) % self.candidates.len();
        if self.round_remaining == 0 && self.remaining > 0 {
            self.candidates.sort_unstable_by(|&left, &right| {
                root.edges[right]
                    .value
                    .partial_cmp(&root.edges[left].value)
                    .expect("completed search estimates are finite")
                    .then_with(|| left.cmp(&right))
            });
            self.candidates.truncate(self.candidates.len().div_ceil(2));
            self.candidates.sort_unstable();
            self.start_round();
        }
    }

    pub(super) fn best<A, O>(&self, root: &Node<A, O>) -> Option<usize> {
        stable_max(self.candidates.iter().map(|&index| {
            let edge = &root.edges[index];
            (edge.visits > 0).then_some(edge.value)
        }))
        .map(|index| self.candidates[index])
    }
}
