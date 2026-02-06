use burn::tensor::{backend::Backend, Device, Tensor};

#[derive(Clone)]
pub(crate) struct AutoregressiveCache<B: Backend> {
    /// Tensor cache with shape `[batch_size, num_heads, seq_len, d_model]`
    cache: Tensor<B, 4>,
    pub(crate) max_seq_len: usize,
    cur_seq_len: Vec<usize>,
}

impl<B: Backend> AutoregressiveCache<B> {
    /// Creates a new empty cache.
    pub fn new(
        max_batch_size: usize,
        num_heads: usize,
        max_seq_len: usize,
        d_model: usize,
        device: &Device<B>,
    ) -> Self {
        Self {
            cache: Tensor::empty([max_batch_size, num_heads, max_seq_len, d_model], device),
            max_seq_len,
            cur_seq_len: vec![0; max_batch_size],
        }
    }

    /// Reset the cache state.
    pub fn reset(&mut self) {
        self.cache = Tensor::empty(self.cache.shape(), &self.cache.device());
        self.cur_seq_len.iter_mut().for_each(|v| *v = 0);
    }

    /// Reset a single cache slot.
    pub fn reset_slot(&mut self, slot: usize) {
        if slot < self.cur_seq_len.len() {
            self.cur_seq_len[slot] = 0;
        }
    }

    /// Forward pass for a single cache slot (batch size = 1).
    pub fn forward_single_slot(&mut self, slot: usize, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let [_batch_size, num_heads, seq_len, d_model] = tensor.dims();
        let cur_seq_len = self.cur_seq_len[slot];
        let mut new_seq_len = cur_seq_len + seq_len;

        if new_seq_len > self.max_seq_len {
            let keep_len = self.max_seq_len - seq_len;
            let prev_slice = self.cache.clone().slice([
                slot..slot + 1,
                0..num_heads,
                seq_len..self.max_seq_len,
                0..d_model,
            ]);
            self.cache = self.cache.clone().slice_assign(
                [slot..slot + 1, 0..num_heads, 0..keep_len, 0..d_model],
                prev_slice,
            );
            self.cur_seq_len[slot] = keep_len;
            new_seq_len = self.max_seq_len;
        }

        let cur_seq_len = self.cur_seq_len[slot];
        self.cache = self.cache.clone().slice_assign(
            [
                slot..slot + 1,
                0..num_heads,
                cur_seq_len..new_seq_len,
                0..d_model,
            ],
            tensor,
        );

        self.cur_seq_len[slot] = new_seq_len;

        self.cache.clone().slice([
            slot..slot + 1,
            0..num_heads,
            0..self.cur_seq_len[slot],
            0..d_model,
        ])
    }

    /// Forward pass for a contiguous batch of slots starting at 0.
    /// Assumes all slots advance by the same seq_len (uniform lengths).
    pub fn forward(&mut self, tensor: Tensor<B, 4>) -> Tensor<B, 4> {
        let [batch_size, num_heads, seq_len, d_model] = tensor.dims();

        for slot in 0..batch_size {
            let slot_tensor = tensor
                .clone()
                .slice([slot..slot + 1, 0..num_heads, 0..seq_len, 0..d_model]);
            let _ = self.forward_single_slot(slot, slot_tensor);
        }

        let out_len = self.cur_seq_len[0];
        self.cache
            .clone()
            .slice([0..batch_size, 0..num_heads, 0..out_len, 0..d_model])
    }

    /// Returns the cached sequence length for slot 0.
    pub fn len(&self) -> usize {
        self.cur_seq_len[0]
    }

    /// Returns the cached sequence length for a specific slot.
    pub fn len_at(&self, slot: usize) -> usize {
        self.cur_seq_len[slot]
    }

    /// Returns cached sequence lengths for all slots.
    pub fn lens(&self) -> &[usize] {
        &self.cur_seq_len
    }
}
