import torch
import triton
import triton.language as tl


def pad_sorted_idxs(
    counts: torch.LongTensor,
    n_toks: int,
    alignment: int,
    block_size: int = 128,
    max_blocks: int = 1024,
) -> tuple[torch.IntTensor, torch.IntTensor]:
    """
    Given the counts of tokens-per-expert, create the 1D index into a padded-out version of the
    tensor where each expert's index range fits into a block whose length is a multiple of
    `alignment`. Also returns the offsets into this index tensor.

    Example: if `x` is unpadded, then we fill `x_padded` by `x_padded[idxs] = x`.

    `idxs.shape = x.shape[:1]`

    This removes various cudaStreamSynchronizes compared to a pure torch impl using
    `repeat_interleave`. NOTE: @goon - though it doesn't end up having much e2e perf impact.

    """
    counts_aligned = ((counts + alignment - 1) // alignment) * alignment
    offs = counts_aligned.cumsum(dim=0, dtype=torch.int32)
    counts_cumsum = counts.cumsum(dim=0, dtype=torch.int32)
    idxs = torch.arange(n_toks, dtype=torch.int32, device=counts.device)

    # No action is needed for the zeroth expert
    num_blocks = min(counts.numel() - 1, max_blocks)
    # grid = one block per expert unless capped and then we loop...
    grid = (num_blocks,)
    _pad_sorted_idxs_kernel[grid](
        counts,
        counts_cumsum,
        offs,
        idxs,
        counts.numel(),
        BLOCK_SIZE=block_size,
    )
    return idxs, offs


@triton.jit
def _pad_sorted_idxs_kernel(
    counts_ptr,
    counts_cumsum_ptr,
    offs_ptr,
    idxs_ptr,
    n_local_experts: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,  # Number of threads per block
):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)

    # No action needed for the first expert's idxs.
    for exp_idx in range(pid + 1, n_local_experts, num_programs):
        # Load the idx range and derive the offset we need to add.

        count = tl.load(counts_ptr + exp_idx)
        # Need the cumsums up to, but not including, the current exp_idx
        count_cs = tl.load(counts_cumsum_ptr + exp_idx - 1)
        offs = tl.load(offs_ptr + exp_idx - 1)

        offsets = tl.arange(0, BLOCK_SIZE)
        for chunk_start in range(0, count, BLOCK_SIZE):
            chunk_offsets = chunk_start + offsets
            values = chunk_offsets + offs
            dest_indices = chunk_offsets + count_cs
            mask = chunk_offsets < count
            tl.store(idxs_ptr + dest_indices, values, mask=mask)
