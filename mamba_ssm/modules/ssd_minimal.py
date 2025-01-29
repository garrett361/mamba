# Copyright (c) 2024, Albert Gu and Tri Dao.
"""Minimal implementation of SSD.

This is the same as Listing 1 from the paper.
"""

import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from torch.profiler import record_function


def segsum_unstable(x):
    """Naive segment sum calculation."""
    T = x.size(-1)
    x_cumsum = torch.cumsum(x, dim=-1)
    x_segsum = x_cumsum[..., :, None] - x_cumsum[..., None, :]
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def segsum(x):
    """More stable segment sum calculation."""
    T = x.size(-1)
    x = repeat(x, "... d -> ... d e", e=T)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=-1)
    x = x.masked_fill(~mask, 0)
    x_segsum = torch.cumsum(x, dim=-2)
    mask = torch.tril(torch.ones(T, T, device=x.device, dtype=bool), diagonal=0)
    x_segsum = x_segsum.masked_fill(~mask, -torch.inf)
    return x_segsum


def ssd_minimal_discrete(X, A, B, C, block_len, initial_states=None):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_groups, d_state)
        C: (batch, length, n_groups, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [
        rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_cumsum = torch.cumsum(A, dim=-1)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclgn,bcsgn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    with record_function("right"):
        decay_states = torch.exp((A_cumsum[:, :, :, -1:] - A_cumsum))
        states = torch.einsum("bclgn,bhcl,bclhp->bcghpn", B, decay_states, X)

    # 3. Compute the inter-chunk SSM recurrence; produces correct SSM states at chunk boundaries
    # (middle term of factorization of off-diag blocks; A terms)
    with record_function("center"):
        if initial_states is None:
            initial_states = torch.zeros_like(states[:, :1])
        states = torch.cat([initial_states, states], dim=1)
        decay_chunk = torch.exp(segsum(F.pad(A_cumsum[:, :, :, -1], (1, 0))))
        new_states = torch.einsum("bhzc,bcghpn->bzghpn", decay_chunk, states)
        states, final_state = new_states[:, :-1], new_states[:, -1]

    # 4. Compute state -> output conversion per chunk
    # (left term of low-rank factorization of off-diagonal blocks; C terms)
    with record_function("left"):
        state_decay_out = torch.exp(A_cumsum)
        Y_off = torch.einsum("bclgn,bcghpn,bhcl->bclhp", C, states, state_decay_out)

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y, final_state


def ssd_minimal_no_chunk_quadratic(X, A, B, C):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_groups, d_state)
        C: (batch, length, n_groups, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype

    A = rearrange(A, "b l h -> b h l")
    L = torch.exp(segsum(A))
    Y = torch.einsum("blgn,bsgn,bhls,bshp->blhp", C, B, L, X)

    return Y


def ssd_minimal_no_chunk_linear(X, A, B, C):
    """
    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_groups, d_state)
        C: (batch, length, n_groups, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    A_cs = A.cumsum(dim=1)
    out = torch.einsum("bsh,bsgn,bshp->bsghp", (-A_cs).exp(), B, X).cumsum(dim=1)
    out = torch.einsum("bsgp,bsh,bsghp->bshp", C, A_cs.exp(), out)
    return out


def ssd_minimal_discrete_alt(X, A, B, C, block_len):
    """
    An alternative pure pytorch implementation. Uses masks, rather than relying on cancellations of
    sums, for numeric stability than ssd_minimal_discrete_alt_naive.

    Arguments:
        X: (batch, length, n_heads, d_head)
        A: (batch, length, n_heads)
        B: (batch, length, n_groups, d_state)
        C: (batch, length, n_groups, d_state)
    Return:
        Y: (batch, length, n_heads, d_head)
    """
    assert X.dtype == A.dtype == B.dtype == C.dtype
    assert X.shape[1] % block_len == 0

    # Rearrange into blocks/chunks
    X, A, B, C = [
        rearrange(x, "b (c l) ... -> b c l ...", l=block_len) for x in (X, A, B, C)
    ]

    A = rearrange(A, "b c l h -> b h c l")
    A_sum = A.sum(dim=-1)  # (b, h, c)
    A_cs = A.cumsum(dim=-1)  # (b, h, c, l)

    # 1. Compute the output for each intra-chunk (diagonal blocks)
    L = torch.exp(segsum(A))
    Y_diag = torch.einsum("bclgn,bcsgn,bhcls,bcshp->bclhp", C, B, L, X)

    # 2. Compute the state for each intra-chunk
    # (right term of low-rank factorization of off-diagonal blocks; B terms)
    with record_function("right"):
        T = (A_sum[..., None] - A_cs).exp()
        right_factor = torch.einsum("bhcl,bclgn,bclhp->bcghpn", T, B, X)

    # 3. Center-factor. (A terms)
    with record_function("center"):
        A_sum_cs = A_sum.cumsum(dim=-1)
        center_right = (-A_sum_cs).exp()  # (b, h, c)
        center_right = (
            rearrange(center_right, "b h c -> b c h")[:, :, None, :, None, None]
            * right_factor
        )
        center_right = center_right.cumsum(dim=1) - center_right  # (b, c, g, h, p n)
        center_factor = (
            rearrange((A_sum_cs - A_sum).exp(), "b h c -> b c h")[
                :, :, None, :, None, None
            ]
            * center_right
        )  # (b, c, g, h, p, n)

    # 4. Left-factor (C terms)
    with record_function("left"):
        Y_off = torch.einsum("bclgn,bcghpn,bhcl->bclhp", C, center_factor, A_cs.exp())

    # Add output of intra-chunk and inter-chunk terms (diagonal and off-diagonal blocks)
    Y = rearrange(Y_diag + Y_off, "b c l h p -> b (c l) h p")
    return Y
