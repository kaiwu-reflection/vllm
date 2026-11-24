# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""BF16x3 router GEMM for SM120 (RTX PRO 6000 / RTX 50xx).

The SM90/SM100 BF16x3 kernels use wgmma/tcgen05, which SM120 lacks; this
Triton kernel runs the same split on `mma.sync` BF16 tensor cores. Each FP32
weight is split into hi + mid + lo BF16 parts in registers, so BF16
activations produce FP32-accurate logits without an FP32 SIMT GEMM. Split-K
partials are reduced in a fixed order to keep routing deterministic.
"""

import torch

from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op


@triton.jit
def _bf16x3_partial_kernel(
    X,
    W,
    P,
    M,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    m_mask = rm[:, None] < M
    acc = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
    k_per_split: tl.constexpr = K // SPLIT_K
    for k0 in range(pid_k * k_per_split, (pid_k + 1) * k_per_split, BLOCK_K):
        rk = k0 + tl.arange(0, BLOCK_K)
        x = tl.load(X + rm[:, None] * K + rk[None, :], m_mask, other=0.0)
        w = tl.load(W + rn[None, :] * K + rk[:, None])
        hi = w.to(tl.bfloat16)
        rem = w - hi.to(tl.float32)
        mid = rem.to(tl.bfloat16)
        lo = (rem - mid.to(tl.float32)).to(tl.bfloat16)
        # Tensor-core accumulation drops low bits over long K, so each chunk
        # starts from zero and is added on the FP32 CUDA cores.
        chunk = tl.zeros((BLOCK_M, BLOCK_N), tl.float32)
        chunk = tl.dot(x, lo, chunk)
        chunk = tl.dot(x, mid, chunk)
        acc += tl.dot(x, hi, chunk)
    tl.store(P + (pid_k * M + rm[:, None]) * N + rn[None, :], acc, m_mask)


@triton.jit
def _splitk_reduce_kernel(
    Partials, Out, MN, SPLIT_K: tl.constexpr, BLOCK: tl.constexpr
):
    offs = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = offs < MN
    acc = tl.zeros((BLOCK,), tl.float32)
    for s in tl.static_range(SPLIT_K):
        acc += tl.load(Partials + s * MN + offs, mask, other=0.0)
    tl.store(Out + offs, acc, mask)


_BLOCK_K = 64


def _config(m: int, n: int, k: int) -> tuple[int, int, int, int]:
    """(BLOCK_M, BLOCK_N, SPLIT_K, num_stages), tuned on RTX PRO 6000."""
    if m <= 32:
        block_m, block_n, split_k = 16, 32, 16
    elif m <= 128:
        block_m, block_n, split_k = 32, 64, 16
    elif m <= 256:
        block_m, block_n, split_k = 64, 32, 16
    elif m <= 1024:
        block_m, block_n, split_k = 64, 32, 8
    else:
        # Enough tiles without split-K, and no partials buffer at prefill sizes.
        block_m, block_n, split_k = 64, 32, 1
    if n % block_n:
        block_n = 32
    while k % (_BLOCK_K * split_k):
        split_k //= 2
    return block_m, block_n, split_k, 3


def shape_supported(k: int, n: int) -> bool:
    return n % 32 == 0 and k % _BLOCK_K == 0


def _bf16x3_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    m, k = x.shape
    n = weight.shape[0]
    assert x.dtype == torch.bfloat16 and weight.dtype == torch.float32
    assert x.is_contiguous() and weight.is_contiguous()
    assert weight.shape[1] == k and shape_supported(k, n)
    out = torch.empty((m, n), device=x.device, dtype=torch.float32)
    if m == 0:
        return out
    block_m, block_n, split_k, stages = _config(m, n, k)
    partials = (
        out
        if split_k == 1
        else torch.empty((split_k, m, n), device=x.device, dtype=torch.float32)
    )
    _bf16x3_partial_kernel[(triton.cdiv(m, block_m), n // block_n, split_k)](
        x,
        weight,
        partials,
        m,
        n,
        k,
        block_m,
        block_n,
        _BLOCK_K,
        split_k,
        num_warps=4,
        num_stages=stages,
    )
    if split_k == 1:
        return out
    block = 1024
    _splitk_reduce_kernel[(triton.cdiv(m * n, block),)](
        partials, out, m * n, split_k, block, num_warps=4
    )
    return out


def warmup(k: int, n: int, max_num_tokens: int, device: torch.device) -> list[int]:
    """Compile every config bucket reachable within `max_num_tokens`."""
    weight = torch.zeros((n, k), device=device, dtype=torch.float32)
    sizes = sorted({min(m, max_num_tokens) for m in (32, 128, 256, 1024, 1025)})
    for m in sizes:
        _bf16x3_router_gemm(
            torch.zeros((m, k), device=device, dtype=torch.bfloat16), weight
        )
    return sizes


def _bf16x3_router_gemm_fake(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty((x.shape[0], weight.shape[0]), dtype=torch.float32)


direct_register_custom_op(
    op_name="bf16x3_router_gemm_sm120",
    op_func=_bf16x3_router_gemm,
    fake_impl=_bf16x3_router_gemm_fake,
)


def bf16x3_router_gemm(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.ops.vllm.bf16x3_router_gemm_sm120(x, weight)
