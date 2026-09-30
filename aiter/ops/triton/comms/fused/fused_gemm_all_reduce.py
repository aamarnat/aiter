"""
Fused GEMM + All-Reduce operations using Iris RDMA communication.

This module provides fused GEMM and all-reduce operations optimized for
K-dimension sharding using Iris two-shot all-reduce with versioned locks.

Registers torch.ops.aiter.fused_gemm_all_reduce_k_shard custom op.
"""

import torch
import os

os.environ["TRITON_ALLOW_NON_CONSTEXPR_GLOBALS"] = "1"

import iris

from iris.ops.matmul_all_reduce import matmul_all_reduce
from iris.ops.config import FusedConfig
from iris.ops.workspace import FusedWorkspace

print("✓ persistent_gemm_all_reduce kernel available")

IRIS_INSTANCE = None
_CACHED_NUM_SMS = None  # cached SM count to avoid host query during CUDAGraph capture

# Cached across calls to avoid per-call allocation.
# Keyed by (M, N, dtype) to support multiple shapes without re-allocation
# during CUDAGraph capture (different layers have different N dimensions).
_BUFFER_CACHE = {}  # (M, N, dtype) -> (C_global, C_local, FusedWorkspace)


def init_iris(heap_size: int = 1 << 30):
    """Pre-initialize the iris context for fused GEMM+all_reduce.

    This MUST be called collectively on ALL ranks before any fused kernel
    execution (e.g., before CUDAGraph capture). It allocates the iris
    symmetric heap which requires coordinated IPC handle exchange.

    Args:
        heap_size: Symmetric heap size in bytes. Default 1GB.
    """
    global IRIS_INSTANCE
    if IRIS_INSTANCE is not None:
        return IRIS_INSTANCE
    IRIS_INSTANCE = iris.iris(heap_size=heap_size)
    print(
        f"✓ Iris context initialized for fused GEMM+AR: "
        f"rank {IRIS_INSTANCE.cur_rank}/{IRIS_INSTANCE.num_ranks}, "
        f"heap={heap_size / (1 << 20):.0f}MB"
    )
    return IRIS_INSTANCE


def _get_iris():
    global IRIS_INSTANCE
    if IRIS_INSTANCE is None:
        raise RuntimeError(
            "Iris not initialized. Call init_iris() collectively on all "
            "ranks before fused kernel execution (e.g., during pass setup)."
        )
    return IRIS_INSTANCE


def _prev_power_of_2(n):
    """Return the largest power of 2 that is <= n."""
    if n < 1:
        return 1
    p = 1
    while p * 2 <= n:
        p *= 2
    return p


def _select_block_sizes(M, N, K_local):
    """Select block sizes for persistent two_shot with iris P2P communication.

    Smaller bm creates more tiles, giving more parallel CTAs for the
    communication phase of two_shot. bm=64 is optimal for large M,
    bm=32 for small M. bn=128 balances GEMM efficiency and tile count.

    All block sizes must be powers of 2 (Triton requirement).
    bm must be <= M (kernel assertion: M >= block_size_m).
    """
    if M <= 1024:
        bm = 32
    else:
        bm = 64

    # Clamp bm to the largest power-of-2 <= M (handles small M from CUDAGraph capture)
    if bm > M:
        bm = _prev_power_of_2(M)
    # Minimum usable block size is 16 (below this, GEMM is too small to benefit from fusion)
    bm = max(bm, 16)

    bn = 128
    while bn > 16 and bn > N:
        bn //= 2

    bk = 64
    while bk > 8 and bk > K_local:
        bk //= 2
    return bm, bn, bk


def gemm_all_reduce_k_shard_wrapper(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    reduce_dim: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper for iris.ops.matmul_all_reduce, registered as
    torch.ops.aiter.fused_gemm_all_reduce_k_shard.

    Uses two_shot variant with versioned locks for zero pre-kernel overhead.
    Caches the output buffer (via as_symmetric) and workspace across calls.

    Args:
        x: Local input tensor (M, K_local)
        weights: List containing one weight tensor [(K_local, N)]
        reduce_dim: Unused (kept for op signature compatibility)
        group_name: Unused (kept for op signature compatibility)

    Returns:
        (C_local, C_global)
        - C_local: zeros placeholder with shape (M, N)
        - C_global: All-reduced result (M, N) in iris shared memory
    """
    global _CACHED_NUM_SMS

    iris_inst = _get_iris()
    weight = weights[0]
    M, K_local = x.shape
    N = weight.shape[1]

    # Cache SM count once (host-side query forbidden during CUDAGraph capture)
    if _CACHED_NUM_SMS is None:
        _CACHED_NUM_SMS = torch.cuda.get_device_properties(
            x.device
        ).multi_processor_count

    # Reuse symmetric output buffer when (M, N, dtype) match.
    # Dict-based cache supports multiple shapes so that different layers
    # (with different N) don't trigger as_symmetric() during CUDAGraph capture.
    cache_key = (M, N, x.dtype)
    if cache_key not in _BUFFER_CACHE:
        buf = torch.empty((M, N), dtype=x.dtype, device=x.device)
        c_global = iris_inst.as_symmetric(buf)
        c_local = torch.empty((M, N), dtype=x.dtype, device=x.device)
        workspace = FusedWorkspace()
        _BUFFER_CACHE[cache_key] = (c_global, c_local, workspace)

    c_global, c_local, workspace = _BUFFER_CACHE[cache_key]

    bm, bn, bk = _select_block_sizes(M, N, K_local)

    config = FusedConfig(
        block_size_m=bm,
        block_size_n=bn,
        block_size_k=bk,
        group_size_m=4,
        num_xcds=8,
        num_sms=_CACHED_NUM_SMS,
        all_reduce_variant="two_shot",
    )

    workspace = matmul_all_reduce(
        iris_inst,
        c_global,
        x,
        weight,
        config=config,
        workspace=workspace,
        async_op=True,  # two_shot uses versioned locks; post-kernel barrier is unnecessary
    )
    _BUFFER_CACHE[cache_key] = (c_global, c_local, workspace)

    return c_local, c_global


def warmup_buffers(compile_sizes=None, weight_shapes=None, dtype=torch.bfloat16):
    """Pre-cache iris state and optionally pre-allocate symmetric buffers.

    Must be called collectively on all ranks before CUDAGraph capture.
    This ensures that host-side operations (device_context tensor creation,
    SM count query, as_symmetric IPC mapping) happen outside the capture region.

    Args:
        compile_sizes: Optional list of M dimensions that will be compiled/captured.
            If weight_shapes is also provided, pre-allocates buffers for all
            (M, N) combinations.
        weight_shapes: Optional list of (K_local, N) weight shapes. Only the N
            dimension is used. If None, only caches device_context and num_sms;
            per-shape buffers will be created during the warmup forward pass.
        dtype: Tensor dtype. Default: torch.bfloat16.
    """
    global _CACHED_NUM_SMS

    iris_inst = _get_iris()
    device = iris_inst.device

    # Cache device_context tensor (avoids torch.tensor() during capture)
    iris_inst.get_device_context()

    # Cache SM count (avoids get_device_properties() during capture)
    if _CACHED_NUM_SMS is None:
        _CACHED_NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count

    warmed = 0
    if compile_sizes and weight_shapes:
        n_values = sorted(set(n for _, n in weight_shapes))
        for M in compile_sizes:
            for N in n_values:
                cache_key = (M, N, dtype)
                if cache_key not in _BUFFER_CACHE:
                    buf = torch.empty((M, N), dtype=dtype, device=device)
                    c_global = iris_inst.as_symmetric(buf)
                    c_local = torch.empty((M, N), dtype=dtype, device=device)
                    workspace = FusedWorkspace()
                    _BUFFER_CACHE[cache_key] = (c_global, c_local, workspace)
                    warmed += 1

    print(
        f"✓ Iris warmup complete: cached device_context + num_sms={_CACHED_NUM_SMS}"
        + (f", pre-allocated {warmed} buffer(s)" if warmed else "")
    )


def gemm_all_reduce_k_shard_fake(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    reduce_dim: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for meta mode / shape inference."""
    M = x.shape[0]
    N = weights[0].shape[1]
    local_output = torch.empty((M, N), dtype=x.dtype, device=x.device)
    global_output = torch.empty((M, N), dtype=x.dtype, device=x.device)
    return local_output, global_output


# Register the custom op
try:
    from torch.library import Library
    from csrc.cpp_itfs.torch_utils import direct_register_custom_op

    _fused_ops_lib = Library("aiter", "FRAGMENT")

    direct_register_custom_op(
        op_name="fused_gemm_all_reduce_k_shard",
        op_func=gemm_all_reduce_k_shard_wrapper,
        mutates_args=[],
        fake_impl=gemm_all_reduce_k_shard_fake,
        target_lib=_fused_ops_lib,
        dispatch_key="CUDA",
    )

    print("✓ Registered torch.ops.aiter.fused_gemm_all_reduce_k_shard custom op")
except Exception as e:
    print(f"⚠ Failed to register aiter.fused_gemm_all_reduce_k_shard custom op: {e}")
    import traceback

    traceback.print_exc()


# Public API
__all__ = [
    "gemm_all_reduce_k_shard_wrapper",
    "gemm_all_reduce_k_shard_fake",
    "init_iris",
    "warmup_buffers",
]
