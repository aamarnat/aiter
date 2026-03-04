"""
Fused GEMM + All-Reduce operations using Iris RDMA communication.

This module provides fused GEMM and all-reduce operations optimized for
K-dimension sharding using Iris persistent kernels with atomic operations.

Registers torch.ops.aiter.fused_gemm_all_reduce_k_shard custom op.
"""

import torch
import torch.distributed as dist
import os

# Set environment variable for Triton
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

import iris

from iris.ops.matmul_all_reduce import matmul_all_reduce
from iris.ops.config import FusedConfig

print("✓ persistent_gemm_all_reduce kernel available")


def gemm_all_reduce_k_shard_wrapper(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    reduce_dim: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper for iris.ops.matmul_all_reduce, registered as
    torch.ops.aiter.fused_gemm_all_reduce_k_shard.

    Each rank holds a local input (M, K_local) and a local weight shard
    (K_local, N).  The kernel computes C_local = x @ weight, then atomically
    accumulates across all ranks so every rank ends up with:

        C_global = sum_r (x_r @ weight_r)   (i.e. all_reduce of the local GEMMs)

    Args:
        x: Local input tensor (M, K_local)
        weights: List containing one weight tensor [(K_local, N)]
        reduce_dim: Unused (kept for op signature compatibility)
        group_name: Unused (kept for op signature compatibility)

    Returns:
        (C_local, C_global)
        - C_local: zeros placeholder with shape (M, N) — the fused kernel
                   writes directly into C_global without a separate local buffer
        - C_global: All-reduced result (M, N) in iris shared memory
    """
    print(f"[EXEC] gemm_all_reduce_k_shard_wrapper called (device: {x.device})")

    shmem = iris.iris()
    weight = weights[0]
    M, K_local = x.shape
    N = weight.shape[1]

    # Place x in iris shared memory so the kernel can use RDMA for the reduce
    A_iris = shmem.zeros(x.shape, dtype=x.dtype, device=x.device)
    A_iris.copy_(x)

    # Output in iris shared memory — matmul_all_reduce_preamble zeros it and
    # calls shmem.barrier() before the kernel launches
    C_global = shmem.zeros((M, N), dtype=x.dtype, device=x.device)

    # C_local is a placeholder; the fused kernel accumulates into C_global directly
    C_local = torch.zeros((M, N), dtype=x.dtype, device=x.device)

    # Choose block sizes that fit the actual problem dimensions
    bm = 128
    while bm > 16 and bm > M:
        bm //= 2
    bn = 64
    while bn > 16 and bn > N:
        bn //= 2
    bk = 64
    while bk > 8 and bk > K_local:
        bk //= 2

    config = FusedConfig(
        block_size_m=bm,
        block_size_n=bn,
        block_size_k=bk,
        group_size_m=1,
        num_xcds=1,
        all_reduce_variant="atomic",
    )

    # iris.ops.matmul_all_reduce: C_global = all_reduce(A_iris @ weight)
    # Calls shmem.barrier() internally (async_op=False)
    matmul_all_reduce(shmem, C_global, A_iris, weight, config=config)

    torch.cuda.synchronize()
    dist.barrier()

    return C_local, C_global


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
]
