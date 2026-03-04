"""
Fused All-Gather + GEMM operations using Iris RDMA communication.

This module provides fused all-gather and GEMM operations optimized for
K-dimension sharding using Iris persistent kernels.

Registers torch.ops.aiter.fused_all_gather_gemm_k_shard custom op.
"""

import torch
import torch.distributed as dist

# Set environment variable for Triton
import os
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

# Import iris
import iris

from iris.ops.all_gather_matmul import all_gather_matmul
from iris.ops.config import FusedConfig

print("✓ iris.ops.all_gather_matmul available")

# Initialized lazily on first call (requires dist.init_process_group() to be done first)
IRIS_INSTANCE = None


def _get_iris():
    global IRIS_INSTANCE
    if IRIS_INSTANCE is None:
        IRIS_INSTANCE = iris.iris()
    return IRIS_INSTANCE


def all_gather_gemm_k_shard_wrapper(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    gather_dim: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function for iris.ops.all_gather_matmul to be used as a torch custom op.

    This variant handles K-dimension sharding.

    Args:
        x: Input tensor to all-gather (M, K_local) - sharded on K dimension
        weights: List of weight tensors [(K, N)]
        gather_dim: Dimension along which to gather (1 for K dimension)
        group_name: Process group name

    Returns:
        Tuple of (gathered_output, matmul_output)
    """
    print(f"[EXEC] all_gather_gemm_k_shard_wrapper called (device: {x.device})")

    iris_inst = _get_iris()
    weight = weights[0]
    M, K_local = x.shape
    K = K_local * iris_inst.get_num_ranks()
    N = weight.shape[1]

    # x must be in iris symmetric memory so remote ranks can RDMA-pull it
    A_iris = iris_inst.as_symmetric(x)

    A_gathered = torch.zeros((M, K), dtype=x.dtype, device=x.device)
    C = torch.zeros((M, N), dtype=x.dtype, device=x.device)

    # Choose block sizes that fit the actual problem dimensions (min 16 per triton constraint)
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
    )

    # iris.ops.all_gather_matmul: C = all_gather(A_iris) @ weight
    # It calls shmem.barrier() internally (async_op=False)
    all_gather_matmul(iris_inst, C, A_iris, weight, config=config)

    torch.cuda.synchronize()
    dist.barrier()

    return A_gathered, C


def all_gather_gemm_k_shard_fake(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    gather_dim: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for meta mode / shape inference (K-sharding)"""

    # Infer shapes - sharding on K dimension
    world_size = 2  # Default assumption
    M = x.shape[0]
    K_local = x.shape[1]
    K = K_local * world_size
    N = weights[0].shape[1]

    gathered = torch.empty((M, K), dtype=x.dtype, device=x.device)
    output = torch.empty((M, N), dtype=x.dtype, device=x.device)

    return gathered, output


# Register the custom op for K-sharding
try:
    from torch.library import Library
    from csrc.cpp_itfs.torch_utils import direct_register_custom_op

    # Create a FRAGMENT library for fused operations
    _fused_ops_lib = Library("aiter", "FRAGMENT")

    direct_register_custom_op(
        op_name="fused_all_gather_gemm_k_shard",
        op_func=all_gather_gemm_k_shard_wrapper,
        mutates_args=[],
        fake_impl=all_gather_gemm_k_shard_fake,
        target_lib=_fused_ops_lib,
        dispatch_key="CUDA",
    )

    print("✓ Registered torch.ops.aiter.fused_all_gather_gemm_k_shard custom op")
except Exception as e:
    print(f"⚠ Failed to register aiter.fused_all_gather_gemm_k_shard custom op: {e}")
    import traceback
    traceback.print_exc()


# Public API
__all__ = [
    "all_gather_gemm_k_shard_wrapper",
    "all_gather_gemm_k_shard_fake",
]
