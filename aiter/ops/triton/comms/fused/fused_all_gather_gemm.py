"""
Fused All-Gather + GEMM operations using Iris RDMA communication.

This module provides fused all-gather and GEMM operations optimized for
K-dimension sharding using Iris persistent kernels.

Registers torch.ops.aiter.fused_all_gather_gemm_k_shard custom op.
"""

import torch
import torch.distributed as dist
from pathlib import Path
import sys
import os

# Set environment variable for Triton
os.environ['TRITON_ALLOW_NON_CONSTEXPR_GLOBALS'] = '1'

# Import iris
import iris

# Import persistent_ag_gemm kernel
# Add iris examples directory to path
# Path: aiter/aiter/ops/triton/comms/fused/fused_all_gather_gemm.py -> need 7 parents to reach /workspace
workspace_path = Path(__file__).parent.parent.parent.parent.parent.parent.parent  # /workspace
iris_examples_path = workspace_path / "iris" / "examples" / "14_all_gather_gemm"
if iris_examples_path.exists() and str(iris_examples_path) not in sys.path:
    sys.path.insert(0, str(iris_examples_path))

IRIS_AVAILABLE = False
from all_gather_gemm_pull import persistent_ag_gemm  # type: ignore
IRIS_AVAILABLE = True
print("✓ persistent_ag_gemm kernel available")


def all_gather_gemm_k_shard_wrapper(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    gather_dim: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function for iris.x.all_gather_gemm to be used as a torch custom op.
    
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
    
    if not IRIS_AVAILABLE or persistent_ag_gemm is None:
        raise RuntimeError("persistent_ag_gemm kernel not available")
    
    shmem = iris.iris()
    
    # Get distributed info
    try:
        if dist.is_initialized():
            world_size = dist.get_world_size()
            cur_rank = dist.get_rank()
        else:
            # Not in distributed mode, use defaults for testing
            world_size = 1
            cur_rank = 0
    except (RuntimeError, ValueError):
        # Not in distributed mode, use defaults for testing
        world_size = 1
        cur_rank = 0
    
    # Unpack dimensions - sharding on K dimension
    M = x.shape[0]
    K_local = x.shape[1]
    K = K_local * world_size
    weight = weights[0]
    N = weight.shape[1]

    # Allocate a tensor in Iris's shared memory heap for remote access
    A_iris = shmem.zeros(x.shape, dtype=x.dtype, device=x.device)
    A_iris.copy_(x)
    
    # Allocate output tensors
    # Note: persistent_ag_gemm performs all-gather internally via RDMA (iris.load),
    A_gathered = torch.zeros((M, K), dtype=x.dtype, device=x.device)
    C = torch.zeros((M, N), dtype=x.dtype, device=x.device)  # Output tensor for kernel

    # Kernel parameters
    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_K = 64
    GROUP_SIZE_M = 6
    NUM_XCDS = 1  # Single chiplet
    # Use actual GPU multiprocessor count
    NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count
    
    # Compute strides
    stride_am = A_iris.stride(0)
    stride_ak = A_iris.stride(1)
    stride_bk = weight.stride(0)
    stride_bn = weight.stride(1)
    stride_cm = C.stride(0)
    stride_cn = C.stride(1)
    
    # Get heap bases from shmem for RDMA communication
    heap_bases = shmem.get_heap_bases()
    
    # Kernel configuration
    # EVEN_K checks if K (total) is evenly divisible by BLOCK_K
    EVEN_K = (K % BLOCK_K == 0)
    
    # Launch the persistent_ag_gemm kernel
    grid = (NUM_SMS,)
    
    persistent_ag_gemm[grid](
        A_iris,
        weight,
        C,
        M,
        N,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
        NUM_XCDS=NUM_XCDS,
        EVEN_K=EVEN_K,
        heap_bases=heap_bases,
        cur_rank=cur_rank,
        world_size=world_size,
    )

    torch.cuda.synchronize()
    shmem.barrier()
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
if IRIS_AVAILABLE:
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
else:
    print("⚠ Cannot register aiter.fused_all_gather_gemm_k_shard: persistent_ag_gemm kernel not available")


# Public API
__all__ = [
    "all_gather_gemm_k_shard_wrapper",
    "all_gather_gemm_k_shard_fake",
    "IRIS_AVAILABLE",
]
