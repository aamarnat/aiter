"""
Fused GEMM + All-Scatter operations using Iris RDMA communication.

This module provides fused GEMM and all-scatter operations optimized for
N-dimension sharding using Iris persistent kernels.

Registers torch.ops.aiter.fused_gemm_all_scatter_n_shard custom op.
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

# Import persistent_gemm_all_scatter kernel
# Add iris examples directory to path
# Path: aiter/aiter/ops/triton/comms/fused/fused_gemm_all_scatter.py -> need 7 parents to reach /workspace
workspace_path = Path(__file__).parent.parent.parent.parent.parent.parent.parent  # /workspace
iris_examples_path = workspace_path / "iris" / "examples" / "07_gemm_all_scatter"
if iris_examples_path.exists() and str(iris_examples_path) not in sys.path:
    sys.path.insert(0, str(iris_examples_path))

IRIS_AVAILABLE = False
from gemm_all_scatter import persistent_gemm_all_scatter  # type: ignore
IRIS_AVAILABLE = True
print("✓ persistent_gemm_all_scatter kernel available")


def gemm_all_scatter_n_shard_wrapper(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    scatter_dim: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Wrapper function for GEMM + all-scatter to be used as a torch custom op (aiter.fused_gemm_all_scatter_n_shard).
    
    This variant handles N-dimension sharding where each rank computes a portion
    of the output N-dimension and scatters it to all other ranks.
    
    Args:
        x: Input tensor (M, K)
        weights: List of weight tensors [(K, N_local)]
        scatter_dim: Dimension along which to scatter (1 for N dimension)
        group_name: Process group name
        
    Returns:
        Tuple of (local_output, global_scattered_output)
    """
    print(f"[EXEC] gemm_all_scatter_n_shard_wrapper called (device: {x.device})")
    
    if not IRIS_AVAILABLE or persistent_gemm_all_scatter is None:
        raise RuntimeError("persistent_gemm_all_scatter kernel not available")
    
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
    
    # Unpack dimensions - each rank computes a portion of N dimension
    M = x.shape[0]
    K = x.shape[1]
    weight = weights[0]
    N_local = weight.shape[1]
    N_global = N_local * world_size

    # Allocate output tensors
    C_local = torch.zeros((M, N_local), dtype=x.dtype, device=x.device)  # Local GEMM output
    
    # Allocate global output tensor in Iris's shared memory heap for remote access
    C_global = shmem.zeros((M, N_global), dtype=x.dtype, device=x.device)

    # Kernel parameters
    BLOCK_M = 256
    BLOCK_N = 64
    BLOCK_K = 64
    GROUP_SIZE_M = 6
    NUM_XCDS = 1  # Single chiplet
    # Use actual GPU multiprocessor count
    NUM_SMS = torch.cuda.get_device_properties(x.device).multi_processor_count
    
    # Compute strides
    stride_am = x.stride(0)
    stride_ak = x.stride(1)
    stride_bk = weight.stride(0)
    stride_bn = weight.stride(1)
    stride_cm = C_local.stride(0)
    stride_cn = C_local.stride(1)
    stride_cm_global = C_global.stride(0)
    stride_cn_global = C_global.stride(1)
    
    # Get heap bases from shmem for RDMA communication
    heap_bases = shmem.get_heap_bases()
    
    # Kernel configuration
    # EVEN_K checks if K is evenly divisible by BLOCK_K
    EVEN_K = (K % BLOCK_K == 0)
    
    # Launch the persistent_gemm_all_scatter kernel
    grid = (NUM_SMS,)
    
    # Bias parameters (not used in this version)
    bias_ptr = None
    stride_bias = 0
    BIAS = False
    
    persistent_gemm_all_scatter[grid](
        x,
        weight,
        C_local,
        C_global,
        bias_ptr,
        M,
        N_local,
        K,
        stride_am,
        stride_ak,
        stride_bk,
        stride_bn,
        stride_cm,
        stride_cn,
        stride_cm_global,
        stride_cn_global,
        stride_bias,
        BLOCK_SIZE_M=BLOCK_M,
        BLOCK_SIZE_N=BLOCK_N,
        BLOCK_SIZE_K=BLOCK_K,
        GROUP_SIZE_M=GROUP_SIZE_M,
        NUM_SMS=NUM_SMS,
        NUM_XCDS=NUM_XCDS,
        BIAS=BIAS,
        EVEN_K=EVEN_K,
        heap_bases=heap_bases,
        cur_rank=cur_rank,
        world_size=world_size,
    )

    torch.cuda.synchronize()
    shmem.barrier()
    dist.barrier()
    
    return C_local, C_global


def gemm_all_scatter_n_shard_fake(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    scatter_dim: int,
    group_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for meta mode / shape inference (N-sharding)"""
    
    # Infer shapes - sharding on N dimension
    world_size = 2  # Default assumption
    M = x.shape[0]
    K = x.shape[1]
    N_local = weights[0].shape[1]
    N_global = N_local * world_size
    
    local_output = torch.empty((M, N_local), dtype=x.dtype, device=x.device)
    global_output = torch.empty((M, N_global), dtype=x.dtype, device=x.device)
    
    return local_output, global_output


# Register the custom op for N-sharding
if IRIS_AVAILABLE:
    try:
        from torch.library import Library
        from csrc.cpp_itfs.torch_utils import direct_register_custom_op
        
        # Create a FRAGMENT library for fused operations
        _fused_ops_lib = Library("aiter", "FRAGMENT")
        
        direct_register_custom_op(
            op_name="fused_gemm_all_scatter_n_shard",
            op_func=gemm_all_scatter_n_shard_wrapper,
            mutates_args=[],
            fake_impl=gemm_all_scatter_n_shard_fake,
            target_lib=_fused_ops_lib,
            dispatch_key="CUDA",
        )
        
        print("✓ Registered torch.ops.aiter.fused_gemm_all_scatter_n_shard custom op")
    except Exception as e:
        print(f"⚠ Failed to register aiter.fused_gemm_all_scatter_n_shard custom op: {e}")
        import traceback
        traceback.print_exc()
else:
    print("⚠ Cannot register aiter.fused_gemm_all_scatter_n_shard: persistent_gemm_all_scatter kernel not available")


# Public API
__all__ = [
    "gemm_all_scatter_n_shard_wrapper",
    "gemm_all_scatter_n_shard_fake",
    "IRIS_AVAILABLE",
]
