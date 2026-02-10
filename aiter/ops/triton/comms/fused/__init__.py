# SPDX-License-Identifier: MIT
# Copyright (C) 2025, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused Communication + Computation Kernels

This submodule contains Triton kernels that fuse communication operations
with computation operations for improved performance.

Examples:
- reduce_scatter + rmsnorm + quant + all_gather
- all_reduce + rmsnorm + quant
- reduce_scatter + gemm + all_gather
- all_gather + gemm
- gemm + all_scatter
- gemm + all_reduce
"""

from .reduce_scatter_rmsnorm_quant_all_gather import (
    reduce_scatter_rmsnorm_quant_all_gather,
)

from .fused_all_gather_gemm import (
    all_gather_gemm_k_shard_wrapper,
    all_gather_gemm_k_shard_fake,
)

from .fused_gemm_all_scatter import (
    gemm_all_scatter_n_shard_wrapper,
    gemm_all_scatter_n_shard_fake,
)

from .fused_gemm_all_reduce import (
    gemm_all_reduce_k_shard_wrapper,
    gemm_all_reduce_k_shard_fake,
)

__all__ = [
    "reduce_scatter_rmsnorm_quant_all_gather",
    # All-Gather + GEMM
    "all_gather_gemm_k_shard_wrapper",
    "all_gather_gemm_k_shard_fake",
    # GEMM + All-Scatter
    "gemm_all_scatter_n_shard_wrapper",
    "gemm_all_scatter_n_shard_fake",
    # GEMM + All-Reduce
    "gemm_all_reduce_k_shard_wrapper",
    "gemm_all_reduce_k_shard_fake",
]
