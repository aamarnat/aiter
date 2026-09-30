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
- gemm + reduce_scatter (M-sharded)

Import isolation
----------------
Most of the modules below do an unguarded ``import iris`` (and run
``@triton.jit`` decorators) at module scope, so a missing or version-mismatched
iris raised here and took down the *whole* package -- every unrelated op in it
included. This package has hit that failure three times, and it is the reason a
per-module guard inside a single submodule is not enough: whichever iris-using
module is imported first poisons the package before the guarded one is reached.

Each iris-dependent submodule is therefore imported defensively here. A failure
demotes that one module's ops to "unavailable" and drops their names from
``__all__``, leaving the rest of the package importable. Importing a submodule
directly (``from ...fused.fused_gemm_all_reduce import x``) still raises, which
is correct: that caller asked for exactly that module.

``except Exception`` is deliberate: a triton/iris version mismatch surfaces as
AttributeError (or worse), not ImportError.
"""

from .reduce_scatter_rmsnorm_quant_all_gather import (
    reduce_scatter_rmsnorm_quant_all_gather,
)

__all__ = [
    "reduce_scatter_rmsnorm_quant_all_gather",
]


def _unavailable(module_name, names, exc):
    print(
        f"⚠ aiter.ops.triton.comms.fused: could not import {module_name}: "
        f"{type(exc).__name__}: {exc}. Its ops ({', '.join(names)}) are "
        f"unavailable; the rest of the package is unaffected."
    )


# All-Gather + GEMM (K-sharded)
try:
    from .fused_all_gather_gemm import (
        all_gather_gemm_k_shard_wrapper,
        all_gather_gemm_k_shard_fake,
    )

    __all__ += ["all_gather_gemm_k_shard_wrapper", "all_gather_gemm_k_shard_fake"]
except Exception as _e:  # pragma: no cover - environment dependent
    _unavailable(
        "fused_all_gather_gemm",
        ("all_gather_gemm_k_shard_wrapper", "all_gather_gemm_k_shard_fake"),
        _e,
    )

# GEMM + All-Scatter
try:
    from .fused_gemm_all_scatter import (
        gemm_all_scatter_n_shard_wrapper,
        gemm_all_scatter_n_shard_fake,
    )

    __all__ += ["gemm_all_scatter_n_shard_wrapper", "gemm_all_scatter_n_shard_fake"]
except Exception as _e:  # pragma: no cover - environment dependent
    _unavailable(
        "fused_gemm_all_scatter",
        ("gemm_all_scatter_n_shard_wrapper", "gemm_all_scatter_n_shard_fake"),
        _e,
    )

# GEMM + All-Reduce
try:
    from .fused_gemm_all_reduce import (
        gemm_all_reduce_k_shard_wrapper,
        gemm_all_reduce_k_shard_fake,
    )

    __all__ += ["gemm_all_reduce_k_shard_wrapper", "gemm_all_reduce_k_shard_fake"]
except Exception as _e:  # pragma: no cover - environment dependent
    _unavailable(
        "fused_gemm_all_reduce",
        ("gemm_all_reduce_k_shard_wrapper", "gemm_all_reduce_k_shard_fake"),
        _e,
    )

# GEMM + Reduce-Scatter (M-sharded) and standalone Reduce-Scatter
try:
    from .fused_gemm_reduce_scatter import (
        gemm_reduce_scatter_m_shard_wrapper,
        gemm_reduce_scatter_m_shard_fake,
        reduce_scatter_m_shard_wrapper,
        reduce_scatter_m_shard_fake,
    )

    __all__ += [
        "gemm_reduce_scatter_m_shard_wrapper",
        "gemm_reduce_scatter_m_shard_fake",
        "reduce_scatter_m_shard_wrapper",
        "reduce_scatter_m_shard_fake",
    ]
except Exception as _e:  # pragma: no cover - environment dependent
    _unavailable(
        "fused_gemm_reduce_scatter",
        (
            "gemm_reduce_scatter_m_shard_wrapper",
            "gemm_reduce_scatter_m_shard_fake",
            "reduce_scatter_m_shard_wrapper",
            "reduce_scatter_m_shard_fake",
        ),
        _e,
    )
