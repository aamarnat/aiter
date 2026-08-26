# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Fused GEMM + ReduceScatter (and standalone ReduceScatter) over the M dimension.

Wraps ``iris.ops.matmul_reduce_scatter_fast.fast_reduce_scatter`` -- a persistent
one-shot *pull* reduce-scatter -- so that a vLLM Inductor pass can call it from
inside a HIP graph.

Registers two custom ops in the ``aiter`` namespace:

- ``torch.ops.aiter.gemm_reduce_scatter_m_shard(x, weights, bias, scatter_dim, group_name)``
  Fused path: ``reduce_scatter(x @ weights[0] + bias)`` over rows.
- ``torch.ops.aiter.reduce_scatter_m_shard(x, scatter_dim, group_name)``
  Standalone fallback for an already-computed ``(M, N)`` partial sum.

Sharding convention (matches ``iris``): ``x`` is ``(M, K_local)`` -- this rank's
K-shard -- and ``weights[0]`` is ``(K_local, N)``, i.e. ``(K, N)`` and *not*
``(N, K)``. vLLM's ``rocm_unquantized_gemm`` holds the weight as ``(N, K_local)``,
so the pass is expected to pass ``[weight.t()]``. The output is ``(M // world_size, N)``.

Capture safety
--------------
Every *symmetric-heap* allocation is collective (host barrier + heap-base
all-gather) and Iris never frees, so all heap allocation is confined to
``init_iris()`` and ``warmup_buffers()``, which must be called collectively on
all ranks *before* graph capture. A run-time cache miss allocates nothing: it
takes a row-prefix view of the staging slab warmup already reserved for the
scheduler's whole M range (see "Heap sizing" below). It still JIT-compiles the
RS kernel for that specialization, which capture forbids, so a miss inside a
capture region is refused -- but outside one it is cheap and expected. The
steady-state call is a couple of dict lookups, an ordinary ``torch.empty`` for
the output, a ``torch.mm``/``torch.addmm`` into the cached heap buffer, up to two
1-element ``dist.all_reduce`` sync tokens and the Triton RS launch -- no
allocation, no host sync, no device query.

Heap sizing
-----------
The symmetric heap is sized from the workload, not fixed. See the block comment
above ``_HEAP_ALIGNMENT`` for the failure that motivated it, why enumerating the
scheduler's shapes is impossible, and why one slab per (N, dtype) with per-shape
views is the answer. The entry points are ``heap_bytes_required()`` (pure
arithmetic, callable before the context exists), ``init_iris(heap_size=...)`` and
``warmup_buffers(max_m=...)``; ``AITER_GEMM_RS_HEAP_BYTES`` overrides the
derivation, and every number is printed at warmup.

The per-call ``torch.empty`` for the output is deliberate and is capture-legal:
ordinary caching-allocator requests made during capture are served from the
graph's private memory pool. Returning a cached output buffer instead would be a
correctness bug, because these ops are registered ``mutates_args=[]`` with a
``-> Tensor`` schema (i.e. declared pure): two calls sharing a cache key would
alias, and ``op(a) + op(b)`` would evaluate to ``2 * op(b)``.

Cross-rank synchronization and the published speedup
----------------------------------------------------
The underlying kernel does **no** cross-rank synchronization. Two stream-level
guards are needed for correctness (see ``_cross_rank_sync``):

- RAW, between the GEMM/copy and the RS launch: every peer's write to its
  staging buffer must land before anyone pulls it.
- WAR, after the RS: every peer's pull must finish before the shared staging
  buffer is overwritten by a later call.

Both are ON by default. The mode is selectable with the ``AITER_GEMM_RS_SYNC``
environment variable, read **once at import**:

- ``both`` (default): RAW + WAR. Correct.
- ``raw``: RAW only. Drops the WAR guard; a peer may overwrite staging while
  this rank is still pulling from it.
- ``none``: neither. Fastest and **racy**.

Be honest when quoting numbers: the 1.23-2.06x speedup reported for this kernel
was measured in the ``none`` configuration, which has a known, uncorrected
cross-rank race and was never correctness-checked. Any comparison against RCCL
must report *both* the ``none`` number and the ``both`` number; the ``both``
number is the only one that corresponds to a correct implementation, and each
guard has been budgeted at roughly 0.07 ms/call upstream -- large enough,
relative to a ~92 us reduce-scatter, to invert the result. The variable exists so
that delta can be measured without editing code.

That ``none`` configuration has since been measured end-to-end and is broken:
gsm8k 0.005 against 0.905 on an otherwise identical serving arm. So the guards
cannot simply be deleted. What *can* go is their cost.

Which primitive implements the guards -- ``AITER_GEMM_RS_SYNC_IMPL``
--------------------------------------------------------------------
``AITER_GEMM_RS_SYNC`` selects *how many* guards run. This second variable, also
read once at import, selects *what a guard is*:

- ``rccl`` (default): a 1-element ``dist.all_reduce``. Correct, and what every
  number recorded for this op so far was measured with. It is a full RCCL
  collective launch, budgeted at 15-50 us against a 30-90 us kernel -- which is
  the entire reason the correctly-synchronised op loses to plain RCCL.
- ``device``: ``iris.device_barrier()`` -- a single-CTA Triton kernel that
  atomically bumps this rank's flag on the symmetric heap
  (``atomic_add``, ``release``, ``sys``) and spin-polls every peer's flag for the
  same epoch (``atomic_cas``, ``acquire``, ``sys``). Same ordering guarantee,
  no RCCL launch. Graph-capturable, and iris tests it under capture-and-replay
  around a remote read and a remote write
  (``iris/tests/unittests/test_barriers.py``).

Both primitives give the same stream-level cross-rank ordering, so ``both``
remains correct under either. The default is deliberately left at ``rccl`` so
that the recorded 0.36-0.99x baseline stays reproducible without editing code;
``device`` is the configuration under test.

Every rank must agree on *both* variables. The mode fixes how many collectives
each call issues and the impl fixes which collective, so a rank that disagrees
on either deadlocks -- one rank waiting in RCCL while another spins on heap
flags produces no diagnostic at all. ``warmup_buffers`` checks both
(``_check_sync_mode_agreement``).
"""

import os
import sys
import time

import torch
import torch.distributed as dist

# Must be set before iris/triton import: the iris kernels read module-level
# globals (heap bases) that are not constexpr.
os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")

IRIS_AVAILABLE = False
try:
    import iris  # noqa: F401
    from iris.ops.matmul_reduce_scatter_fast import fast_reduce_scatter

    IRIS_AVAILABLE = True
except Exception as _iris_import_error:  # pragma: no cover - environment dependent
    # Deliberately broad, and deliberately non-fatal. This import transitively
    # runs @triton.jit at module scope and pulls in triton/iris internals, so a
    # version mismatch surfaces as AttributeError (or worse) rather than
    # ImportError.
    #
    # This guard is only load-bearing because the package __init__ guards the
    # sibling modules too: fused_all_gather_gemm / fused_gemm_all_scatter /
    # fused_gemm_all_reduce all `import iris` unguarded at module scope and are
    # imported before this module, so a broken iris used to poison the whole
    # package before this file was ever reached. See
    # aiter/ops/triton/comms/fused/__init__.py.
    iris = None
    fast_reduce_scatter = None
    print(
        f"⚠ Could not import iris fast reduce-scatter "
        f"(iris.ops.matmul_reduce_scatter_fast): {_iris_import_error}. "
        f"torch.ops.aiter.gemm_reduce_scatter_m_shard and "
        f"torch.ops.aiter.reduce_scatter_m_shard will not be registered; "
        f"the rest of aiter.ops.triton.comms.fused is unaffected."
    )


# --- cross-rank sync mode, resolved once at import (never per call) ----------
_SYNC_MODE = os.environ.get("AITER_GEMM_RS_SYNC", "both").strip().lower()
if _SYNC_MODE not in ("both", "raw", "none"):
    print(
        f"⚠ AITER_GEMM_RS_SYNC={_SYNC_MODE!r} is not one of 'both', 'raw', "
        f"'none'; falling back to 'both'."
    )
    _SYNC_MODE = "both"

# RAW guard: peers' writes to staging must land before anyone pulls.
_SYNC_RAW = _SYNC_MODE in ("both", "raw")
# WAR guard: peers' pulls must finish before staging is overwritten.
_SYNC_WAR = _SYNC_MODE == "both"

# --- which primitive implements a guard, resolved once at import -------------
# Orthogonal to _SYNC_MODE: that says how many guards, this says what one is.
# See the module docstring. Default stays 'rccl' so the recorded baseline is
# reproducible without editing code.
_SYNC_IMPL = os.environ.get("AITER_GEMM_RS_SYNC_IMPL", "rccl").strip().lower()
if _SYNC_IMPL not in ("rccl", "device"):
    print(
        f"⚠ AITER_GEMM_RS_SYNC_IMPL={_SYNC_IMPL!r} is not one of 'rccl', "
        f"'device'; falling back to 'rccl'."
    )
    _SYNC_IMPL = "rccl"


# --- diagnostic tracing, off unless asked for --------------------------------
# Set AITER_GEMM_RS_TRACE=1 to emit one line per symmetric-heap allocation and
# (at =2) one line per launch. Pair it with AMD_SERIALIZE_KERNEL=3: dispatch
# then becomes synchronous, so the last line printed before an HSA memory fault
# is the launch that caused it, rather than whatever the host happened to be
# doing when an asynchronous fault surfaced.
#
# This is DIAGNOSTIC ONLY -- it changes no behaviour when unset, and it is not
# the fix for anything. Reading it once at import keeps the hot path free.
_TRACE = int(os.environ.get("AITER_GEMM_RS_TRACE", "0") or 0)
_TRACE_ALLOC = _TRACE >= 1
_TRACE_LAUNCH = _TRACE >= 2


def _trace(msg: str) -> None:
    """Unbuffered, rank-tagged. stderr so it interleaves with the HSA fault."""
    try:
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else -1
    except Exception:
        rank = -1
    print(f"[rs-trace r{rank}] {msg}", file=sys.stderr, flush=True)


def _heap_offset():
    """Bytes handed out by the iris allocator so far, or None. Never raises."""
    try:
        return int(IRIS_INSTANCE.heap.allocator.heap_offset)
    except Exception:
        return None


IRIS_INSTANCE = None
_CACHED_NUM_SMS = None  # cached SM count to avoid a host query during graph capture

# --- symmetric-heap sizing, derived from the workload ------------------------
#
# THE FAILURE THIS REPLACES. The heap used to be a fixed 1 GiB (a default argument
# here, called with no argument from vLLM's pass_manager), and every distinct M got
# its own ``ctx.zeros((M, N))``. The iris allocator is a BUMP allocator that never
# frees, so each prefill width the scheduler invented consumed M*N*itemsize forever.
# On Qwen2.5-72B TP=8 that killed the server a few requests in:
#
#     [Iris] TorchAllocator: OOM requested=21889024 available=12582912
#
# Enumerating every M the scheduler can produce is not an option: M is any multiple
# of the TP size up to ``max_num_batched_tokens``, and one buffer each would need
# sum(M)*N*itemsize -- 68 GiB at max_num_batched_tokens=8192, N=8192, bf16.
#
# THE FIX. One SLAB per (N, dtype, world_size), sized for the largest M the
# scheduler can hand the op, and every shape is a row-prefix VIEW of it. Cost drops
# from O(sum of M) to O(M_max) -- 128 MiB for the case above -- and it covers every
# M <= M_max with no holes, so no shape can allocate at run time.
#
# The view is exactly as safe as a dedicated allocation: iris's reduce-scatter reads
# the tensor's own shape and strides and translates addresses as
# ``peer_base + (ptr - my_base)``. ``slab[:M]`` has ``data_ptr() == slab.data_ptr()``
# and slabs are allocated collectively in the same order on every rank, so that
# offset is rank-identical. Sharing one slab across different M is the same aliasing
# argument the buffer cache already makes for sharing one buffer across layers: the
# stream orders the calls, and the RAW/WAR guards extend that ordering across ranks.
_HEAP_ALIGNMENT = 1024  # iris TorchAllocator.allocate()'s default alignment

# Staging buffers per (N, dtype) slab set. ONE today.
#
# Task 3.1 (double-buffered staging) makes this 2 and is the reason sizing is
# expressed as a multiplier rather than a number: bumping this constant re-derives
# the heap instead of forcing a re-measurement. When 3.1 lands it must allocate TWO
# SEPARATE M_max slabs and alternate between them -- not two prefix views of one
# slab, which overlap and would defeat the WAR elimination that is the point of
# double buffering.
_STAGING_BUFFERS_PER_SHAPE = 1

# Headroom for iris's own heap traffic: ``device_barrier`` allocates one
# ``num_ranks``-element int32 flags tensor per process group (32 B at TP=8, rounded
# to 1 KiB), and future iris versions may add more. 1 MiB is ~0.8% of a typical
# derived requirement -- cheap insurance against an off-by-one-allocation refusal.
_HEAP_SLACK_BYTES = 1 << 20

# Only used when nobody derives a size. Kept so that callers predating the derived
# path (op_tests, the evidence scripts) behave exactly as before.
_DEFAULT_HEAP_BYTES = 1 << 30

# Escape hatch, read at init. Set it when the derivation is wrong for a deployment
# rather than editing code; the derived and effective values are both logged so the
# override is visible in the job log.
_HEAP_BYTES_ENV = "AITER_GEMM_RS_HEAP_BYTES"

# The largest M any staging slab must cover, declared by the caller that knows the
# scheduler's ceiling (vLLM's pass_manager passes max_num_batched_tokens). 0 means
# "not declared": slabs are then sized to the largest M actually seen, which is the
# pre-slab behaviour and is what the op_tests and evidence scripts get.
_MAX_M = 0

# (N, dtype, world_size) -> the (rows, N) symmetric-heap slab every M views into.
_SLABS = {}

# Set once warmup_buffers() has run. After that, a *new* (N, dtype) slab is heap the
# capacity check never budgeted for, which is a different and worse event than an
# unwarmed M -- so it is warned about separately.
_WARMUP_DONE = False

# Cached across calls to avoid per-call *collective* allocation. Keyed by
# (M, N, dtype) so that layers with different N do not trigger a symmetric-heap
# allocation during capture. Value is a flat tuple so the hot path is a single
# unpack:
#   (staging, sync_tensor, M_local, device, block_m, block_n, num_sms, num_warps)
# `staging` is the (M, N) symmetric-heap partial-sum buffer, `sync_tensor` a
# preallocated 1-element int32 token, and the last four entries are the RS launch
# config resolved once here rather than per call. The OUTPUT is deliberately not
# cached -- see the module docstring.
#
# Sharing one staging buffer across every layer with the same (M, N, dtype) is
# safe even though it looks unsafe: all calls are issued on the same stream, so a
# later op's GEMM cannot begin writing staging until the earlier op's RS kernel
# has finished reading it (stream order), and the WAR guard extends that ordering
# across ranks. With AITER_GEMM_RS_SYNC=raw|none the cross-rank half of that
# argument no longer holds -- that is exactly the race those modes accept.
_BUFFER_CACHE = {}

# One sync token is enough for every shape: it only carries stream ordering.
_SYNC_TENSOR = None

# Memoized TP world size per group_name, so the hot path is a dict lookup rather
# than a process-group resolve.
_GROUP_WORLD_SIZE = {}

# Shapes registered by callers ahead of an argument-less warmup_buffers().
_PENDING_SHAPES = set()  # {(M, N, dtype)}

# dtypes warmed when warmup_buffers() is called without an explicit dtype. Both
# are warmed because the cache key uses `x.dtype`: warming only bfloat16 makes an
# fp16 deployment miss on every shape and allocate inside the capture region.
_DEFAULT_WARMUP_DTYPES = (torch.bfloat16, torch.float16)

# Fallback probe shape for an argument-less warmup_buffers() with nothing
# registered. See warmup_buffers() for why this exists and what it does *not*
# accomplish. Expressed as (M_local_per_rank, N); M is scaled by world_size.
_DEFAULT_WARMUP_PROBE = (64, 128)

# Per-TP reduce-scatter launch configs, mirroring iris's _AUTO_CONFIG.
#
# NOTE: these constants are UNVERIFIED. They shipped with the iris kernel but
# disagree with the tuning sweep that accompanied it (e.g. TP2: packaged
# block_m=128/num_sms=196 vs sweep block_m=256/num_sms=304), and nothing in tree
# exercises them. W4 re-tunes them per shape; until then treat the performance
# numbers in these docstrings as a starting point, not a measurement.
_AUTO_CONFIG = {
    2: (128, 64, 196, 4),  # (block_m, block_n, num_sms, num_warps)
    4: (64, 64, 32, 4),
    8: (32, 64, 32, 4),
}

_DEFAULT_CONFIG = (64, 64, 64, 4)


def _align_up(n: int, to: int = _HEAP_ALIGNMENT) -> int:
    return -(-n // to) * to


def heap_bytes_required(
    max_m: int,
    n_values,
    dtypes,
    world_size: int = 1,
    buffers_per_shape: int | None = None,
) -> int:
    """Symmetric-heap bytes this op needs to cover every M up to ``max_m``.

    One staging slab per (N, dtype), each ``max_m`` rows tall, times the number of
    buffers a slab set needs (see ``_STAGING_BUFFERS_PER_SHAPE``), plus headroom for
    iris's own allocations. Pure arithmetic: no device, no iris, no side effects, so
    a caller can size the heap *before* creating the context.

    ``world_size`` is not a term -- every rank stages the full (M, N) partial and the
    reduce-scatter is what shards it -- but it is accepted so the signature matches
    everything else here and so the log line can name it.

    Args:
        max_m: Largest M the scheduler can hand the op. For vLLM that is
            ``max_num_batched_tokens`` rounded up to a multiple of the TP size.
        n_values: The N dimensions in play (the model's hidden size, for
            sequence-parallel residual reduce-scatter).
        dtypes: The dtypes in play. Two dtypes cost two slabs per N, because the
            buffer cache is keyed on ``x.dtype``.
        world_size: TP size, for the log line only.
        buffers_per_shape: Override for ``_STAGING_BUFFERS_PER_SHAPE``; used to ask
            "what would this cost after task 3.1".

    Returns:
        Bytes. Includes ``_HEAP_SLACK_BYTES``.
    """
    if buffers_per_shape is None:
        buffers_per_shape = _STAGING_BUFFERS_PER_SHAPE
    per_shape = 0
    for n in sorted(set(n_values)):
        for d in dtypes:
            itemsize = torch.tensor([], dtype=d).element_size()
            per_shape += _align_up(int(max_m) * int(n) * itemsize)
    return per_shape * buffers_per_shape + _HEAP_SLACK_BYTES


def _env_heap_bytes():
    """``AITER_GEMM_RS_HEAP_BYTES`` as an int, or None. Never raises."""
    raw = (os.environ.get(_HEAP_BYTES_ENV) or "").strip()
    if not raw:
        return None
    try:
        value = int(raw, 0)
    except ValueError:
        print(f"⚠ {_HEAP_BYTES_ENV}={raw!r} is not an integer; ignoring it.")
        return None
    if value <= 0:
        print(f"⚠ {_HEAP_BYTES_ENV}={value} is not positive; ignoring it.")
        return None
    return value


def init_iris(heap_size: int | None = None, ctx=None):
    """Pre-initialize (or adopt) the iris context for the reduce-scatter ops.

    This MUST be called collectively on ALL ranks before any of these ops
    execute (e.g. before graph capture). It allocates the iris symmetric heap,
    which requires a coordinated IPC handle exchange.

    ``iris.iris()`` is a plain factory, not a singleton, so building one context
    per fusion would give the process two symmetric heaps and two IPC handle
    exchanges. When the all-reduce sibling has already initialized a context,
    that one is adopted instead of creating a second.

    Args:
        heap_size: Symmetric heap size in bytes. **Pass a value derived from the
            workload** -- ``heap_bytes_required(...)`` -- rather than relying on the
            fallback. None means ``_DEFAULT_HEAP_BYTES``, which is the fixed 1 GiB
            that made the original exhaustion a week-long mystery; it is kept only
            so callers predating the derived path behave as before, and it says so
            in the log. ``AITER_GEMM_RS_HEAP_BYTES`` overrides either.
        ctx: An existing iris context to adopt. Takes priority over everything
            else.

    Returns:
        The iris context.
    """
    global IRIS_INSTANCE
    if ctx is not None:
        IRIS_INSTANCE = ctx
        return IRIS_INSTANCE
    if IRIS_INSTANCE is not None:
        return IRIS_INSTANCE
    if not IRIS_AVAILABLE:
        raise RuntimeError(
            "iris is not importable; the fused reduce-scatter ops are unavailable. "
            "See the import warning emitted by "
            "aiter.ops.triton.comms.fused.fused_gemm_reduce_scatter."
        )

    sibling_ctx = _sibling_iris_instance()
    if sibling_ctx is not None:
        IRIS_INSTANCE = sibling_ctx
        # The adopted heap was sized by the all-reduce fusion, which knows nothing
        # about this op's shapes, so `heap_size` is dropped on the floor here. Say so
        # -- silently ignoring a derived size is how the original bug hid. Whether
        # the adopted heap is actually big enough is decided by
        # `_check_heap_capacity()` at warmup, against the real numbers.
        print(
            "✓ Iris context for fused GEMM+RS adopted from fused_gemm_all_reduce "
            "(one symmetric heap shared by both fusions). NOTE: the adopted heap "
            f"is {_heap_total() or 0} bytes, sized by that fusion; the "
            f"heap_size={heap_size} requested here is ignored. warmup_buffers() "
            "will refuse if it is too small."
        )
        return IRIS_INSTANCE

    derived = heap_size
    override = _env_heap_bytes()
    effective = override if override is not None else heap_size
    if effective is None:
        effective = _DEFAULT_HEAP_BYTES

    IRIS_INSTANCE = iris.iris(heap_size=effective)
    _publish_iris_instance(IRIS_INSTANCE)
    if override is not None:
        source = f"{_HEAP_BYTES_ENV} override (derived was {derived})"
    elif derived is not None:
        source = "derived from the workload"
    else:
        source = (
            "FALLBACK DEFAULT -- no caller derived a size, so this heap is not "
            "sized for any particular workload"
        )
    print(
        f"✓ Iris context initialized for fused GEMM+RS: "
        f"rank {IRIS_INSTANCE.get_rank()}/{IRIS_INSTANCE.get_num_ranks()}, "
        f"heap={effective} B ({effective / (1 << 20):.0f} MiB), {source}"
    )
    return IRIS_INSTANCE


def _heap_total():
    """Total bytes of the symmetric heap, or None. Never raises."""
    try:
        return int(IRIS_INSTANCE.heap.allocator.heap_size)
    except Exception:
        return None


def _heap_base():
    """Local base address of the symmetric heap, or None. Never raises."""
    try:
        return int(IRIS_INSTANCE.heap.allocator.get_base_address())
    except Exception:
        return None


def _check_slab_offset_agreement(slab, N: int, dtype, world_size: int) -> None:
    """Verify every rank put this slab at the SAME heap offset. Collective.

    This is the invariant the whole slab-and-view design rests on, and it is
    invisible to any single-rank test.

    The RS kernel reads a peer's staging buffer as ``peer_base + (ptr - my_base)``
    -- it never learns the peer's actual pointer. So a row-prefix view ``slab[:M]``
    resolves to the right memory on a peer if and only if that peer's slab sits at
    the same offset into ITS heap. It does, because iris's allocator is a bump
    allocator and every rank runs the identical allocation sequence: the shape set
    is derived from the same model config, deduplicated and then SORTED, so the warm
    loop visits shapes in the same order everywhere; ``_create_entry`` (which
    allocates the slab) runs before ``_warm_both_sync_impls`` (which allocates the
    barrier flags) on every rank; and iris never frees, so an offset can never be
    reused out of order.

    That is an argument, though, and the failure it guards against is silent: a rank
    whose slab is 1 KiB further along would read a shifted window of every peer's
    partial sums and produce plausible-looking wrong numbers with no fault. So the
    argument is checked rather than trusted. One all-reduce at warmup, where a
    collective is free.

    Encoded as ``(offset, -offset)`` reduced with MAX for the same reason
    ``_check_sync_mode_agreement`` does it: MAX alone cannot tell "everyone agrees
    on 4096" from "someone else said 4096".
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    base = _heap_base()
    if base is None:
        return
    mine = int(slab.data_ptr()) - base
    t = torch.tensor([mine, -mine], dtype=torch.int64, device=slab.device)
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    got = [int(v) for v in t]
    if got != [mine, -mine]:
        raise RuntimeError(
            f"fused GEMM+RS: the staging slab for (N={N}, dtype={dtype}, "
            f"world_size={world_size}) is at heap offset {mine} on this rank but not "
            f"on every rank (max seen {got[0]}, min seen {-got[1]}). The "
            f"reduce-scatter addresses a peer's staging as peer_base + (ptr - "
            f"my_base), so mismatched offsets make every rank read the WRONG peer "
            f"memory and return plausible but wrong results with no fault. The ranks "
            f"have diverged in allocation order: warmup must be called collectively, "
            f"with the same shapes and dtypes, before anything else touches the heap."
        )


def _publish_iris_instance(ctx) -> None:
    """Share our context with the all-reduce sibling.

    ``iris.iris()`` is a factory, not a singleton, so without this the module that
    initializes second builds a second symmetric heap and does a second IPC handle
    exchange. Adoption has to be bidirectional because the pass order is not pinned.
    """
    try:
        from . import fused_gemm_all_reduce
    except Exception:  # pragma: no cover - environment dependent
        return
    if getattr(fused_gemm_all_reduce, "IRIS_INSTANCE", None) is None:
        fused_gemm_all_reduce.IRIS_INSTANCE = ctx


def _sibling_iris_instance():
    """The all-reduce sibling's iris context, if it has one. Never raises."""
    try:
        from . import fused_gemm_all_reduce
    except Exception:  # pragma: no cover - environment dependent
        return None
    return getattr(fused_gemm_all_reduce, "IRIS_INSTANCE", None)


def _get_iris():
    global IRIS_INSTANCE
    if IRIS_INSTANCE is None:
        sibling_ctx = _sibling_iris_instance()
        if sibling_ctx is not None:
            IRIS_INSTANCE = sibling_ctx
            return IRIS_INSTANCE
        raise RuntimeError(
            "Iris not initialized. Call init_iris() collectively on all "
            "ranks before fused kernel execution (e.g. during pass setup)."
        )
    return IRIS_INSTANCE


def _resolve_config(world_size: int, M_local: int) -> tuple[int, int, int, int]:
    """Resolve the RS launch config once, at cache-fill time.

    The block_m halving loop is a pure performance heuristic: the kernel masks
    the M tail, so an oversized block_m is correct, just wasteful. num_sms is
    clamped to the device CU count for every world size, not just the fallback
    path -- the packaged TP2 entry asks for 196 WGs, which overshoots some parts.
    """
    block_m, block_n, num_sms, num_warps = _AUTO_CONFIG.get(world_size, _DEFAULT_CONFIG)
    if _CACHED_NUM_SMS is not None:
        num_sms = min(num_sms, _CACHED_NUM_SMS)
    while block_m > M_local and block_m > 4:
        block_m //= 2
    return block_m, block_n, num_sms, num_warps


def _world_size_from_group(group_name: str) -> int:
    """Authoritative world size for these ops: the TP group named by ``group_name``.

    Both the real and the fake implementation go through here, on purpose. The
    fake sizes every downstream Inductor buffer, so if it disagreed with the
    kernel by even one factor the result would be silent corruption or a shape
    crash. Deriving both from the process group the op semantically collectives
    over keeps them in lockstep.

    The real path additionally cross-checks this against the iris communicator
    (see ``_create_entry``): the iris heap spans WORLD, and the RS kernel pulls
    from every WORLD rank regardless of what group_name says.
    """
    cached = _GROUP_WORLD_SIZE.get(group_name)
    if cached is not None:
        return cached

    world_size = None
    if group_name:
        # A named group that cannot be resolved must NOT silently fall back to
        # WORLD: that is exactly the fake-vs-real divergence this function exists
        # to prevent, and memoization would make it permanent.
        from torch.distributed.distributed_c10d import _resolve_process_group

        try:
            group = _resolve_process_group(group_name)
        except Exception as e:
            raise RuntimeError(
                f"Cannot resolve process group {group_name!r} for reduce-scatter. "
                f"Refusing to guess a world size, since the traced (fake) and "
                f"executed shapes would then disagree."
            ) from e
        world_size = dist.get_world_size(group)
    if world_size is None and IRIS_INSTANCE is not None:
        world_size = IRIS_INSTANCE.get_num_ranks()
    if world_size is None and dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
    if world_size is None:
        raise RuntimeError(
            f"Cannot determine world size for reduce-scatter: process group "
            f"{group_name!r} is unknown, iris is uninitialized and the default "
            f"process group is not initialized."
        )

    _GROUP_WORLD_SIZE[group_name] = world_size
    return world_size


def _check_iris_world_size(world_size: int, group_name: str) -> None:
    """Fail loudly when the TP group and the iris communicator disagree.

    iris calls ``init_distributed()`` with no group, so its context spans the
    DEFAULT process group -- WORLD. The RS kernel pulls a shard from every rank
    in that communicator. If TP != WORLD (any DP or PP deployment) the op cannot
    do what its name says, and the failure is otherwise silent: the fake reports
    ``M // tp`` rows while the kernel writes ``M // world``.

    Making a TP subgroup work would require an iris context per subgroup and is
    explicitly out of scope; this turns a miscompile into an explicable error.
    """
    iris_world = _get_iris().get_num_ranks()
    if world_size != iris_world:
        raise RuntimeError(
            f"Reduce-scatter group mismatch: process group {group_name!r} has "
            f"world_size={world_size}, but the iris context spans "
            f"{iris_world} ranks. iris initializes over the DEFAULT process "
            f"group (WORLD) and the reduce-scatter kernel pulls from all "
            f"{iris_world} of those ranks, so it cannot implement a "
            f"{world_size}-rank collective. The fused reduce-scatter ops "
            f"currently require TP == WORLD; disable them for this deployment "
            f"(DP/PP with a TP subgroup is not supported)."
        )


def _check_scatter_dim(scatter_dim: int) -> None:
    """Reject a scatter dim other than the row dim.

    Two integer comparisons; no device interaction, no allocation, no sync, so
    this is safe to leave on the hot path. A silently mis-scattered output is
    far more expensive than the comparison.
    """
    if scatter_dim not in (0, -2):
        raise ValueError(
            f"scatter_dim={scatter_dim} is not supported: the iris fast "
            f"reduce-scatter shards over rows (dim 0) of a 2D (M, N) tensor."
        )


def _check_divisible(M: int, world_size: int) -> None:
    if world_size <= 0 or M % world_size != 0:
        raise ValueError(
            f"M={M} must be divisible by world_size={world_size}: the "
            f"reduce-scatter gives each rank exactly M // world_size rows."
        )


def _refuse_if_capturing(M: int, N: int, dtype) -> None:
    """Fail loudly rather than deadlocking if a cache miss happens mid-capture.

    Filling a cache entry JIT-compiles the RS kernel for that specialization, which
    is not capture-safe, and -- if the (N, dtype) slab does not exist yet -- performs
    a collective ``ctx.zeros``. A miss is rank-local, so the collective case either
    hangs on peers that are not capturing or diverges the collective count across
    ranks. Both present as an unexplained hang; this turns them into a message
    naming the shape that was not warmed.

    Since the slabs cover every M up to the declared ceiling, reaching this inside a
    capture region means a shape vLLM captured was not warmed -- a genuine bug in
    the caller's warm set, not the heap running out.
    """
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:
        return
    if capturing:
        raise RuntimeError(
            f"fused GEMM+RS: cache miss for (M={M}, N={N}, dtype={dtype}) during "
            f"CUDA/HIP graph capture. Filling the cache JIT-compiles the kernel, "
            f"which capture forbids. Every CAPTURED shape must be warmed: pass it in "
            f"warmup_buffers(compile_sizes=..., weight_shapes=...) or register_shape()."
        )


def declare_max_m(max_m: int) -> None:
    """Declare the largest M the scheduler can hand these ops.

    Sizing and coverage both key off this one number, which is what keeps them from
    drifting apart: ``heap_bytes_required(max_m, ...)`` reserves exactly the slab the
    staging views are cut from, so any M the slab can hold is a shape the heap was
    sized for, by construction.

    Callers that do not know their ceiling may skip this; slabs are then sized to the
    largest M actually seen, which is what the op_tests and the evidence scripts do.
    """
    global _MAX_M
    _MAX_M = max(_MAX_M, int(max_m))


def _get_slab(N: int, dtype: torch.dtype, world_size: int, M: int):
    """The (rows, N) staging slab for (N, dtype), allocating it if absent. COLLECTIVE.

    ``ctx.zeros`` ends in a host ``dist.barrier()`` plus a heap-base all-gather and
    Iris never frees, so this must run at warmup: it is not legal during graph
    capture. Every M is a row-prefix view of what this returns, so it runs at most
    once per (N, dtype, world_size) when a ceiling has been declared.
    """
    key = (N, dtype, world_size)
    slab = _SLABS.get(key)
    if slab is not None and M <= slab.shape[0]:
        return slab

    if _MAX_M and M > _MAX_M:
        # The scheduler exceeded the ceiling the heap was sized from. Refuse rather
        # than allocate: growing the heap on the hot path is the never-freeing
        # bump-allocator failure this whole design exists to remove, and the real
        # defect is that the declared ceiling is wrong.
        raise RuntimeError(
            f"fused GEMM+RS: M={M} exceeds the declared ceiling M_max={_MAX_M} for "
            f"(N={N}, dtype={dtype}, world_size={world_size}). The symmetric heap was "
            f"sized for {_MAX_M} rows and nothing larger can be staged without "
            f"allocating mid-forward. Raise the ceiling (vLLM: "
            f"max_num_batched_tokens) so the heap is re-derived from it, or cap the "
            f"scheduler at {_MAX_M} tokens."
        )

    if slab is None and _WARMUP_DONE:
        # An N that warmup never saw, so `_check_heap_capacity` never budgeted for
        # it. This DOES allocate on the never-freeing heap. It cannot happen in the
        # vLLM path (sequence parallelism reduce-scatters only the residual stream,
        # so there is exactly one N, and warmup is told it) -- but if it ever does,
        # the heap can be exhausted again and the log must say why.
        print(
            f"⚠ fused GEMM+RS: allocating a staging slab for (N={N}, dtype={dtype}) "
            f"AFTER warmup. This N was not in the warm set, so the heap budget did "
            f"not include it: {max(_MAX_M, M)} x {N} more bytes are being taken from "
            f"a heap that never frees. Declare this N at warmup "
            f"(warmup_buffers(weight_shapes=...) or register_shape())."
        )

    rows = max(_MAX_M, M)
    iris_inst = _get_iris()
    if _TRACE_ALLOC:
        _off_before = _heap_offset()
        _t0 = time.perf_counter()
        _trace(
            f"ALLOC begin M={rows} N={N} dtype={dtype} world={world_size} "
            f"heap_offset_before={_off_before}"
        )
    new_slab = iris_inst.zeros((rows, N), dtype=dtype)
    if _TRACE_ALLOC:
        _trace(
            f"ALLOC end   M={rows} N={N} heap_offset_after={_heap_offset()} "
            f"wall_ms={(time.perf_counter() - _t0) * 1e3:.2f} "
            f"staging_ptr=0x{new_slab.data_ptr():x}"
        )
    if slab is not None:
        # The slab outgrew itself: either no ceiling was declared and M got bigger,
        # or the ceiling was raised after this slab was built. The old slab cannot be
        # freed (bump allocator) and existing cache entries keep viewing it, which
        # stays correct -- but it is wasted heap.
        print(
            f"⚠ fused GEMM+RS: grew the (N={N}, dtype={dtype}) staging slab from "
            f"{slab.shape[0]} to {rows} rows (M_max="
            f"{_MAX_M or 'undeclared'}). The old slab cannot be reclaimed; declare "
            f"the ceiling once, before warmup, to avoid paying for both."
        )
    # Before the slab is used for anything, and before it is cached: a divergent
    # offset must not survive to become wrong numbers.
    _check_slab_offset_agreement(new_slab, N, dtype, world_size)
    _SLABS[key] = new_slab
    return new_slab


def _create_entry(M: int, N: int, dtype: torch.dtype, world_size: int):
    """Build the cache entry for (M, N, dtype) as a view of the (N, dtype) slab.

    Collective only when the slab does not exist yet; after that this is a reshape
    and a dict store. Never legal during graph capture regardless, because the RS
    kernel is JIT-compiled per specialization and that is not capture-safe either.
    """
    global _SYNC_TENSOR

    _refuse_if_capturing(M, N, dtype)

    iris_inst = _get_iris()
    device = iris_inst.get_device()

    _check_divisible(M, world_size)
    M_local = M // world_size

    _cache_num_sms(device)

    # Only the staging buffer has to be on the symmetric heap: the RS kernel
    # translates addresses as `peer_base + (ptr - my_base)`, so it can only pull from
    # a heap pointer. The output is an ordinary torch tensor, allocated fresh per
    # call.
    #
    # `slab[:M]` is exactly (M, N) and contiguous. The shape must be exact: the GEMM
    # writes it with `out=`, and `out=` on a mismatched shape RESIZES -- which would
    # silently replace the heap buffer with a fresh off-heap allocation on the hot
    # path and make every peer read garbage without raising.
    staging = _get_slab(N, dtype, world_size, M)[:M]
    assert tuple(staging.shape) == (M, N), (
        f"staging view is {tuple(staging.shape)}, expected {(M, N)}"
    )

    if _SYNC_TENSOR is None:
        _SYNC_TENSOR = torch.zeros(1, dtype=torch.int32, device=device)

    entry = (staging, _SYNC_TENSOR, M_local, device) + _resolve_config(
        world_size, M_local
    )
    _BUFFER_CACHE[(M, N, dtype, world_size)] = entry
    return entry


def _entry_on_miss(M: int, N: int, dtype, world_size: int, group_name: str, op: str):
    """Fill the buffer cache from the hot path, LOUDLY. Not on the fast path.

    Task 3 Step 3 of the iris-heap plan: it is not possible to enumerate every M the
    scheduler can produce -- M is any multiple of the TP size up to
    ``max_num_batched_tokens`` -- so a miss is expected and must be *deliberate and
    logged* rather than silent. Since the slab already covers every M up to the
    declared ceiling, what a miss now costs is a Triton JIT for that specialization,
    not a symmetric-heap allocation. One line per distinct shape (the second call
    hits the cache), so this cannot flood a log.
    """
    print(
        f"ℹ fused GEMM+RS: buffer-cache miss for (M={M}, N={N}, dtype={dtype}, "
        f"world_size={world_size}) on the {op} path -- a shape warmup did not cover. "
        f"Filling it now: a view of the pre-allocated staging slab plus a Triton JIT "
        f"for this specialization. No symmetric-heap allocation, provided "
        f"M <= M_max={_MAX_M or 'undeclared'}. Legal here because this call is not "
        f"inside a graph capture region."
    )
    _check_iris_world_size(world_size, group_name)
    return _create_entry(M, N, dtype, world_size)


def _cache_num_sms(device) -> None:
    global _CACHED_NUM_SMS
    if _CACHED_NUM_SMS is None:
        _CACHED_NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count


def _cross_rank_sync(sync_tensor: torch.Tensor) -> None:
    """Stream-level cross-rank sync. Dispatches on ``AITER_GEMM_RS_SYNC_IMPL``.

    Neither implementation synchronizes the host, so both can be recorded into a
    graph; both only order this rank's stream against its peers'. Both run over
    the *default* process group / the full iris communicator -- ``group_name`` is
    deliberately not used here, since the ranks that must be ordered are exactly
    the ranks the RS kernel pulls from, and ``_check_iris_world_size`` guarantees
    the two agree.

    Both have a first-call side effect that is illegal inside a capture region --
    RCCL communicator creation, or the collective heap allocation of the iris
    barrier flags plus a Triton JIT. ``warmup_buffers()`` triggers BOTH,
    regardless of which is selected, so that flipping either environment
    variable can never move a collective into a capture region.

    The branch is on a module-level bool resolved at import, so the hot path
    costs one predictable test and no environment lookup.
    """
    if _SYNC_IMPL == "device":
        _device_barrier()
    else:
        dist.all_reduce(sync_tensor)


def _device_barrier() -> None:
    """iris device-side barrier: same ordering as the RCCL token, no RCCL launch.

    ``ctx.device_barrier()`` launches a single-CTA Triton kernel that bumps this
    rank's flag on the symmetric heap with an ``atomic_add`` (release, system
    scope) and then spin-polls every peer's flag for the same epoch with an
    ``atomic_cas`` (acquire, system scope). A rank therefore leaves the barrier
    only once every peer has *entered* it, which is the same guarantee the
    1-element ``dist.all_reduce`` provides and is what both guards rely on:

    - as the RAW guard, leaving it means every peer's GEMM kernel has completed
      (their barrier launch is stream-ordered after it) and its writes to the
      staging buffer are visible at system scope;
    - as the WAR guard, leaving it means every peer's RS kernel has completed,
      so the staging buffer can be overwritten.

    The epoch counter lives on the device, one int32 per rank, and is advanced by
    the kernel itself -- there is no host-side state -- which is what makes it
    replay-safe under graph capture rather than frozen at capture time.
    """
    _get_iris().device_barrier()


def register_shape(M: int, N: int, dtype: torch.dtype = torch.bfloat16) -> None:
    """Record an (M, N, dtype) that a later ``warmup_buffers()`` should pre-allocate.

    Lets a caller that discovers shapes incrementally (e.g. while walking an FX
    graph) hand them to an argument-less ``warmup_buffers()``.
    """
    _PENDING_SHAPES.add((M, N, dtype))


def _warm_one(M: int, N: int, dtype: torch.dtype, world_size: int) -> None:
    """Run one COMPLETE dummy iteration for (M, N, dtype).

    Allocating the staging buffer is not enough. Three first-call side effects
    would otherwise fire inside the graph capture region:

    1. the first ``dist.all_reduce`` creates the RCCL communicator lazily, which
       cannot happen mid-capture;
    2. the RS Triton kernel is JIT-compiled per distinct constexpr
       specialization (block_m, block_n, num_sms, world_size, M_ALIGNED,
       N_ALIGNED, dtype), so every shape needs its own real launch;
    3. the symmetric-heap allocation itself is collective.

    So this issues the sync *and* launches the kernel, on the same code path the
    hot path uses. The sync is issued unconditionally, even under
    ``AITER_GEMM_RS_SYNC=none``, so that flipping the variable never moves RCCL
    initialization into a capture region.

    ``_warm_both_sync_impls`` extends the same argument across
    ``AITER_GEMM_RS_SYNC_IMPL``: whichever primitive is selected, both are
    primed, so flipping *that* variable cannot move a collective into a capture
    region either.
    """
    entry = _BUFFER_CACHE.get((M, N, dtype, world_size))
    if entry is None:
        entry = _create_entry(M, N, dtype, world_size)
    staging, sync_tensor, M_local, device, block_m, block_n, num_sms, num_warps = entry

    # hipBLASLt creates its handle and a per-problem workspace on the first matmul
    # of a given (M, K, N, dtype), so the GEMM must be warmed on the same shape too.
    # K is unknown here; use N, which at least forces handle creation.
    dummy_a = torch.empty((M, N), dtype=dtype, device=device)
    dummy_b = torch.empty((N, N), dtype=dtype, device=device)
    torch.mm(dummy_a, dummy_b, out=staging)
    del dummy_a, dummy_b

    _warm_both_sync_impls(sync_tensor)
    dummy_out = torch.empty((M_local, N), dtype=dtype, device=device)
    fast_reduce_scatter(
        _get_iris(),
        dummy_out,
        staging,
        block_m,
        block_n,
        num_sms,
        num_warps,
    )
    _warm_both_sync_impls(sync_tensor)


def _warm_both_sync_impls(sync_tensor: torch.Tensor) -> None:
    """Trigger the first-call side effects of BOTH sync primitives. Collective.

    Each primitive has a first-call side effect that cannot happen inside a
    graph capture region: ``dist.all_reduce`` creates the RCCL communicator
    lazily, and ``device_barrier`` allocates its per-group flags tensor on the
    symmetric heap (collective -- ``ctx.zeros`` bottoms out in a host barrier)
    and JIT-compiles the barrier kernel.

    Both are primed here whichever one ``AITER_GEMM_RS_SYNC_IMPL`` selects, for
    the same reason ``_warm_one`` issues a sync even under
    ``AITER_GEMM_RS_SYNC=none``: an environment variable should never be able to
    move a collective into a capture region. The flags tensor is
    ``num_ranks`` int32s -- 32 bytes at TP=8 -- so priming the unused one costs
    nothing that matters against a symmetric heap already under pressure.
    """
    dist.all_reduce(sync_tensor)
    _device_barrier()


def _check_sync_mode_agreement() -> None:
    """Verify every rank read the same AITER_GEMM_RS_SYNC and ..._SYNC_IMPL.

    The mode decides how many guards each op issues and the impl decides which
    primitive a guard is, so a rank that disagrees on *either* hangs the job with
    no diagnostic -- easy to hit when a launcher exports the variables on some
    nodes only. A mode mismatch changes the collective count; an impl mismatch is
    nastier still, one rank blocking in RCCL while another spin-polls heap flags
    that will never be bumped. Checked once at warmup, where a collective is
    free.

    Both values are checked in a single all-reduce. Each is encoded as the pair
    ``(code, -code)`` and reduced with MAX, so any disagreement moves at least
    one of the two elements away from this rank's own value -- MAX alone could
    not distinguish "everyone agrees on 2" from "someone else said 2".
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    mode_codes = {"both": 0, "raw": 1, "none": 2}
    impl_codes = {"rccl": 0, "device": 1}
    mine_mode = mode_codes[_SYNC_MODE]
    mine_impl = impl_codes[_SYNC_IMPL]
    mine = [mine_mode, -mine_mode, mine_impl, -mine_impl]
    t = torch.tensor(mine, dtype=torch.int64, device=_get_iris().get_device())
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    got = [int(v) for v in t]
    if got[:2] != mine[:2]:
        raise RuntimeError(
            f"AITER_GEMM_RS_SYNC disagrees across ranks (this rank: {_SYNC_MODE!r}). "
            f"The mode controls how many collectives each fused GEMM+RS issues, so a "
            f"mismatch deadlocks. Export the same value on every rank."
        )
    if got[2:] != mine[2:]:
        raise RuntimeError(
            f"AITER_GEMM_RS_SYNC_IMPL disagrees across ranks (this rank: "
            f"{_SYNC_IMPL!r}). The impl controls which primitive each guard uses, so "
            f"a mismatch deadlocks -- ranks on 'rccl' block in a collective the ranks "
            f"on 'device' never join, and those spin on heap flags nobody bumps. "
            f"Export the same value on every rank."
        )


class HeapTooSmallError(RuntimeError):
    """The symmetric heap cannot hold the staging this deployment needs.

    A distinct type, not a bare RuntimeError, because callers must be able to treat
    it differently from every other warmup failure. vLLM's pass manager wraps warmup
    in a broad ``except Exception`` so that a broken iris degrades instead of killing
    the server -- which is right for an import error, and exactly wrong for this:
    by the time warmup runs, the fusion pass has already put the fused op in the
    graph, so degrading is not available. Swallowing this would turn a diagnosable
    startup refusal back into the mid-request OOM the refusal exists to prevent.
    """


def _check_heap_capacity(max_m: int, n_values, dtypes, world_size: int) -> int:
    """Refuse at warmup if the heap cannot hold the staging slabs. Returns the need.

    Task 2 Step 3: fail loudly and early. A server that starts and dies on request
    three costs a node-hour and a day of confusion; a refusal at warmup naming the
    four numbers costs nothing and diagnoses itself. This is also what catches an
    adopted sibling heap (see ``init_iris``), which was sized by a different fusion.
    """
    need = heap_bytes_required(max_m, n_values, dtypes, world_size)
    total = _heap_total()
    used = _heap_offset()
    if total is None or used is None:
        # A non-bump allocator has no offset to read. Do not invent a refusal.
        print(
            f"⚠ fused GEMM+RS: cannot read the heap's size/offset, so the capacity "
            f"check is skipped. Needed {need} B for M_max={max_m}, "
            f"N={sorted(set(n_values))}, dtypes={[str(d) for d in dtypes]}."
        )
        return need
    free = total - used
    if need > free:
        raise HeapTooSmallError(
            f"fused GEMM+RS: the iris symmetric heap is too small. Staging needs "
            f"{need} B ({need / (1 << 20):.1f} MiB) for M_max={max_m}, "
            f"N={sorted(set(n_values))}, dtypes={[str(d) for d in dtypes]}, "
            f"{_STAGING_BUFFERS_PER_SHAPE} buffer(s)/shape at world_size={world_size}; "
            f"the heap is {total} B ({total / (1 << 20):.1f} MiB) with {free} B "
            f"({free / (1 << 20):.1f} MiB) free. Refusing to start a server that "
            f"would die mid-request. Fix by deriving the heap from the same numbers "
            f"(heap_bytes_required(...) into init_iris(heap_size=...)), by setting "
            f"{_HEAP_BYTES_ENV}={need}, or by lowering the token ceiling that "
            f"produced M_max={max_m}."
        )
    return need


def warmup_buffers(
    compile_sizes=None,
    weight_shapes=None,
    dtype=None,
    group_name: str = "",
    max_m: int | None = None,
):
    """Pre-allocate buffers and run a full dummy iteration. Call collectively, pre-capture.

    Every first-call side effect these ops have happens here: the collective
    symmetric-heap allocation, lazy RCCL communicator creation, and the Triton
    JIT compile of each RS specialization. Anything not warmed here happens on
    its first forward, which is fatal inside a graph capture region.

    Args:
        compile_sizes: M dimensions that will be captured. Combined with
            ``weight_shapes`` to form the (M, N) grid to warm.
        weight_shapes: (K_local, N) weight shapes; only N is used.
        dtype: A dtype, an iterable of dtypes, or None. **None warms both
            bfloat16 and float16**, because the buffer cache is keyed on
            ``x.dtype``: defaulting to bfloat16 alone made an fp16 deployment
            miss on every shape and allocate inside the capture region. Pass an
            explicit dtype to halve the heap footprint.
        group_name: TP process group name, used to derive the world size. Empty
            falls back to the iris communicator.
        max_m: The largest M the scheduler can ever hand these ops -- for vLLM,
            ``max_num_batched_tokens`` rounded up to a multiple of the TP size. The
            staging slabs are sized for it, so **every** M up to it is covered, not
            just the warmed ones, and the heap the caller derived from
            ``heap_bytes_required(max_m, ...)`` is exactly the heap the slabs need.
            None means "size the slabs to the largest warmed M", which is right for
            fixed-shape callers (op_tests, the evidence scripts) and wrong for a
            scheduler: an M above the largest warmed one is then an allocation.

    Shape coverage, and what happens on a miss
    ------------------------------------------
    The warmed set cannot be the set of shapes the scheduler produces -- there are
    ``max_num_batched_tokens / tp`` of them and one buffer each would need tens of
    GiB. What the warmed set buys is the *Triton JIT*: every warmed (M, N, dtype)
    compiles its specialization here, and every shape vLLM captures into a graph is
    warmed, so no captured path can miss.

    An unwarmed M is therefore expected on the eager prefill path, and is handled,
    not prevented: ``_entry_on_miss`` takes a view of the already-allocated slab and
    JIT-compiles that specialization, logging one line. It allocates nothing on the
    symmetric heap, so it cannot exhaust it. Above ``max_m`` there is no slab to view
    and the op raises with the numbers rather than allocating.

    No-argument behaviour
    ---------------------
    vLLM's pass_manager calls this with no arguments, before any pass has run,
    so ``_PENDING_SHAPES`` may legitimately be empty. Rather than reduce to a
    print (the old behaviour) or raise -- which would hard-fail a deployment
    through a call site this repo does not own -- it warms a tiny documented
    probe shape, ``_DEFAULT_WARMUP_PROBE`` scaled by world_size, for each default
    dtype. Be clear about what that buys: it removes hazards (1) and (3) above
    for good, but it warms only *one* Triton specialization, whose block_m is
    chosen for the probe's M_local and generally will not match a real shape. It
    is a floor, not a substitute. Callers that know their shapes must pass
    ``compile_sizes``/``weight_shapes`` or call ``register_shape()`` first; a
    loud warning says so.
    """
    global _WARMUP_DONE

    iris_inst = _get_iris()
    device = iris_inst.get_device()
    _cache_num_sms(device)

    world_size = _world_size_from_group(group_name) if group_name else None
    if world_size is None:
        world_size = iris_inst.get_num_ranks()
    _check_iris_world_size(world_size, group_name)

    _check_sync_mode_agreement()

    if dtype is None:
        dtypes = tuple(_DEFAULT_WARMUP_DTYPES)
    elif isinstance(dtype, torch.dtype):
        dtypes = (dtype,)
    else:
        dtypes = tuple(dtype)

    shapes = set(_PENDING_SHAPES)
    if compile_sizes and weight_shapes:
        n_values = sorted({n for _, n in weight_shapes})
        shapes.update(
            (M, N, d) for M in compile_sizes for N in n_values for d in dtypes
        )

    if not shapes:
        probe_m_local, probe_n = _DEFAULT_WARMUP_PROBE
        shapes.update((probe_m_local * world_size, probe_n, d) for d in dtypes)
        print(
            "⚠ warmup_buffers() was given no shapes (no compile_sizes/weight_shapes "
            "and no register_shape() calls). Warming only the "
            f"{_DEFAULT_WARMUP_PROBE} probe shape, which initializes RCCL and one "
            "Triton specialization but pre-allocates none of your real shapes. The "
            "first forward for each real (M, N, dtype) will still perform a "
            "COLLECTIVE symmetric-heap allocation and a Triton JIT compile, both "
            "illegal inside a HIP graph capture region. Pass compile_sizes and "
            "weight_shapes, or call register_shape() before warmup."
        )

    # Sizing and coverage from ONE number. Everything below -- the capacity refusal,
    # the slab rows, and therefore which M can be served without allocating -- is
    # derived from `slab_m`, so the two cannot drift apart the way "warm the capture
    # sizes" and "hardcode 1 GiB" did.
    usable = [(M, N, d) for (M, N, d) in shapes if M % world_size == 0]
    slab_m = max([m for m, _, _ in usable] + [int(max_m or 0)])
    # Declared unconditionally, not only when the caller passed max_m: without it the
    # warm loop would size the first slab to the smallest M and then grow it once per
    # shape, allocating (and stranding) 48 slabs on the never-freeing heap.
    declare_max_m(slab_m)
    n_values = sorted({n for _, n, _ in usable}) or [_DEFAULT_WARMUP_PROBE[1]]

    # Before anything is allocated, so a refusal costs nothing and names the numbers.
    need = _check_heap_capacity(slab_m, n_values, dtypes, world_size)

    warmed = 0
    skipped = []
    for M, N, shape_dtype in sorted(shapes, key=lambda s: (s[0], s[1], str(s[2]))):
        if M % world_size != 0:
            skipped.append((M, N))
            continue
        already = (M, N, shape_dtype, world_size) in _BUFFER_CACHE
        # Run the dummy iteration even on a cache hit: the entry may have been
        # created by a cache miss on the hot path, which never JIT-warms the
        # other specializations, and re-running is cheap and idempotent.
        _warm_one(M, N, shape_dtype, world_size)
        warmed += 0 if already else 1

    if _TRACE_ALLOC:
        # The VA map, so a later HSA fault address can be tested for membership.
        # If the faulting address lies outside every range printed here, the
        # fault did not come from this op's heap traffic.
        try:
            alloc = iris_inst.heap.allocator
            base = int(alloc.get_base_address())
            _trace(
                f"HEAPMAP local_base=0x{base:x} size=0x{int(alloc.heap_size):x} "
                f"end=0x{base + int(alloc.heap_size):x} "
                f"used={_heap_offset()} peers="
                + ",".join(f"0x{int(v):x}" for v in alloc.heap_bases_array)
            )
        except Exception as e:  # pragma: no cover - diagnostics must never fail
            _trace(f"HEAPMAP unavailable: {type(e).__name__}: {e}")
        _trace(f"WARMUP COMPLETE warmed={warmed} cached={len(_BUFFER_CACHE)}")

    # The sync config is printed here, not just documented, because every perf
    # number this op produces is only interpretable against it and job logs are
    # the only durable record of what actually ran.
    print(
        f"✓ Iris GEMM+RS warmup complete: num_sms={_CACHED_NUM_SMS}, "
        f"world_size={world_size}, dtypes={[str(d) for d in dtypes]}, "
        f"sync={_SYNC_MODE}, sync_impl={_SYNC_IMPL}, "
        f"pre-allocated {warmed} buffer(s), {len(_BUFFER_CACHE)} cached"
        + (f", skipped {skipped} (M not divisible by {world_size})" if skipped else "")
    )

    # The heap budget, in the job log, every run. A fixed 1 GiB that nothing printed
    # is what turned this failure into a week of guessing; a line that states what
    # was needed, what was taken and what is left makes the next one self-diagnosing.
    total, used = _heap_total(), _heap_offset()
    slab_bytes = sum(int(s.numel()) * s.element_size() for s in _SLABS.values())
    after_31 = heap_bytes_required(
        slab_m, n_values, dtypes, world_size, buffers_per_shape=2
    )
    print(
        f"✓ Iris GEMM+RS heap: M_max={slab_m} N={n_values} "
        f"dtypes={[str(d) for d in dtypes]} buffers/shape={_STAGING_BUFFERS_PER_SHAPE} "
        f"-> required={need} B ({need / (1 << 20):.1f} MiB); "
        f"{len(_SLABS)} staging slab(s) hold {slab_bytes} B "
        f"({slab_bytes / (1 << 20):.1f} MiB); "
        f"heap total={total} used={used}"
        + (
            f" ({100.0 * used / total:.1f}%), free={(total - used) / (1 << 20):.1f} MiB"
            if total and used is not None
            else ""
        )
        + f"; after task 3.1 (2 buffers/shape) this would need {after_31} B "
        f"({after_31 / (1 << 20):.1f} MiB)"
    )
    _WARMUP_DONE = True


def _check_gemm_inputs(x: torch.Tensor, weights: list[torch.Tensor]) -> torch.Tensor:
    """Validate what this module skips by bypassing iris's ``_validate``.

    Calling ``fast_reduce_scatter`` directly instead of
    ``matmul_reduce_scatter_fast`` skips iris's ``_validate`` and
    ``_validate_staging_buffer`` entirely, so the checks have to live here. All
    of them are host-side integer/tuple comparisons: capture-safe, no device
    interaction.
    """
    if len(weights) != 1:
        raise ValueError(
            f"gemm_reduce_scatter_m_shard takes exactly one weight, got {len(weights)}."
        )
    weight = weights[0]
    if x.ndim != 2 or weight.ndim != 2:
        raise ValueError(
            f"x and weights[0] must be 2D, got shapes {tuple(x.shape)} and "
            f"{tuple(weight.shape)}."
        )
    if x.dtype != weight.dtype:
        raise ValueError(
            f"x dtype {x.dtype} != weights[0] dtype {weight.dtype}. Checked before "
            f"the staging buffer is allocated, because Iris never frees its heap."
        )
    if x.shape[1] != weight.shape[0]:
        raise ValueError(
            f"Incompatible matrix dimensions: x is {tuple(x.shape)} (M, K_local), "
            f"weights[0] is {tuple(weight.shape)}. The weight must be laid out "
            f"(K_local, N), NOT (N, K_local) -- vLLM holds it as (N, K_local), so "
            f"the caller must pass [weight.t()]. NOTE: this check cannot "
            f"catch a transposed weight when N == K_local -- the shapes are then "
            f"indistinguishable and the wrong product is computed silently."
        )
    return weight


def gemm_reduce_scatter_m_shard_wrapper(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    bias: torch.Tensor | None,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """
    Fused GEMM + reduce-scatter over rows, registered as
    ``torch.ops.aiter.gemm_reduce_scatter_m_shard``.

    Computes ``reduce_scatter_over_M(x @ weights[0] + bias)``: the GEMM writes
    this rank's full (M, N) partial into the cached symmetric-heap staging
    buffer, then the iris one-shot pull kernel has each rank reduce the M-shard
    it owns across all peers.

    Bias placement (the subtle part -- do not "simplify" this)
    ---------------------------------------------------------
    The bias is folded into the PRE-reduction partial via ``torch.addmm``, never
    added to the scattered output. ``RowParallelLinear`` passes a non-None bias
    on rank 0 only, precisely because the collective sums the per-rank partials:
    rank 0's ``+b`` is then summed into every output row exactly once, on every
    rank. Adding ``b`` after the reduce-scatter would instead apply it only to
    the row slice rank 0 happens to own, leaving every other rank's rows
    unbiased. ``torch.addmm`` broadcasts the (N,) bias across all M rows of the
    partial, which is exactly what is wanted here.

    Args:
        x: Local input (M, K_local) -- this rank's K-shard.
        weights: Single-element list holding the (K_local, N) weight. Note the
            (K, N) layout: vLLM's weight is (N, K_local), so the pass passes
            ``[weight.t()]``.
        bias: Optional (N,) bias, folded into the partial. Usually non-None on
            rank 0 only.
        scatter_dim: Must be the row dim (0 or -2).
        group_name: TP process group name. Both this path and the fake derive
            the world size from it; the real path additionally requires it to
            match the iris communicator.

    Returns:
        This rank's reduced shard, ``(M // world_size, N)``, freshly allocated on
        every call. The op is declared pure, so it must not alias across calls.
    """
    _check_scatter_dim(scatter_dim)
    weight = _check_gemm_inputs(x, weights)

    M = x.shape[0]
    N = weight.shape[1]
    world_size = _world_size_from_group(group_name)

    # --- hot path: dict lookups, output alloc, GEMM into the heap buffer, RS ---
    entry = _BUFFER_CACHE.get((M, N, x.dtype, world_size))
    if entry is None:
        # Capture-unsafe (Triton JIT); deliberate and logged outside capture.
        entry = _entry_on_miss(M, N, x.dtype, world_size, group_name, "fused")
    staging, sync_tensor, M_local, device, block_m, block_n, num_sms, num_warps = entry

    # Fresh, never cached: these ops are registered as pure, so a shared output
    # would make op(a) + op(b) evaluate to 2 * op(b). Capture-legal -- served
    # from the graph's private pool.
    output = torch.empty((M_local, N), dtype=x.dtype, device=device)

    if _TRACE_LAUNCH:
        _trace(
            f"FUSED M={M} N={N} K_local={x.shape[1]} dtype={x.dtype} "
            f"bias={bias is not None} staging_ptr=0x{staging.data_ptr():x} "
            f"out_ptr=0x{output.data_ptr():x}"
        )

    if bias is None:
        torch.mm(x, weight, out=staging)
    else:
        torch.addmm(bias, x, weight, out=staging)

    # RAW: every peer's GEMM must land before anyone pulls.
    if _SYNC_RAW:
        _cross_rank_sync(sync_tensor)

    fast_reduce_scatter(
        _get_iris(),
        output,
        staging,
        block_m,
        block_n,
        num_sms,
        num_warps,
    )

    # WAR: every peer's pull must finish before the staging buffer is reused.
    if _SYNC_WAR:
        _cross_rank_sync(sync_tensor)

    return output


def gemm_reduce_scatter_m_shard_fake(
    x: torch.Tensor,
    weights: list[torch.Tensor],
    bias: torch.Tensor | None,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """Fake implementation for meta mode / shape inference.

    Derives the world size the same way the real path does, from ``group_name``.
    Inductor sizes every downstream buffer from this, so any disagreement with
    the kernel is silent corruption.
    """
    _check_scatter_dim(scatter_dim)
    weight = _check_gemm_inputs(x, weights)
    M = x.shape[0]
    N = weight.shape[1]
    world_size = _world_size_from_group(group_name)
    _check_divisible(M, world_size)
    return torch.empty((M // world_size, N), dtype=x.dtype, device=x.device)


def reduce_scatter_m_shard_wrapper(
    x: torch.Tensor,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """
    Standalone reduce-scatter over rows, registered as
    ``torch.ops.aiter.reduce_scatter_m_shard``.

    Fallback for a partial sum that some other op already produced. ``x`` is an
    ordinary (M, N) torch tensor that is *not* on the symmetric heap, so it is
    first copied into the cached heap staging buffer -- the RS kernel translates
    addresses as ``peer_base + (ptr - my_base)`` and would read garbage from an
    off-heap pointer.

    That copy is the price of the fallback: roughly 5 us on a 2048x2880 fp16
    tensor, about 5% on top of the ~92 us reduce-scatter. Prefer the fused
    ``gemm_reduce_scatter_m_shard``, which has its GEMM write the heap buffer
    directly and pays nothing.

    Args:
        x: Partial sum (M, N), any ordinary torch tensor.
        scatter_dim: Must be the row dim (0 or -2).
        group_name: TP process group name. Both this path and the fake derive
            the world size from it; the real path additionally requires it to
            match the iris communicator.

    Returns:
        This rank's reduced shard, ``(M // world_size, N)``, freshly allocated on
        every call. The op is declared pure, so it must not alias across calls.
    """
    _check_scatter_dim(scatter_dim)
    if x.ndim != 2:
        raise ValueError(
            f"reduce_scatter_m_shard expects a 2D (M, N) tensor, got shape "
            f"{tuple(x.shape)}."
        )

    M, N = x.shape
    world_size = _world_size_from_group(group_name)

    # --- hot path: dict lookups, output alloc, copy into the heap buffer, RS ---
    entry = _BUFFER_CACHE.get((M, N, x.dtype, world_size))
    if entry is None:
        # Capture-unsafe (Triton JIT); deliberate and logged outside capture.
        entry = _entry_on_miss(M, N, x.dtype, world_size, group_name, "standalone")
    staging, sync_tensor, M_local, device, block_m, block_n, num_sms, num_warps = entry

    # Fresh, never cached -- see gemm_reduce_scatter_m_shard_wrapper.
    output = torch.empty((M_local, N), dtype=x.dtype, device=device)

    if _TRACE_LAUNCH:
        _trace(
            f"STANDALONE M={M} N={N} dtype={x.dtype} "
            f"staging_ptr=0x{staging.data_ptr():x} out_ptr=0x{output.data_ptr():x}"
        )

    staging.copy_(x)

    # RAW: every peer's copy must land before anyone pulls.
    if _SYNC_RAW:
        _cross_rank_sync(sync_tensor)

    fast_reduce_scatter(
        _get_iris(),
        output,
        staging,
        block_m,
        block_n,
        num_sms,
        num_warps,
    )

    # WAR: every peer's pull must finish before the staging buffer is reused.
    if _SYNC_WAR:
        _cross_rank_sync(sync_tensor)

    return output


def reduce_scatter_m_shard_fake(
    x: torch.Tensor,
    scatter_dim: int,
    group_name: str,
) -> torch.Tensor:
    """Fake implementation for meta mode / shape inference.

    Derives the world size the same way the real path does, from ``group_name``.
    """
    _check_scatter_dim(scatter_dim)
    if x.ndim != 2:
        raise ValueError(
            f"reduce_scatter_m_shard expects a 2D (M, N) tensor, got shape "
            f"{tuple(x.shape)}."
        )
    M, N = x.shape
    world_size = _world_size_from_group(group_name)
    _check_divisible(M, world_size)
    return torch.empty((M // world_size, N), dtype=x.dtype, device=x.device)


# Register the custom ops. Gated on iris being importable so that a missing or
# mismatched iris degrades to "these two ops are unavailable" rather than
# breaking the whole package at import time.
#
# Registration is all-or-nothing: the Library FRAGMENT is built into a local and
# only published to the module global once BOTH ops are in. If the second
# registration raises, the fragment is destroyed, which deregisters the first --
# otherwise the process would be left with a half-populated aiter namespace that
# a pattern-matching pass could still find and emit calls to.
if IRIS_AVAILABLE:
    _fused_ops_lib = None
    _lib = None
    try:
        from torch.library import Library

        from csrc.cpp_itfs.torch_utils import direct_register_custom_op

        _lib = Library("aiter", "FRAGMENT")

        # (Tensor x, Tensor[] weights, Tensor? bias, int scatter_dim, str group_name) -> Tensor
        direct_register_custom_op(
            op_name="gemm_reduce_scatter_m_shard",
            op_func=gemm_reduce_scatter_m_shard_wrapper,
            mutates_args=[],
            fake_impl=gemm_reduce_scatter_m_shard_fake,
            target_lib=_lib,
            dispatch_key="CUDA",
        )

        # (Tensor x, int scatter_dim, str group_name) -> Tensor
        direct_register_custom_op(
            op_name="reduce_scatter_m_shard",
            op_func=reduce_scatter_m_shard_wrapper,
            mutates_args=[],
            fake_impl=reduce_scatter_m_shard_fake,
            target_lib=_lib,
            dispatch_key="CUDA",
        )

        _fused_ops_lib = _lib
        print(
            "✓ Registered torch.ops.aiter.gemm_reduce_scatter_m_shard and "
            "torch.ops.aiter.reduce_scatter_m_shard custom ops"
        )
    except Exception as e:
        if _lib is not None:
            destroy = getattr(_lib, "_destroy", None)
            if destroy is not None:
                try:
                    destroy()
                except Exception:  # pragma: no cover - best-effort rollback
                    pass
            _lib = None
        print(
            f"⚠ Failed to register aiter reduce-scatter custom ops "
            f"(rolled back, neither op is registered): {e}"
        )
        import traceback

        traceback.print_exc()


# Public API
__all__ = [
    "gemm_reduce_scatter_m_shard_wrapper",
    "gemm_reduce_scatter_m_shard_fake",
    "reduce_scatter_m_shard_wrapper",
    "reduce_scatter_m_shard_fake",
    "init_iris",
    "heap_bytes_required",
    "declare_max_m",
    "HeapTooSmallError",
    "register_shape",
    "warmup_buffers",
]
