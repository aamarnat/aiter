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
``init_iris()`` / ``warmup_buffers()`` / a cache miss. Those must be called
collectively on all ranks *before* graph capture. The steady-state call is a
couple of dict lookups, an ordinary ``torch.empty`` for the output, a
``torch.mm``/``torch.addmm`` into the cached heap buffer, up to two 1-element
``dist.all_reduce`` sync tokens and the Triton RS launch -- no *collective*
allocation, no host sync, no device query.

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


def init_iris(heap_size: int = 1 << 30, ctx=None):
    """Pre-initialize (or adopt) the iris context for the reduce-scatter ops.

    This MUST be called collectively on ALL ranks before any of these ops
    execute (e.g. before graph capture). It allocates the iris symmetric heap,
    which requires a coordinated IPC handle exchange.

    ``iris.iris()`` is a plain factory, not a singleton, so building one context
    per fusion would give the process two symmetric heaps and two IPC handle
    exchanges. When the all-reduce sibling has already initialized a context,
    that one is adopted instead of creating a second.

    Args:
        heap_size: Symmetric heap size in bytes. Default 1GB. Ignored when a
            context is adopted or supplied.
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
        print(
            "✓ Iris context for fused GEMM+RS adopted from fused_gemm_all_reduce "
            "(one symmetric heap shared by both fusions)"
        )
        return IRIS_INSTANCE

    IRIS_INSTANCE = iris.iris(heap_size=heap_size)
    _publish_iris_instance(IRIS_INSTANCE)
    print(
        f"✓ Iris context initialized for fused GEMM+RS: "
        f"rank {IRIS_INSTANCE.get_rank()}/{IRIS_INSTANCE.get_num_ranks()}, "
        f"heap={heap_size / (1 << 20):.0f}MB"
    )
    return IRIS_INSTANCE


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

    Filling a cache entry is collective (``ctx.zeros`` bottoms out in a host
    ``dist.barrier()``), and a miss is rank-local, so a miss during capture either
    hangs on peers that are not capturing or diverges the collective count across
    ranks. Both present as an unexplained hang; this turns them into a message
    naming the shape that was not warmed.
    """
    try:
        capturing = torch.cuda.is_current_stream_capturing()
    except Exception:
        return
    if capturing:
        raise RuntimeError(
            f"fused GEMM+RS: cache miss for (M={M}, N={N}, dtype={dtype}) during "
            f"CUDA/HIP graph capture. Filling the cache is collective and would "
            f"deadlock. Warm this shape before capture via "
            f"warmup_buffers(compile_sizes=..., weight_shapes=...) or register_shape()."
        )


def _create_entry(M: int, N: int, dtype: torch.dtype, world_size: int):
    """Allocate the cache entry for (M, N, dtype). COLLECTIVE -- never on the hot path.

    ``ctx.zeros`` ends in a host ``dist.barrier()`` plus a heap-base all-gather,
    and Iris never frees, so this must run at warmup only: it is not legal
    during graph capture and it leaks ``M*N*itemsize`` per call if it is not
    hoisted out of the loop.
    """
    global _SYNC_TENSOR

    _refuse_if_capturing(M, N, dtype)

    iris_inst = _get_iris()
    device = iris_inst.get_device()

    _check_divisible(M, world_size)
    M_local = M // world_size

    _cache_num_sms(device)

    # Only the (M, N) staging buffer has to be on the symmetric heap: the RS
    # kernel translates addresses as `peer_base + (ptr - my_base)`, so it can
    # only pull from a heap pointer. The output is an ordinary torch tensor,
    # allocated fresh per call.
    if _TRACE_ALLOC:
        _off_before = _heap_offset()
        _t0 = time.perf_counter()
        _trace(
            f"ALLOC begin M={M} N={N} dtype={dtype} world={world_size} "
            f"heap_offset_before={_off_before}"
        )
        staging = iris_inst.zeros((M, N), dtype=dtype)
        _trace(
            f"ALLOC end   M={M} N={N} heap_offset_after={_heap_offset()} "
            f"wall_ms={(time.perf_counter() - _t0) * 1e3:.2f} "
            f"staging_ptr=0x{staging.data_ptr():x}"
        )
    else:
        staging = iris_inst.zeros((M, N), dtype=dtype)

    if _SYNC_TENSOR is None:
        _SYNC_TENSOR = torch.zeros(1, dtype=torch.int32, device=device)

    entry = (staging, _SYNC_TENSOR, M_local, device) + _resolve_config(
        world_size, M_local
    )
    _BUFFER_CACHE[(M, N, dtype, world_size)] = entry
    return entry


def _cache_num_sms(device) -> None:
    global _CACHED_NUM_SMS
    if _CACHED_NUM_SMS is None:
        _CACHED_NUM_SMS = torch.cuda.get_device_properties(device).multi_processor_count


def _cross_rank_sync(sync_tensor: torch.Tensor) -> None:
    """Stream-level cross-rank sync on a preallocated 1-element tensor.

    Unlike ``ctx.barrier()`` this does not synchronize the host, so it can be
    recorded into a graph; it only orders this rank's stream against its peers'.
    It runs on the *default* process group, which is the communicator iris's
    symmetric heap spans -- ``group_name`` is deliberately not used here, since
    the ranks that must be ordered are exactly the ranks the RS kernel pulls
    from. ``_check_iris_world_size`` guarantees the two agree.

    The FIRST such call lazily creates the RCCL communicator, which is illegal
    inside a graph capture region; ``warmup_buffers()`` issues one so that never
    happens during capture.
    """
    dist.all_reduce(sync_tensor)


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

    _cross_rank_sync(sync_tensor)
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
    _cross_rank_sync(sync_tensor)


def _check_sync_mode_agreement() -> None:
    """Verify every rank read the same AITER_GEMM_RS_SYNC value.

    The mode decides how many ``dist.all_reduce`` calls each op issues, so a rank
    that disagrees changes the collective count and the job hangs with no
    diagnostic -- easy to hit when a launcher exports the variable on some nodes
    only. Checked once at warmup, where a collective is free.
    """
    if not (dist.is_available() and dist.is_initialized()):
        return
    codes = {"both": 0, "raw": 1, "none": 2}
    mine = codes[_SYNC_MODE]
    t = torch.tensor([mine, -mine], dtype=torch.int64, device=_get_iris().get_device())
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    if int(t[0]) != mine or int(t[1]) != -mine:
        raise RuntimeError(
            f"AITER_GEMM_RS_SYNC disagrees across ranks (this rank: {_SYNC_MODE!r}). "
            f"The mode controls how many collectives each fused GEMM+RS issues, so a "
            f"mismatch deadlocks. Export the same value on every rank."
        )


def warmup_buffers(
    compile_sizes=None, weight_shapes=None, dtype=None, group_name: str = ""
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

    print(
        f"✓ Iris GEMM+RS warmup complete: num_sms={_CACHED_NUM_SMS}, "
        f"world_size={world_size}, dtypes={[str(d) for d in dtypes]}, "
        f"pre-allocated {warmed} buffer(s), {len(_BUFFER_CACHE)} cached"
        + (f", skipped {skipped} (M not divisible by {world_size})" if skipped else "")
    )


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
        _check_iris_world_size(world_size, group_name)
        # Collective; capture-unsafe (warmup only).
        entry = _create_entry(M, N, x.dtype, world_size)
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
        _check_iris_world_size(world_size, group_name)
        # Collective; capture-unsafe (warmup only).
        entry = _create_entry(M, N, x.dtype, world_size)
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
    "register_shape",
    "warmup_buffers",
]
