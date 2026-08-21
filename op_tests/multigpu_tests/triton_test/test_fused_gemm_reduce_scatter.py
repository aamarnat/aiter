#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Multi-GPU op test for the M-sharded reduce-scatter custom ops:

- ``torch.ops.aiter.gemm_reduce_scatter_m_shard(x, weights, bias, scatter_dim, group_name)``
- ``torch.ops.aiter.reduce_scatter_m_shard(x, scatter_dim, group_name)``

Both wrap ``iris.ops.matmul_reduce_scatter_fast.fast_reduce_scatter``; see
``aiter/ops/triton/comms/fused/fused_gemm_reduce_scatter.py``.

What is covered
---------------
1. ``test_gemm_reduce_scatter`` -- correctness + perf for three candidates against
   one untimed fp32 torch reference:
     * ``fused``      : ``gemm_reduce_scatter_m_shard``
     * ``standalone`` : GEMM into a caller buffer, then ``reduce_scatter_m_shard``
     * ``rccl``       : GEMM into a caller buffer, then ``dist.reduce_scatter_tensor``
   ``rccl`` doubles as an independent cross-check of the reduce-scatter semantics
   (which rows land on which rank) and as the baseline the fused op must beat.
2. ``test_graph_capture`` -- capture the op in a HIP graph, replay it, and assert
   the output is bitwise stable, still correct, and that the iris symmetric heap
   did not grow. An earlier revision of the iris path allocated per call and
   would have failed this.
3. ``test_unbarriered_drift`` -- >= 1000 back-to-back iterations with no host
   barrier and no host sync, which is the regime the headline speedup is measured
   in and which the upstream correctness run (barriered) does not validate.

Shapes are the GPT-OSS-120B MoE ``o_proj`` shapes: K=4096, N=2880, fp16, with
``M`` swept over decode/prefill sizes. Two extra shapes exist purely to hit the
kernel's tail masking -- see ``MNK_DEFAULT``.

Sharding convention (mirrors the op and vLLM's ``RowParallelLinear``):
  * ``x`` is ``(M, K_local)`` with ``K_local = K // world_size``
  * the weight is held as ``(N, K_local)`` and passed as ``[weight.t()]``, i.e. a
    NON-contiguous ``(K_local, N)`` view -- exactly what the vLLM pass passes
  * ``bias`` is non-None on rank 0 only, and must land in every output row of
    every rank exactly once

Usage:
    python test_fused_gemm_reduce_scatter.py                 # TP 2/4/8, all shapes
    python test_fused_gemm_reduce_scatter.py -n 8            # TP 8 only
    python test_fused_gemm_reduce_scatter.py -s 2048,2880,4096
"""

import argparse
import itertools
import logging
import multiprocessing as mp
import os
import sys
import traceback

import pandas as pd
import torch
import torch.distributed as dist

import aiter
from aiter import dtypes
from aiter.jit.utils.chip_info import get_gfx
from aiter.test_common import (
    benchmark,
    checkAllclose,
    ensure_spawn_method,
    run_perftest,
)

logger = logging.getLogger("aiter")

# The op module is imported for its side effect (it registers the two custom ops)
# and for the collective setup helpers init_iris/register_shape/warmup_buffers.
# It is imported at module scope on purpose: with the spawn start method every
# worker re-imports this file, so the ops are registered in every child too.
IRIS_AVAILABLE = False
FUSED_RS = None
try:
    from aiter.ops.triton.comms.fused import (
        fused_gemm_reduce_scatter as FUSED_RS,  # noqa: N812
    )

    IRIS_AVAILABLE = FUSED_RS.IRIS_AVAILABLE
except Exception as e:  # pragma: no cover - environment dependent
    print(f"Warning: could not import the fused GEMM+RS op module: {e}")

# NOTE on IrisCommContext: the sibling tests in this directory open an
# `IrisCommContext` because they call the raw `reduce_scatter`/`all_gather`
# helpers, which take a ctx argument. These two ops do not -- they own a
# process-global iris context created by `FUSED_RS.init_iris()`, which is the
# collective entry point vLLM's pass calls and therefore the one under test.
# `iris.iris()` is a plain constructor, not a singleton (iris/host/iris.py:1326),
# so additionally opening an IrisCommContext here would build a SECOND symmetric
# heap in the same process: double the heap memory and a second bump allocator
# to disambiguate in the "no heap growth" assertion below. So this test uses
# `init_iris()` only, and matches the sibling process-spawn pattern in every
# other respect (ensure_spawn_method / set_device / init_dist_env / mp.Pool /
# destroy_dist_env in a finally).

SUPPORTED_GFX = ["gfx942", "gfx950"]

# GPT-OSS-120B MoE o_proj, (M, N, K). K is the *full* reduction dim; each rank
# owns K_local = K // world_size of it.
#   32   -> decode
#   896  -> mid prefill; also ragged: M_local % block_m != 0 for TP 2/4/8
#   2048 -> prefill; M_local % block_m == 0 for TP 2/4/8 (the aligned fast path)
#   2016 -> dedicated M-tail case, ragged for TP 2/4/8 (see _tail_note)
#   2888 -> dedicated N-tail case: N % block_n(=64) == 8
MNK_DEFAULT = [
    (32, 2880, 4096),
    (896, 2880, 4096),
    (2048, 2880, 4096),
    (2016, 2880, 4096),
    (2048, 2888, 4096),
]

TP_DEFAULT = [2, 4, 8]

# Fixed iteration counts. run_perftest's default (num_rotate_args=0) derives the
# iteration count from free memory, which differs per rank and would deadlock a
# collective, so every timed loop here pins num_rotate_args=1.
PERF_ITERS = 50
PERF_WARMUP = 5
GRAPH_REPLAYS = 128  # >= 100 required
DRIFT_ITERS = 1024  # >= 1000 required, per phase

# checkAllclose returns the fraction of mismatching elements (1.0 if
# catastrophic). fp16 accumulation over a K=4096 reduction should be exact to
# well within rtol/atol, so anything above this is a real failure.
MAX_ERR_RATIO = 0.02
RTOL = 1e-2
ATOL = 1e-2


_PORT_LO, _PORT_HI = 20000, 32000


def _free_port(attempts=50):
    """A bindable TCP port BELOW the ephemeral range, for the rendezvous store.

    Replaces three hardcoded constants. Two distinct failures, both measured on
    radha f13-21, and the second is why this does not simply bind port 0:

    1. The constants gave each *test function* one port, but each function then
       sweeps TP=2, 4 and 8 and reused that one port across all three, so a later
       sweep step hit the previous store's socket lingering in TIME_WAIT:
       ``DistNetworkError ... port: 49380 ... EADDRINUSE``, before a single shape
       was checked.
    2. Asking the kernel for an ephemeral port instead (bind to 0, read it back,
       close, re-bind in the child) still failed:
       ``... port: 34607 ... EADDRINUSE``. Linux hands out 32768-60999 for
       *outgoing* connections too, and this test churns a lot of them -- RCCL and
       the store itself -- so between releasing the probe socket and the child
       binding it, something else can legitimately be given the same port. Both
       49380 and 34607 are inside that range; the fixed constants were only ever
       a worse instance of the same bug.

    So: pick below the ephemeral range, where the kernel will not assign the port
    to anything else on its own, and confirm it is bindable. ``SO_REUSEADDR`` is
    deliberately NOT set on the probe -- it would report a TIME_WAIT port as
    free, which is failure (1) again.
    """
    import random
    import socket

    for _ in range(attempts):
        port = random.randint(_PORT_LO, _PORT_HI)
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("", port))
            except OSError:
                continue
        return port
    raise RuntimeError(
        f"no free TCP port in [{_PORT_LO}, {_PORT_HI}] after {attempts} tries"
    )


# ---------------------------------------------------------------------------
# reference (untimed, never in the table)
# ---------------------------------------------------------------------------


def run_torch(xs, ws_nk, bias, rank, world_size, dtype):
    """Reference: sum every rank's partial, then take this rank's row slice.

    ``full = sum_r (x_r @ w_r) [+ bias]``; the result is ``full`` rows
    ``[rank*M_local : (rank+1)*M_local]``.

    Deliberately mirrors the kernel's arithmetic rather than being a pure fp32
    computation: each rank's partial is produced by a single ``mm``/``addmm`` in
    ``dtype`` (the op stages an (M, N) ``dtype`` partial on the symmetric heap),
    the cross-rank reduction accumulates in fp32 (the iris kernel uses
    ``acc_dtype = tl.float32``), and the cast back to ``dtype`` happens once at
    the end. ``addmm`` -- not ``mm`` then ``+`` -- because that is what the op
    does, and it rounds once instead of twice.

    Args:
        xs: Per-rank ``(M, K_local)`` inputs, indexed by rank.
        ws_nk: Per-rank ``(N, K_local)`` weights, indexed by rank (the layout
            vLLM stores; the op is handed ``.t()``).
        bias: Optional ``(N,)`` bias. Folded into rank 0's partial only, which is
            the RowParallelLinear convention: the collective then sums it into
            every output row of every rank exactly once.
        rank: Rank whose shard to return.
        world_size: Number of ranks.
        dtype: Output dtype.

    Returns:
        ``(M // world_size, N)`` reference shard for ``rank``.
    """
    full = None
    for r in range(world_size):
        w_t = ws_nk[r].t()
        if bias is not None and r == 0:
            partial = torch.addmm(bias, xs[r], w_t)
        else:
            partial = torch.mm(xs[r], w_t)
        partial = partial.to(torch.float32)
        full = partial if full is None else full + partial

    M_local = full.shape[0] // world_size
    return full[rank * M_local : (rank + 1) * M_local, :].to(dtype)


# ---------------------------------------------------------------------------
# shared worker-side helpers
# ---------------------------------------------------------------------------


def _build_rank_inputs(M, N, K_local, world_size, dtype, device, seed):
    """Build every rank's inputs identically, on every rank.

    Generated on the CPU from an explicit ``torch.Generator`` so that all ranks
    (and all processes) see byte-identical tensors: a per-device RNG would let a
    rank's reference silently disagree with its peers' actual inputs, which is
    the one failure mode this test cannot afford.

    ``x`` and the weight are scaled by ``K ** -0.25`` so the reduced output has
    roughly unit variance for every TP degree: the reduced value is a sum over
    ``K_local * world_size == K`` products of two N(0, s^2) variables, so its std
    is ``s^2 * sqrt(K)``, which is 1 at ``s = K ** -0.25``. Keeping the output
    O(1) keeps the fp16 quantization error well inside ATOL and, more
    importantly, keeps the bias-placement check below free of catastrophic
    cancellation.
    """
    K = K_local * world_size
    scale = K**-0.25
    gen = torch.Generator().manual_seed(seed)

    xs, ws_nk = [], []
    for _ in range(world_size):
        x = (torch.randn(M, K_local, generator=gen) * scale).to(dtype).to(device)
        w = (torch.randn(N, K_local, generator=gen) * scale).to(dtype).to(device)
        xs.append(x)
        ws_nk.append(w)

    # Comparable in magnitude to the reduced GEMM output (~1.0), so the
    # "bias landed in this row" check has real signal.
    bias = (torch.randn(N, generator=gen) * 0.5).to(dtype).to(device)
    return xs, ws_nk, bias


def _register_group_name() -> str:
    """Register the TP group with torch and return its name, as vLLM does.

    The two ops take a ``group_name`` from which BOTH the real and the fake
    implementation derive the world size (``_world_size_from_group`` ->
    ``_resolve_process_group``); the real path additionally requires it to match
    the iris communicator. Registering the group here means the meta path
    exercised by ``_check_fake_shapes`` is the same one Inductor will hit,
    instead of silently falling through to the ``IRIS_INSTANCE`` fallback.
    """
    try:
        from torch.distributed.distributed_c10d import _register_process_group

        from aiter.dist.parallel_state import get_tp_group

        tp = get_tp_group()
        _register_process_group(tp.unique_name, tp.device_group)
        return tp.unique_name
    except Exception as e:
        print(f"[warn] could not register the TP group with torch ({e}); using ''")
        return ""


def _check_fake_shapes(x, w_t, bias, group_name, world_size):
    """Assert both fake impls infer the right output shape.

    Cheap, but it is the only thing standing between a wrong ``world_size``
    inference and a shape mismatch that only appears deep inside Inductor.
    """
    from torch._subclasses.fake_tensor import FakeTensorMode

    M, N = x.shape[0], w_t.shape[1]
    expected = (M // world_size, N)
    with FakeTensorMode():
        fx = torch.empty(x.shape, dtype=x.dtype, device=x.device)
        fw = torch.empty(w_t.shape, dtype=w_t.dtype, device=w_t.device)
        fb = (
            None
            if bias is None
            else torch.empty(bias.shape, dtype=bias.dtype, device=bias.device)
        )
        fused_out = torch.ops.aiter.gemm_reduce_scatter_m_shard(
            fx, [fw], fb, 0, group_name
        )
        fp = torch.empty((M, N), dtype=x.dtype, device=x.device)
        rs_out = torch.ops.aiter.reduce_scatter_m_shard(fp, 0, group_name)

    assert tuple(fused_out.shape) == expected, (
        f"gemm_reduce_scatter_m_shard fake impl: got {tuple(fused_out.shape)}, want {expected}"
    )
    assert tuple(rs_out.shape) == expected, (
        f"reduce_scatter_m_shard fake impl: got {tuple(rs_out.shape)}, want {expected}"
    )


def _symmetric_heap_offset():
    """Bytes ever bump-allocated from the iris symmetric heap, or None.

    The default iris allocator (``TorchAllocator``) is a bump allocator over one
    pre-allocated pool and iris never frees, so ``heap_offset`` is an exact,
    monotonic byte count of every symmetric allocation made so far. Comparing it
    before and after a replay loop is a direct measurement of "did this op
    allocate on the heap", not a proxy. Returns None under a different allocator
    (e.g. vmem), in which case callers fall back to the torch allocator delta.
    """
    inst = getattr(FUSED_RS, "IRIS_INSTANCE", None)
    if inst is None:
        return None
    try:
        return int(inst.heap.allocator.heap_offset)
    except AttributeError:
        return None


def _cache_probe(M, N, dtype, world_size):
    """Report the resolved launch config and buffer identity for (M, N, dtype).

    Reaches into the op's private ``_BUFFER_CACHE`` on purpose: proving that a
    ragged shape really did hit the kernel's M/N tail path requires knowing the
    ``block_m``/``block_n`` the op resolved, and there is no public accessor.
    """
    entry = getattr(FUSED_RS, "_BUFFER_CACHE", {}).get((M, N, dtype, world_size))
    if entry is None:
        return {}
    # No output buffer is cached, by design: the ops are registered pure, so a
    # shared output would alias across calls and make `op(a) + op(b)` evaluate
    # to `2 * op(b)`. Only the staging buffer is cached and pointer-stable.
    staging, _sync, M_local, _device, block_m, block_n, num_sms, _num_warps = entry
    return {
        "block_m": block_m,
        "block_n": block_n,
        "num_sms": num_sms,
        "m_tail": bool(M_local % block_m),
        "n_tail": bool(N % block_n),
        "staging_ptr": staging.data_ptr(),
    }


def _setup_rank(tp_size, gpu_id, cfg):
    """Common per-worker bring-up: device, dist env, iris, warmed buffers, inputs.

    Returns everything the three workers need. Every allocation the ops perform
    happens here, in ``warmup_buffers()``, before any timed or captured region --
    which is exactly the contract the op documents.
    """
    device = torch.device(f"cuda:{gpu_id}")
    torch.cuda.set_device(device)
    aiter.init_dist_env(tp_size, gpu_id)

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    M, N, K, dtype = cfg["M"], cfg["N"], cfg["K"], cfg["dtype"]
    assert M % world_size == 0, f"M={M} must be divisible by world_size={world_size}"
    assert K % world_size == 0, f"K={K} must be divisible by world_size={world_size}"
    K_local = K // world_size

    torch.cuda.synchronize()
    dist.barrier()

    group_name = _register_group_name()
    xs, ws_nk, bias_full = _build_rank_inputs(
        M, N, K_local, world_size, dtype, device, cfg["seed"]
    )

    # RowParallelLinear passes a non-None bias on rank 0 only.
    use_bias = cfg["bias_mode"] == "rank0"
    bias_ref = bias_full if use_bias else None
    bias_arg = bias_full if (use_bias and rank == 0) else None

    x = xs[rank]
    # (N, K_local) contiguous -> .t() is the non-contiguous (K_local, N) view the
    # vLLM pass hands to the op. Do NOT "clean this up" into a contiguous tensor:
    # the transposed view is the layout the model actually runs.
    w_t = ws_nk[rank].t()

    ref = run_torch(xs, ws_nk, bias_ref, rank, world_size, dtype)
    ref_nobias = (
        run_torch(xs, ws_nk, None, rank, world_size, dtype) if use_bias else None
    )

    FUSED_RS.init_iris(heap_size=cfg["heap_size"])
    FUSED_RS.register_shape(M, N, dtype)
    FUSED_RS.warmup_buffers(dtype=dtype)

    _check_fake_shapes(x, w_t, bias_arg, group_name, world_size)

    return {
        "device": device,
        "rank": rank,
        "world_size": world_size,
        "M": M,
        "N": N,
        "K_local": K_local,
        "M_local": M // world_size,
        "dtype": dtype,
        "group_name": group_name,
        "x": x,
        "w_t": w_t,
        "bias_arg": bias_arg,
        "bias_full": bias_full if use_bias else None,
        "ref": ref,
        "ref_nobias": ref_nobias,
    }


def _assert_correct(env, out, name):
    """Assert this rank's shard, and (with bias) that bias landed in every row.

    Runs on EVERY rank, which is the whole point for the bias case: folding the
    bias after the reduce-scatter instead of before it produces a correct shard
    on rank 0 and an unbiased shard everywhere else, so a rank-0-only assertion
    would pass a broken kernel.
    """
    rank = env["rank"]
    ref32 = env["ref"].to(torch.float32)
    out32 = out.to(torch.float32)
    err = checkAllclose(
        ref32,
        out32,
        rtol=RTOL,
        atol=ATOL,
        msg=f"[rank {rank}] {name}: shard vs torch ref",
    )
    assert err <= MAX_ERR_RATIO, (
        f"[rank {rank}] {name}: shard mismatch, err ratio {err:.4f} > {MAX_ERR_RATIO}"
    )

    bias_err = 0.0
    if env["ref_nobias"] is not None:
        # out - ref_nobias must equal the bias, broadcast over EVERY row of this
        # rank's shard. Values are O(1) and the bias is O(0.5), so this
        # difference is not a cancellation trap.
        delta = out32 - env["ref_nobias"].to(torch.float32)
        want = env["bias_full"].to(torch.float32).unsqueeze(0).expand_as(delta)
        bias_err = checkAllclose(
            want,
            delta,
            rtol=RTOL,
            atol=ATOL,
            msg=f"[rank {rank}] {name}: bias present exactly once in every row",
        )
        assert bias_err <= MAX_ERR_RATIO, (
            f"[rank {rank}] {name}: bias is not in every row of this rank's shard "
            f"(err ratio {bias_err:.4f}). A post-reduce-scatter bias add would "
            f"look like this on every rank but rank 0."
        )
    return err, bias_err


def _log_worker_error(gpu_id, e):
    logger.error(
        f"\n-->[Error on GPU {gpu_id}]: {e!s}\n"
        f"-->[Traceback]: {''.join(traceback.format_exception(*sys.exc_info()))}"
    )


# ---------------------------------------------------------------------------
# workers (must stay at module scope: the spawn start method pickles by name)
# ---------------------------------------------------------------------------


def _worker_correctness(tp_size, gpu_id, cfg):
    """Time and check fused / standalone / rccl on one rank."""
    try:
        env = _setup_rank(tp_size, gpu_id, cfg)
        device, M, N = env["device"], env["M"], env["N"]
        dtype, group_name = env["dtype"], env["group_name"]
        x, w_t, bias_arg = env["x"], env["w_t"], env["bias_arg"]

        # The unfused paths own their (M, N) partial buffer, as a caller would.
        partial = torch.empty((M, N), dtype=dtype, device=device)
        rccl_out = torch.empty((env["M_local"], N), dtype=dtype, device=device)

        def _gemm_into_partial():
            if bias_arg is None:
                torch.mm(x, w_t, out=partial)
            else:
                torch.addmm(bias_arg, x, w_t, out=partial)

        def _fused():
            return torch.ops.aiter.gemm_reduce_scatter_m_shard(
                x, [w_t], bias_arg, 0, group_name
            )

        def _standalone():
            _gemm_into_partial()
            return torch.ops.aiter.reduce_scatter_m_shard(partial, 0, group_name)

        def _rccl():
            _gemm_into_partial()
            dist.reduce_scatter_tensor(rccl_out, partial)
            return rccl_out

        res = {"rank": env["rank"]}
        for name, fn in (
            ("fused", _fused),
            ("standalone", _standalone),
            ("rccl", _rccl),
        ):
            torch.cuda.synchronize()
            dist.barrier()
            out, us = run_perftest(
                fn,
                num_iters=PERF_ITERS,
                num_warmup=PERF_WARMUP,
                num_rotate_args=1,
            )
            # Both ops return a cached buffer that the next candidate reuses.
            out = out.clone()
            err, bias_err = _assert_correct(env, out, name)
            res[f"{name}_us"] = us
            res[f"{name}_err"] = err
            res[f"{name}_bias_err"] = bias_err

        res.update(_cache_probe(M, N, dtype, tp_size))
        torch.cuda.synchronize()
        return res

    except Exception as e:
        _log_worker_error(gpu_id, e)
        raise
    finally:
        aiter.destroy_dist_env()


def _worker_graph(tp_size, gpu_id, cfg):
    """Capture the fused op in a HIP graph, replay it, and check for heap growth."""
    try:
        env = _setup_rank(tp_size, gpu_id, cfg)
        M, N, dtype = env["M"], env["N"], env["dtype"]
        x, w_t, bias_arg = env["x"], env["w_t"], env["bias_arg"]
        group_name = env["group_name"]

        def _fused():
            return torch.ops.aiter.gemm_reduce_scatter_m_shard(
                x, [w_t], bias_arg, 0, group_name
            )

        # Eager warmup: triton JIT compile and NCCL communicator creation must
        # both happen before capture, or capture records a cudaMalloc.
        eager_out, eager_us = run_perftest(
            _fused, num_iters=PERF_ITERS, num_warmup=PERF_WARMUP, num_rotate_args=1
        )
        eager_err, eager_bias_err = _assert_correct(env, eager_out.clone(), "eager")

        torch.cuda.synchronize()
        dist.barrier()

        probe_before = _cache_probe(M, N, dtype, tp_size)
        assert probe_before, (
            f"[rank {env['rank']}] warmup_buffers() left ({M}, {N}, {dtype}) "
            f"uncached, so the first captured call would allocate on the "
            f"symmetric heap"
        )
        heap_before = _symmetric_heap_offset()
        cache_len_before = len(getattr(FUSED_RS, "_BUFFER_CACHE", {}))

        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_out = _fused()

        torch.cuda.synchronize()
        dist.barrier()
        heap_after_capture = _symmetric_heap_offset()

        # Stream capture does not execute the captured kernels, so graph_out
        # still holds the eager result here; replay once to get a real baseline.
        # The stability probe tensors are allocated before mem_before so the
        # baseline already includes them and a flat delta means "the replay loop
        # itself allocated nothing".
        graph.replay()
        torch.cuda.synchronize()
        first = graph_out.clone()
        max_drift = torch.zeros((), dtype=torch.float32, device=env["device"])
        d = torch.zeros((), dtype=torch.float32, device=env["device"])

        torch.cuda.synchronize()
        dist.barrier()
        mem_before = torch.cuda.memory_allocated(env["device"])

        for _ in range(cfg["replays"]):
            graph.replay()
            # Device-side, so the replay loop keeps running without a host sync.
            d = (graph_out - first).abs().amax().to(torch.float32)
            max_drift = torch.maximum(max_drift, d)

        torch.cuda.synchronize()
        mem_after = torch.cuda.memory_allocated(env["device"])
        heap_after = _symmetric_heap_offset()
        probe_after = _cache_probe(M, N, dtype, tp_size)
        drift = max_drift.item()

        # Replaying a graph must be bit-identical: same inputs, same buffers.
        assert drift == 0.0, (
            f"[rank {env['rank']}] graph replay is not stable: "
            f"max |replay_i - replay_0| = {drift}"
        )
        graph_err, graph_bias_err = _assert_correct(env, graph_out.clone(), "hipgraph")

        heap_growth = None
        if heap_before is not None:
            heap_growth = heap_after - heap_before
            assert heap_growth == 0, (
                f"[rank {env['rank']}] iris symmetric heap grew by {heap_growth} B "
                f"across capture + {cfg['replays']} replays "
                f"(capture alone: {heap_after_capture - heap_before} B). "
                f"Every symmetric allocation must happen in warmup_buffers()."
            )
        else:
            logger.warning(
                "iris heap offset probe unavailable (non-bump allocator); "
                "falling back to the buffer-cache and torch allocator checks"
            )
        assert len(getattr(FUSED_RS, "_BUFFER_CACHE", {})) == cache_len_before, (
            f"[rank {env['rank']}] the op's buffer cache gained entries during "
            f"capture/replay; a cache miss inside a graph is a collective "
            f"symmetric-heap allocation and is illegal there."
        )
        assert probe_after.get("staging_ptr") == probe_before.get("staging_ptr"), (
            f"[rank {env['rank']}] the staging buffer was reallocated during "
            f"capture/replay"
        )
        # There is deliberately no equivalent assertion for the output buffer:
        # it is allocated fresh on every call (the ops are declared pure, so a
        # cached output would alias across calls). Inside a capture region that
        # allocation is served from the graph's private pool, so it is legal and
        # its address is expected to differ from the pre-capture eager call's.

        _, graph_us = run_perftest(
            graph.replay,
            num_iters=PERF_ITERS,
            num_warmup=PERF_WARMUP,
            num_rotate_args=1,
        )

        res = {
            "rank": env["rank"],
            "eager_us": eager_us,
            "hipgraph_us": graph_us,
            "eager_err": eager_err,
            "hipgraph_err": graph_err,
            "bias_err": max(eager_bias_err, graph_bias_err),
            "replay_drift": drift,
            "heap_growth_B": heap_growth,
            "torch_mem_growth_B": mem_after - mem_before,
        }
        res.update(probe_after)
        torch.cuda.synchronize()
        return res

    except Exception as e:
        _log_worker_error(gpu_id, e)
        raise
    finally:
        aiter.destroy_dist_env()


def _worker_drift(tp_size, gpu_id, cfg):
    """Run the op unbarriered for >= 1000 iterations and check it never drifts.

    "Unbarriered" means no ``dist.barrier()`` and no ``torch.cuda.synchronize()``
    inside the loop: the ranks free-run and the only ordering is the op's own two
    in-stream sync tokens. Two phases:

      A. pure back-to-back calls, one host sync at the very end -- the true
         steady state, and the regime the reported speedup is measured in;
      B. the same loop with a device-side running ``max |out - ref|`` so a
         transient wrong result at iteration 700 cannot hide behind a correct
         final iteration. The extra reduction perturbs timing slightly, which is
         why phase A is timed and phase B is not.
    """
    try:
        env = _setup_rank(tp_size, gpu_id, cfg)
        M, N, dtype = env["M"], env["N"], env["dtype"]
        x, w_t, bias_arg = env["x"], env["w_t"], env["bias_arg"]
        group_name = env["group_name"]
        iters = cfg["iters"]

        def _fused():
            return torch.ops.aiter.gemm_reduce_scatter_m_shard(
                x, [w_t], bias_arg, 0, group_name
            )

        out, us = run_perftest(
            _fused, num_iters=PERF_ITERS, num_warmup=PERF_WARMUP, num_rotate_args=1
        )
        _assert_correct(env, out.clone(), "warmup")

        ref32 = env["ref"].to(torch.float32)
        heap_before = _symmetric_heap_offset()

        # --- phase A: no host barrier, no host sync, nothing between calls ---
        torch.cuda.synchronize()
        dist.barrier()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            out = _fused()
        end.record()
        torch.cuda.synchronize()
        pure_us = start.elapsed_time(end) * 1e3 / iters
        err_a, bias_err_a = _assert_correct(env, out.clone(), f"unbarriered x{iters}")

        # --- phase B: same regime, per-iteration device-side drift probe ---
        max_drift = torch.zeros((), dtype=torch.float32, device=env["device"])
        torch.cuda.synchronize()
        dist.barrier()
        for _ in range(iters):
            out = _fused()
            d = (out.to(torch.float32) - ref32).abs().amax()
            max_drift = torch.maximum(max_drift, d)
        torch.cuda.synchronize()
        drift = max_drift.item()

        err_b, _ = _assert_correct(env, out.clone(), f"unbarriered probed x{iters}")

        # Same bound checkAllclose enforces elementwise, applied to the worst
        # element of any of the `iters` iterations.
        drift_bound = ATOL + RTOL * ref32.abs().max().item()
        assert drift <= drift_bound, (
            f"[rank {env['rank']}] unbarriered drift: some iteration in "
            f"{iters} produced max |out - ref| = {drift:.5f} > {drift_bound:.5f}"
        )

        heap_growth = None
        if heap_before is not None:
            heap_growth = _symmetric_heap_offset() - heap_before
            assert heap_growth == 0, (
                f"[rank {env['rank']}] iris symmetric heap grew by {heap_growth} B "
                f"over {2 * iters} unbarriered iterations"
            )

        res = {
            "rank": env["rank"],
            "fused_us": us,
            "unbarriered_us": pure_us,
            "fused_err": max(err_a, err_b),
            "bias_err": bias_err_a,
            "max_drift": drift,
            "drift_bound": drift_bound,
            "heap_growth_B": heap_growth,
        }
        res.update(_cache_probe(M, N, dtype, tp_size))
        torch.cuda.synchronize()
        return res

    except Exception as e:
        _log_worker_error(gpu_id, e)
        raise
    finally:
        aiter.destroy_dist_env()


# ---------------------------------------------------------------------------
# parent-side plumbing
# ---------------------------------------------------------------------------


def _spawn(worker, tp_size, cfg):
    """Run ``worker`` on ``tp_size`` GPUs and return one result dict per rank.

    ``r.get()`` re-raises any worker exception in the parent, so a per-rank
    assertion failure fails the test rather than being swallowed.

    A fresh rendezvous port per pool -- see ``_free_port``. Each caller sweeps
    several TP sizes through here, and a port reused across them hits the
    previous store's socket in TIME_WAIT.
    """
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(_free_port())
    ensure_spawn_method()

    pool = mp.Pool(processes=tp_size)
    results = [pool.apply_async(worker, args=(tp_size, i, cfg)) for i in range(tp_size)]
    pool.close()
    pool.join()
    return [r.get() for r in results]


def _roofline(M, N, K_local, M_local, world_size, elem_size):
    """FLOPs and bytes moved per rank, per candidate.

    ``flops``: the local GEMM (2*M*K_local*N mul-add) plus the cross-rank adds
    the reduce-scatter performs on this rank's shard.
    ``nbytes``: read x, read w, write the (M, N) partial, pull every rank's copy
    of this rank's shard rows, write the shard. ``standalone`` additionally reads
    the caller's partial and writes it into the heap staging buffer -- the copy
    the op's docstring calls the price of the fallback.
    """
    gemm_flops = 2 * M * K_local * N
    rs_flops = M_local * N * (world_size - 1)
    flops = gemm_flops + rs_flops

    gemm_bytes = (M * K_local + K_local * N + M * N) * elem_size
    rs_bytes = (world_size * M_local * N + M_local * N) * elem_size
    copy_bytes = 2 * M * N * elem_size

    return flops, {
        "fused": gemm_bytes + rs_bytes,
        "standalone": gemm_bytes + copy_bytes + rs_bytes,
        "rccl": gemm_bytes + rs_bytes,
        "eager": gemm_bytes + rs_bytes,
        "hipgraph": gemm_bytes + rs_bytes,
        "unbarriered": gemm_bytes + rs_bytes,
    }


def _fold(rows, candidates, flops, nbytes):
    """Fold per-rank dicts into one table row, asserting on every rank."""
    ret = {"gfx": get_gfx(), "ranks_checked": len(rows)}

    failures = []
    for name in candidates:
        us = [r[f"{name}_us"] for r in rows]
        errs = [r[f"{name}_err"] for r in rows]
        avg_us = sum(us) / len(us)
        ret[f"{name} us"] = avg_us
        ret[f"{name} TFLOPS"] = flops / avg_us / 1e6
        ret[f"{name} TB/s"] = nbytes[name] / avg_us / 1e6
        ret[f"{name} err"] = max(errs)
        failures += [
            (name, r["rank"], e) for r, e in zip(rows, errs) if e > MAX_ERR_RATIO
        ]

    if failures:
        raise AssertionError(
            f"per-rank correctness failures (candidate, rank, err): {failures}"
        )

    # Tail-mask coverage and bias placement are rank-invariant; take rank 0's.
    for key in ("block_m", "block_n", "m_tail", "n_tail"):
        if key in rows[0]:
            ret[key] = rows[0][key]
    bias_errs = [
        r[k] for r in rows for k in r if k.endswith("bias_err") and r[k] is not None
    ]
    if bias_errs:
        ret["bias err"] = max(bias_errs)
    return ret


def _tail_note(rows):
    """Human-readable note on which tail mask this shape exercised.

    This is the evidence that the ragged shapes in ``MNK_DEFAULT`` actually
    reached the kernel's masked path, rather than being silently aligned by the
    op's ``block_m`` halving heuristic.
    """
    if not rows or "block_m" not in rows[0]:
        return "?"
    tags = []
    if rows[0]["m_tail"]:
        tags.append("M-tail")
    if rows[0]["n_tail"]:
        tags.append("N-tail")
    return "+".join(tags) if tags else "aligned"


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------

CORRECTNESS_CANDIDATES = ("fused", "standalone", "rccl")


@benchmark()
def test_gemm_reduce_scatter(tp_size, M, N, K, dtype, bias_mode, heap_size):
    """Correctness + perf for both ops against an untimed fp32 torch reference.

    ``rccl`` (``mm`` + ``dist.reduce_scatter_tensor``) is a timed candidate, not
    the reference: it is the production path the fused op replaces, and it
    independently pins down which rows land on which rank.
    """
    cfg = {
        "M": M,
        "N": N,
        "K": K,
        "dtype": dtype,
        "bias_mode": bias_mode,
        "heap_size": heap_size,
        "seed": 42,
    }
    rows = _spawn(_worker_correctness, tp_size, cfg)

    flops, nbytes = _roofline(
        M,
        N,
        K // tp_size,
        M // tp_size,
        tp_size,
        torch.empty(0, dtype=dtype).element_size(),
    )
    ret = _fold(rows, CORRECTNESS_CANDIDATES, flops, nbytes)
    ret["tail"] = _tail_note(rows)
    return ret


@benchmark()
def test_graph_capture(tp_size, M, N, K, dtype, bias_mode, replays, heap_size):
    """Capture the fused op in a HIP graph and replay it.

    This is the check that matters most: it fails loudly if any symmetric-heap
    allocation, host sync or device query survives on the op's hot path. Capture
    itself would fail on an allocation; the assertions additionally pin the heap
    bump-allocator offset, the buffer cache size and both buffer addresses.
    """
    cfg = {
        "M": M,
        "N": N,
        "K": K,
        "dtype": dtype,
        "bias_mode": bias_mode,
        "heap_size": heap_size,
        "replays": replays,
        "seed": 42,
    }
    rows = _spawn(_worker_graph, tp_size, cfg)

    flops, nbytes = _roofline(
        M,
        N,
        K // tp_size,
        M // tp_size,
        tp_size,
        torch.empty(0, dtype=dtype).element_size(),
    )
    ret = _fold(rows, ("eager", "hipgraph"), flops, nbytes)
    ret["replays"] = replays
    ret["replay drift"] = max(r["replay_drift"] for r in rows)
    ret["heap growth B"] = max((r["heap_growth_B"] or 0) for r in rows)
    ret["torch mem growth B"] = max(r["torch_mem_growth_B"] for r in rows)
    ret["tail"] = _tail_note(rows)
    return ret


@benchmark()
def test_unbarriered_drift(tp_size, M, N, K, dtype, bias_mode, iters, heap_size):
    """Run the fused op with no host barrier for ``iters`` iterations, twice.

    The reported speedup is measured in exactly this regime, and the upstream
    correctness run used barriers, so nothing else validates it.
    """
    cfg = {
        "M": M,
        "N": N,
        "K": K,
        "dtype": dtype,
        "bias_mode": bias_mode,
        "heap_size": heap_size,
        "iters": iters,
        "seed": 42,
    }
    rows = _spawn(_worker_drift, tp_size, cfg)

    flops, nbytes = _roofline(
        M,
        N,
        K // tp_size,
        M // tp_size,
        tp_size,
        torch.empty(0, dtype=dtype).element_size(),
    )
    ret = _fold(rows, ("fused",), flops, nbytes)
    avg_unbarriered = sum(r["unbarriered_us"] for r in rows) / len(rows)
    ret["iters"] = iters
    ret["unbarriered us"] = avg_unbarriered
    ret["unbarriered TFLOPS"] = flops / avg_unbarriered / 1e6
    ret["unbarriered TB/s"] = nbytes["unbarriered"] / avg_unbarriered / 1e6
    ret["max drift"] = max(r["max_drift"] for r in rows)
    ret["drift bound"] = rows[0]["drift_bound"]
    ret["heap growth B"] = max((r["heap_growth_B"] or 0) for r in rows)
    ret["tail"] = _tail_note(rows)
    return ret


def summarize(name, rows):
    """Print the results table. Must never be able to fail the run.

    ``DataFrame.to_markdown`` needs the optional ``tabulate`` package, and
    pandas raises ``ImportError`` when it is absent. That is a formatting
    dependency, but because ``main`` calls this between test groups an
    ImportError here aborts everything after it: measured on radha f13-21, the
    30 correctness configurations all passed and then this line killed the
    process before ``test_graph_capture`` or ``test_unbarriered_drift`` ran at
    all, so the suite reported nothing and skipped its two hardest cases.
    A results table is not worth a test run; fall back to plain text.
    """
    if not rows:
        logger.warning("%s: no configurations ran", name)
        return
    df = pd.DataFrame(rows)
    try:
        table = df.to_markdown(index=False)
    except ImportError:
        table = df.to_string(index=False)
    aiter.logger.info("%s summary:\n%s", name, table)


def main():
    if not IRIS_AVAILABLE or FUSED_RS is None:
        logger.warning(
            "iris / the fused GEMM+RS op module is unavailable; skipping. "
            'Install iris with: pip install -e ".[triton_comms]"'
        )
        return
    if get_gfx() not in SUPPORTED_GFX:
        logger.warning(
            "gemm_reduce_scatter_m_shard unsupported on %s; skipping", get_gfx()
        )
        return

    parser = argparse.ArgumentParser(
        formatter_class=argparse.RawTextHelpFormatter,
        description="config input of test",
    )
    parser.add_argument(
        "-d",
        "--dtype",
        type=dtypes.str2Dtype,
        nargs="*",
        default=[dtypes.fp16],
        help="data type, e.g. -d fp16 bf16",
    )
    parser.add_argument(
        "-n",
        "--num_gpus",
        type=int,
        nargs="*",
        default=TP_DEFAULT,
        help="tensor parallel sizes to sweep, e.g. -n 2 4 8",
    )
    parser.add_argument(
        "-s",
        "--mnk",
        type=dtypes.str2tuple,
        nargs="*",
        default=MNK_DEFAULT,
        help="(M, N, K) shapes; K is the full reduction dim (K_local = K // tp)",
    )
    # No -b: this op has no batch dim, and -b is the repo's batch flag.
    parser.add_argument(
        "--bias",
        type=str,
        nargs="*",
        choices=["none", "rank0"],
        default=["none", "rank0"],
        help="bias mode: none, or a (N,) bias passed on rank 0 only",
    )
    parser.add_argument(
        "--heap_size",
        type=int,
        default=1 << 30,
        help="iris symmetric heap size in bytes (default: 1GB)",
    )
    args = parser.parse_args()

    available = torch.cuda.device_count()
    tp_sizes = []
    for tp in args.num_gpus:
        if tp > available:
            logger.warning(
                "TP=%d requested but only %d GPU(s) visible; skipping", tp, available
            )
            continue
        tp_sizes.append(tp)
    if not tp_sizes:
        logger.warning("no usable TP size; skipping")
        return

    for dtype in args.dtype:
        rows = []
        for tp, (M, N, K), bias_mode in itertools.product(
            tp_sizes, args.mnk, args.bias
        ):
            if M % tp or K % tp:
                logger.warning(
                    "skipping M=%d N=%d K=%d at TP=%d: M and K must both be "
                    "divisible by the world size",
                    M,
                    N,
                    K,
                    tp,
                )
                continue
            rows.append(
                test_gemm_reduce_scatter(tp, M, N, K, dtype, bias_mode, args.heap_size)
            )
        summarize(
            f"gemm_reduce_scatter_m_shard / reduce_scatter_m_shard [{dtype}]", rows
        )

        # Capture and drift are per-regime checks, not a full shape sweep: run
        # them on the largest aligned shape and on a ragged one, with bias (the
        # harder case) and without.
        regime_shapes = [s for s in args.mnk if s in (MNK_DEFAULT[2], MNK_DEFAULT[3])]
        regime_shapes = regime_shapes or args.mnk[:1]

        rows = []
        for tp, (M, N, K), bias_mode in itertools.product(
            tp_sizes, regime_shapes, args.bias
        ):
            if M % tp or K % tp:
                continue
            rows.append(
                test_graph_capture(
                    tp, M, N, K, dtype, bias_mode, GRAPH_REPLAYS, args.heap_size
                )
            )
        summarize(f"gemm_reduce_scatter_m_shard HIP-graph capture [{dtype}]", rows)

        rows = []
        for tp, (M, N, K), bias_mode in itertools.product(
            tp_sizes, regime_shapes, args.bias
        ):
            if M % tp or K % tp:
                continue
            rows.append(
                test_unbarriered_drift(
                    tp, M, N, K, dtype, bias_mode, DRIFT_ITERS, args.heap_size
                )
            )
        summarize(f"gemm_reduce_scatter_m_shard unbarriered drift [{dtype}]", rows)


if __name__ == "__main__":
    mp.freeze_support()
    main()
