# Copyright 2026 The Torch-Spyre Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Op-specific work-division constraints, collected in one place.

work_division.py's core algorithm (span reduction, priority-based
distribution, the matmul cost model) is generic over the iteration space. A
few ops/layouts additionally forbid splitting specific dims, or force a dim's
split to an exact value, for reasons the generic algorithm has no way to know
about — e.g. the backend cannot coordinate-mask a dim spread over cores, or a
QFP8WT tensor's second stick dimension must stay whole.
``collect_work_division_constraints`` calls each rule and merges the results,
so work_division.py's call sites only need one call instead of hand-invoking
every rule.

"""

import dataclasses
import typing
import sympy
from sympy import Expr, Symbol, divisors

from torch._inductor.ir import ComputedBuffer, Reduction
from torch_spyre._C import ElementArrangement

from .constants import BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP, TOPK_OPS
from .errors import Unsupported
from .pass_utils import concretize_expr, indirect_info_from_op, op_read_writes
from .logging_utils import get_inductor_logger
from . import config

if typing.TYPE_CHECKING:
    # Deferred to avoid a circular import: work_division.py imports from this
    # module, so TensorDep can only be used here as a string annotation.
    from .work_division import TensorDep

logger = get_inductor_logger("work_division_constraints")


@dataclasses.dataclass
class WorkDivConstraintContext:
    """Everything a constraint needs to decide which dims it restricts."""

    op: ComputedBuffer
    it_space: dict[Symbol, Expr]
    it_space_adjusted: dict[Symbol, Expr]
    output_td: "TensorDep"
    input_tds: "list[TensorDep]"
    stick_vars: dict[Symbol, int]
    reduction_vars: list[Symbol]
    committed_splits: dict[Symbol, int]


# Maximum number of ranked results the topk opfunc can emit on one core. Asking
# a single core for more fails in the backend allocator, so k above this must be
# split across cores (see topk_k_split_pinned).
TOPK_MAX_K_PER_CORE = 4


def _is_topk_op(op: ComputedBuffer) -> bool:
    """True iff ``op`` is a topkvalue/topkindex reduction."""
    return isinstance(op.data, Reduction) and op.data.reduction_type in TOPK_OPS


def find_topk_k_symbol(
    output_td: "TensorDep", input_tds: "list[TensorDep]"
) -> Symbol | None:
    """Return the output device-coord symbol that indexes topk's ``k`` dim.

    ``_topk_reduction_kwargs`` (lowering.py) substitutes the reduction loop var
    for the k dim in every input load, so k's symbol appears in the output's
    device coords but in no input's -- unlike batch dims, which pass through
    both sides unchanged. That asymmetry identifies k regardless of its size,
    which a size-based guess could not do (k may equal a batch dim's size).

    Returns None when no such symbol exists or more than one does (k=1 folds
    the dim away; anything else is not a well-formed topk).
    """
    input_syms = {
        s for td in input_tds for e in td.device_coords[:-1] for s in e.free_symbols
    }
    output_syms = [
        s for e in output_td.device_coords[:-1] for s in e.free_symbols
    ]
    # Preserve coord order and de-duplicate, so the result is deterministic.
    candidates = list(dict.fromkeys(s for s in output_syms if s not in input_syms))
    return candidates[0] if len(candidates) == 1 else None


def topk_k_split_is_representable(
    output_td: "TensorDep", k_sym: Symbol | None
) -> bool:
    """True iff a k-split on ``k_sym`` survives the split-encoding round-trip.

    Splits are handed to codegen by ``splits_by_index_coeff``, which keys each
    split by the *coefficient* its symbol has in the flat index rather than by
    the symbol itself. That key is only a stable identity while the
    coefficients are distinct. When a non-k dim is spread across the stick
    boundary its coordinate splits into a ``floor(d/64)`` / ``Mod(d, 64)``
    pair, and the outer half's coefficient is the stick stride -- which is also
    what k's coefficient collapses to. The k-split then re-applies to the
    searched dim, so every core searches a slice and returns its own local
    maximum: the op silently yields the top-1 value repeated k times instead of
    the k ranked values.

    Detect that case structurally, by looking for a floor/Mod over any non-k
    symbol in the output's device coordinates, and let the caller reject the op
    rather than emit wrong numbers.
    """
    if k_sym is None:
        return True
    for expr in output_td.device_coords:
        for node in sympy.preorder_traversal(expr):
            if isinstance(node, (sympy.floor, sympy.Mod)):
                syms = node.free_symbols
                if syms and k_sym not in syms:
                    return False
    return True


def topk_valid_k_splits(k_val: int, max_cores: int) -> list[int]:
    """Return every divisor of ``k_val`` that is a legal per-core k-split.

    A divisor ``d`` is legal iff ``d <= max_cores`` (it fits the core budget)
    and ``k_val // d <= TOPK_MAX_K_PER_CORE`` (each core's share fits the
    hardware limit). ``d=1`` therefore appears only when
    ``k_val <= TOPK_MAX_K_PER_CORE``. Sorted ascending, so the first entry is
    the fewest cores that will work.
    """
    return [
        int(d)
        for d in sorted(divisors(k_val))
        if d <= max_cores and k_val // d <= TOPK_MAX_K_PER_CORE
    ]


@dataclasses.dataclass
class ConstraintResult:
    """A constraint's verdict on the iteration space in a WorkDivConstraintContext.

    ``blocked`` dims must not be split beyond whatever split they already
    carry (composes by union across constraints). ``pinned`` dims must equal
    exactly the given split (composes by equality; two constraints pinning the
    same dim to different values is a modeling conflict, not something to
    silently resolve — see collect_work_division_constraints).
    """

    blocked: set[Symbol] = dataclasses.field(default_factory=set)
    pinned: dict[Symbol, int] = dataclasses.field(default_factory=dict)


def collect_work_division_constraints(
    ctx: WorkDivConstraintContext,
) -> ConstraintResult:
    """Run every constraint below against ``ctx`` and merge the results.

    A blocked dim that ``ctx.committed_splits`` has already split beyond 1 is
    dropped from the result (with a warning): a mandatory prior commitment —
    e.g. span_reduction satisfying the hardware span limit — outranks a
    constraint's preference not to split that dim further.

    Raises Unsupported if a pin conflicts with a prior span-limit commitment,
    or if two constraints pin the same dim to different values.
    """
    blocked: set[Symbol] = set()
    pinned: dict[Symbol, int] = {}
    for constraint in (
        coordinate_mask_blocked_vars,
        conv_spatial_blocked_vars,
        qfp8wt_pinned_vars,
        qfp8wt_matmul_k_pinned,
        topk_search_space_pinned,
        topk_k_split_pinned,
        indirect_access_pinned_vars,
    ):
        result = constraint(ctx)

        forced = {s for s in result.blocked if ctx.committed_splits.get(s, 1) > 1}
        if forced:
            logger.warning(
                f"{ctx.op.get_name()}: constraint {constraint.__name__} would "
                f"block dim(s) {sorted(str(s) for s in forced)} from being "
                f"split, but the hardware memory-span limit already committed "
                f"split(s) {[(str(s), ctx.committed_splits[s]) for s in forced]}; "
                f"the constraint is not honoured for those dims."
            )
        blocked |= result.blocked - forced

        for sym, split in result.pinned.items():
            committed_split = ctx.committed_splits.get(sym)
            if committed_split is not None and committed_split != split:
                raise Unsupported(
                    f"{ctx.op.get_name()}: pinned split for {sym} is {split} "
                    f"({constraint.__name__}), but hardware memory-span limit "
                    f"committed {committed_split}."
                )
            if sym in pinned and pinned[sym] != split:
                raise Unsupported(
                    f"{ctx.op.get_name()}: conflicting pinned split for {sym}: "
                    f"{pinned[sym]} (from an earlier constraint) vs {split} "
                    f"(from {constraint.__name__})."
                )
            pinned[sym] = split

    return ConstraintResult(blocked=blocked, pinned=pinned)


def coordinate_mask_blocked_vars(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Block reduction stick vars that cannot be split across cores.

    The backend cannot coordinate-mask a dim spread over cores (mirrors
    ``_get_coordinate_mask`` in codegen/superdsc.py). ``ctx.it_space`` must be
    the element-valued iteration space, since padding is defined on element
    counts.
    """
    blocked = {
        v
        for v in ctx.reduction_vars
        if v in ctx.stick_vars
        and concretize_expr(ctx.it_space[v]) % ctx.stick_vars[v] != 0
    }
    return ConstraintResult(blocked=blocked)


def conv_spatial_blocked_vars(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Block output image dims for strided convolutions.

    Splitting spatial dims produces incorrect per-core DSM addressing. Span-limit
    commitments win, handled uniformly by ``collect_work_division_constraints``.
    """
    if not config.disable_conv2d_spatial_split:
        return ConstraintResult()

    op_info = getattr(ctx.op.data, "op_info", None)
    if not isinstance(op_info, dict):
        return ConstraintResult()
    conv_params = op_info.get("conv_params")
    if not isinstance(conv_params, dict):
        return ConstraintResult()
    # Depthwise conv2d (#3510) records stride as stride_i/stride_j; forward
    # conv2d (#3284) records it as stride_h/stride_w. Accept either spelling so
    # the strided-spatial-split block covers both direct-conv paths.
    stride_i = conv_params.get("stride_i", conv_params.get("stride_h", 1))
    stride_j = conv_params.get("stride_j", conv_params.get("stride_w", 1))
    if (stride_i or 1) <= 1 and (stride_j or 1) <= 1:
        return ConstraintResult()

    write_ranges = list(next(iter(op_read_writes(ctx.op).writes)).ranges)
    blocked = {
        sym
        for sym in write_ranges[-2:]
        if sym in ctx.it_space and concretize_expr(ctx.it_space[sym]) > 1
    }
    return ConstraintResult(blocked=blocked)


def has_qfp8wt_tensor(tds: "list[TensorDep]") -> bool:
    return any(
        hasattr(td.layout.device_layout, "element_arrangement")
        and td.layout.device_layout.element_arrangement == ElementArrangement.QFP8WT
        for td in tds
    )


def qfp8wt_pinned_vars(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Pin QFP8WT tensors' second stick dimension to split=1.

    QFP8WT uses a 2D stick layout (2x64 elements, 128 bytes); both stick dims
    must stay atomic 128-byte units, so any iteration var indexing the second
    stick coordinate of the matmul kernel tensor (second input) or the output
    is pinned to exactly 1.
    """
    all_tds = ctx.input_tds + [ctx.output_td]
    if not has_qfp8wt_tensor(all_tds):
        return ConstraintResult()

    pinned: dict[Symbol, int] = {}

    if len(ctx.input_tds) > 1:
        kernel_td = ctx.input_tds[1]
        if len(kernel_td.device_coords) > 1 and has_qfp8wt_tensor([kernel_td]):
            for var in kernel_td.device_coords[-2].free_symbols:
                pinned[var] = 1

    if len(ctx.output_td.device_coords) > 1 and has_qfp8wt_tensor([ctx.output_td]):
        for var in ctx.output_td.device_coords[-2].free_symbols:
            pinned[var] = 1

    return ConstraintResult(pinned=pinned)


def qfp8wt_matmul_k_pinned(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Pin the reduction (K) dim to split=1 for batchmatmulfp8 with a QFP8WT kernel.

    Splitting K would require partial-sum accumulation across cores, which the
    QFP8WT matmul kernel does not support.
    """
    if not isinstance(ctx.op.data, Reduction):
        return ConstraintResult()
    if ctx.op.data.reduction_type not in (BATCH_MATMUL_OP, BATCH_MATMUL_FP8_OP):
        return ConstraintResult()

    all_tds = ctx.input_tds + [ctx.output_td]
    if not has_qfp8wt_tensor(all_tds):
        return ConstraintResult()

    return ConstraintResult(pinned={v: 1 for v in ctx.reduction_vars})


def topk_search_space_pinned(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Pin every topk dim except ``k`` to split=1.

    The topk opfunc scans the whole searched dimension on one core to rank its
    elements. Splitting that dimension would leave each core holding a partial
    top-k that the hardware has no way to merge, so it must stay whole; only
    ``k`` may be spread over cores (see :func:`topk_k_split_pinned`).

    Note we cannot pin ``ctx.reduction_vars`` here and stop: topk keeps the
    reduced dim in its output (at size ``k``) rather than collapsing it, so the
    searched dim still appears in the output's device coords and is therefore
    *not* classified as a reduction var. Pinning only reduction_vars let the
    planner split the searched dim across cores and silently produced partial
    results. Pinning the complement of ``k`` covers the searched dim and the
    batch dims alike -- batch dims are safe to split in principle, but the
    opfunc's per-core k slicing is only validated with k as the sole split.
    """
    if not _is_topk_op(ctx.op):
        return ConstraintResult()

    k_sym = find_topk_k_symbol(ctx.output_td, ctx.input_tds)
    # When the layout cannot carry a split to codegen intact (see
    # topk_k_split_is_representable), *no* dim may be split -- not even k, and
    # not the batch dims either, since they misroute the same way. Pinning
    # everything reproduces the historical single-core behaviour, which is what
    # k <= TOPK_MAX_K_PER_CORE relies on to stay correct.
    if not topk_k_split_is_representable(ctx.output_td, k_sym):
        return ConstraintResult(pinned={v: 1 for v in ctx.it_space_adjusted})

    return ConstraintResult(pinned={v: 1 for v in ctx.it_space_adjusted if v != k_sym})


def topk_k_split_pinned(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Pin ``k`` to the smallest split that fits the per-core k limit.

    A single core can emit at most ``TOPK_MAX_K_PER_CORE`` ranked results per
    pass; beyond that the backend's allocator fails outright ("Cannot allocate
    even the smallest size"). So ``k > TOPK_MAX_K_PER_CORE`` is not merely
    slower single-core, it is unimplementable there and *must* be spread over
    cores, each core producing its own slice of the k rows.

    We pin the smallest divisor ``d`` of ``k`` with ``k / d <=
    TOPK_MAX_K_PER_CORE``: the fewest cores that satisfy the hardware limit,
    leaving the remaining core budget for the other dims. Pinning (rather than
    merely offering candidates) keeps the distribution planner from choosing a
    legal-looking but unbuildable smaller split.
    """
    if not _is_topk_op(ctx.op):
        return ConstraintResult()

    k_sym = find_topk_k_symbol(ctx.output_td, ctx.input_tds)
    if k_sym is None:
        # k=1 collapses into the generic path; nothing to constrain.
        return ConstraintResult()

    k_val = concretize_expr(ctx.it_space[k_sym])
    max_cores = config.sencores
    splits = topk_valid_k_splits(k_val, max_cores)
    if not splits:
        raise Unsupported(
            f"topk(k={k_val}): no divisor of k in [1, {max_cores}] gives "
            f"k_per_core <= {TOPK_MAX_K_PER_CORE}, so k cannot be split "
            f"across at most {max_cores} cores"
        )

    min_k_split = splits[0]
    if min_k_split > 1:
        if not topk_k_split_is_representable(ctx.output_td, k_sym):
            # k must be split to run at all, but this layout cannot carry the
            # split through to codegen intact. Refusing to compile is the only
            # safe answer: the alternative is silently wrong numbers.
            raise Unsupported(
                f"topk(k={k_val}) requires splitting k across {min_k_split} "
                f"cores (at most {TOPK_MAX_K_PER_CORE} per core), but this "
                f"tensor's layout spreads a non-k dimension across the stick "
                f"boundary, which would misroute the split. Use "
                f"k <= {TOPK_MAX_K_PER_CORE}, or take topk along the last "
                f"dimension."
            )
        return ConstraintResult(pinned={k_sym: min_k_split})

    return ConstraintResult()


def indirect_access_pinned_vars(ctx: WorkDivConstraintContext) -> ConstraintResult:
    """Pin every dim to split=1 for ops with indirect (gather/scatter-style) access.

    The backend's indirect-addressing path runs single-core: an indexed
    dimension's coordinate depends on runtime data, not a static per-core
    offset, so the generic per-core span/coordinate arithmetic does not apply.
    """
    dep_names, _, _ = indirect_info_from_op(ctx.op)
    if not dep_names:
        return ConstraintResult()
    return ConstraintResult(pinned={v: 1 for v in ctx.it_space_adjusted})
