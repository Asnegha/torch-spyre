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

"""Unit tests for restickifying onto a dim of size 1.

A constant target stick moves the stick onto a size-1 dim so each stick holds
one element. The size-1 dim may exist on the host (e.g. topk over a (1, N)
tensor) or be synthetic (e.g. a 1-D gather table). No kernels run here.
"""

import unittest

import sympy
import torch
from torch._inductor.dependencies import MemoryDep
from torch._inductor.ir import FixedLayout

from torch_spyre._C import SpyreTensorLayout
from torch_spyre._inductor.pass_utils import (
    compute_restickify_target_layout,
    compute_size1_restickify_target,
    device_coordinates,
    host_coordinates,
)
from torch_spyre._inductor.propagate_layouts import _candidate_output_stls


def _contiguous(size: list[int]) -> tuple[FixedLayout, MemoryDep]:
    """Host layout and a contiguous read of it, one loop var per non-unit dim."""
    stride = [1] * len(size)
    for d in range(len(size) - 2, -1, -1):
        stride[d] = stride[d + 1] * size[d + 1]
    loop = [
        (sympy.Symbol(f"d{d}", integer=True, nonnegative=True), s, st)
        for d, (s, st) in enumerate(zip(size, stride))
        if s != 1
    ]
    index = sum((v * st for v, _, st in loop), sympy.S.Zero)
    dep = MemoryDep(
        "x", index, tuple(v for v, _, _ in loop), tuple(s for _, s, _ in loop)
    )
    host = FixedLayout(torch.device("cpu"), torch.float16, size, stride)
    return host, dep


def _size1_target(size: list[int], dim_order=None) -> SpyreTensorLayout | None:
    host, dep = _contiguous(size)
    stl = SpyreTensorLayout(
        size, list(host.stride), torch.float16, dim_order or list(range(len(size)))
    )
    return compute_size1_restickify_target(stl, host, dep)


class TestExistingSize1Dim(unittest.TestCase):
    """The host shape already has a size-1 dim to take the stick."""

    def test_moves_stick_onto_the_size1_dim(self):
        """topk over (1, 256) on dim 1: the stick leaves the reduction dim."""
        target = _size1_target([1, 256])

        # The layout (1, 256) gets when stuck on dim 0.
        expected = SpyreTensorLayout([1, 256], [256, 1], torch.float16, [1, 0])
        self.assertEqual(list(target.device_size), list(expected.device_size))
        self.assertEqual(list(target.stride_map), list(expected.stride_map))
        self.assertEqual(list(target.stride_map), [-1, 1, -1])

    def test_two_size1_dims(self):
        target = _size1_target([1, 1, 256])

        self.assertEqual(list(target.device_size), [256, 1, 1, 64])
        self.assertEqual(list(target.stride_map), [1, -1, -1, -1])


class TestSyntheticSize1Dim(unittest.TestCase):
    """No host dim has size 1, so the size-1 dim is synthetic."""

    def test_1d_table_gets_one_entry_per_stick(self):
        """The layout a host [N, 1] tensor already receives."""
        target = _size1_target([128])

        self.assertEqual(list(target.device_size), [128, 64])
        self.assertEqual(list(target.stride_map), [1, -1])

    def test_entry_count_comes_from_the_host_extent(self):
        """A 100-entry table keeps its length, not the 128 slots it sits in."""
        self.assertEqual(list(_size1_target([100]).device_size), [100, 64])

    def test_table_within_one_stick(self):
        """Fewer entries than a stick holds: no tile dim carries the index."""
        for entries in (2, 63, 64):
            with self.subTest(entries=entries):
                target = _size1_target([entries])
                self.assertEqual(list(target.device_size), [entries, 64])
                self.assertEqual(list(target.stride_map), [1, -1])

    def test_2d_operand_keeps_its_outer_dims(self):
        """The operand of a gather output [24, 8] with one element per stick."""
        target = _size1_target([24, 8])

        self.assertEqual(list(target.device_size), [8, 24, 64])
        self.assertEqual(list(target.stride_map), [1, 8, -1])

    def test_stick_spanning_several_tiles(self):
        target = _size1_target([24, 256])

        self.assertEqual(list(target.device_size), [256, 24, 64])
        self.assertEqual(list(target.stride_map), [1, 256, -1])


class TestSymbolicTarget(unittest.TestCase):
    """A target stick with a free symbol still moves the stick between host dims."""

    def test_moves_stick_to_the_other_dim(self):
        host, dep = _contiguous([128, 256])
        stl = SpyreTensorLayout([128, 256], [256, 1], torch.float16, [0, 1])
        ic = host_coordinates(host, dep, None)
        idc = device_coordinates(stl, dep, None)

        target = compute_restickify_target_layout(stl, host, ic[0], ic, idc)

        expected = SpyreTensorLayout([128, 256], [256, 1], torch.float16, [1, 0])
        self.assertEqual(list(target.device_size), list(expected.device_size))
        self.assertEqual(list(target.stride_map), list(expected.stride_map))

    def test_constant_offset_is_not_a_size1_target(self):
        """Only a target stick of 0 means a size-1 dim; other constants are offsets."""
        host, dep = _contiguous([1, 256])
        stl = SpyreTensorLayout([1, 256], [256, 1], torch.float16, [0, 1])
        ic = host_coordinates(host, dep, None)
        idc = device_coordinates(stl, dep, None)

        self.assertIsNotNone(
            compute_restickify_target_layout(stl, host, sympy.S.Zero, ic, idc)
        )
        self.assertIsNone(
            compute_restickify_target_layout(stl, host, sympy.Integer(5), ic, idc)
        )


class TestCandidateOrder(unittest.TestCase):
    """Output stick candidates try a size-1 dim only after every other dim."""

    def _candidate_stick_strides(self, size: list[int], skip_dim: int) -> list[int]:
        """Host stride of each candidate's stick; -1 means a size-1 dim."""
        host, dep = _contiguous(size)
        coords = host_coordinates(host, dep, None)
        stls = _candidate_output_stls(
            coords, dep, size, list(host.stride), coords[skip_dim], torch.float16
        )
        return [stl.stride_map[-1] for stl in stls]

    def test_unaligned_dim_beats_size1_dim(self):
        stick_strides = self._candidate_stick_strides([1, 100, 30], skip_dim=1)
        self.assertEqual(stick_strides, [1])

    def test_size1_dim_when_nothing_else(self):
        self.assertEqual(self._candidate_stick_strides([1, 100], skip_dim=1), [-1])


if __name__ == "__main__":
    unittest.main()
