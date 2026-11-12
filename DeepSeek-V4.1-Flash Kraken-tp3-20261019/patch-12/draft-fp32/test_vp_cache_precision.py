# Contributed by epheien in recipes issue #2 (2026-11-11); thanks for the report, patch and reproducer.
# CPU-only regression test for the BRRR draft-logit cache precision fix (FP32 cache in both build stages).
"""CPU-only regression for proposal/verification distribution consistency.

Run: python3 path/to/patch-brrr/test_vp_cache_precision.py
These reference calculations are not evidence of SQL benchmark improvement.
The one-draft standard-rejection case must preserve the target distribution.
"""
import ast
import math
from pathlib import Path
import struct
import unittest


def fp32(x):
    return struct.unpack('<f', struct.pack('<f', x))[0]


def bf16(x):
    bits = struct.unpack('<I', struct.pack('<f', x))[0]
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def softmax(values):
    m = max(values)
    weights = [math.exp(v - m) for v in values]
    total = sum(weights)
    return [w / total for w in weights]


def standard_rejection_distribution(target, proposal, cached_proposal):
    # Draft tokens follow proposal, but standard verification reconstructs q
    # from the cached logits (including recomputing its normalization).
    accepted = [q * min(1., p / cached_q)
                for p, q, cached_q in zip(target, proposal, cached_proposal)]
    residual = [max(p - q, 0.) for p, q in zip(target, cached_proposal)]
    residual_sum = sum(residual)
    assert residual_sum > 0
    return [a + (1. - sum(accepted)) * r / residual_sum
            for a, r in zip(accepted, residual)]


class CachePrecisionTests(unittest.TestCase):
    def test_both_build_stages_allocate_fp32_proposal_cache(self):
        recipe = Path(__file__).resolve().parent.parent
        sources = []
        for stage in ('patch-brrr', 'patch-fused'):
            source = (recipe / stage / 'brrr/vp_sampling.py').read_text()
            sources.append(source)
            tree = ast.parse(source)
            matches = [node.value for node in ast.walk(tree)
                       if isinstance(node, ast.Assign)
                       and any(isinstance(t, ast.Attribute)
                               and ast.unparse(t) == 'spec.draft_logits'
                               for t in node.targets)]
            self.assertEqual(len(matches), 1)
            dtype = next(k.value for k in matches[0].keywords if k.arg == 'dtype')
            self.assertEqual(ast.unparse(dtype), 'torch.float32')
        self.assertEqual(*sources)

    def test_bf16_rounding_changes_standard_rejection_distribution(self):
        raw = list(map(fp32, [10.04, 10., 9.96]))
        proposal = softmax([v / .1 for v in raw])
        cached = softmax([bf16(v) / .1 for v in raw])
        target = [.1, .45, .45]
        actual = standard_rejection_distribution(target, proposal, cached)
        self.assertGreater(max(abs(a - p) for a, p in zip(actual, target)), .01)

    def test_fp32_cache_preserves_standard_rejection_distribution(self):
        raw = list(map(fp32, [10.04, 10., 9.96]))
        for temperature in (.01, .1, 1., 2.):
            with self.subTest(temperature=temperature):
                proposal = softmax([v / temperature for v in raw])
                cached = softmax([fp32(v) / temperature for v in raw])
                target = [.1, .45, .45]
                actual = standard_rejection_distribution(target, proposal, cached)
                for a, p in zip(actual, target):
                    self.assertAlmostEqual(a, p, places=12)


if __name__ == '__main__':
    unittest.main(verbosity=2)

