"""SiLU row-address correctness, including offsets beyond signed int32."""

import unittest

import torch

from rtp_llm.models_py.triton_kernels.common.activation import silu_and_mul


class SiluAndMulTest(unittest.TestCase):
    def check_result(self, source, output):
        width = source.shape[1] // 2
        reference = (
            torch.nn.functional.silu(source[:, width:].float())
            * source[:, :width].float()
        ).to(source.dtype)
        returned = silu_and_mul(output, source)
        self.assertIs(returned, output)
        torch.testing.assert_close(output, reference, atol=0.01, rtol=0.01)

    def test_contiguous_tail_columns_and_dtypes(self):
        generator = torch.Generator(device="cuda").manual_seed(713)
        for dtype in (torch.float16, torch.bfloat16):
            for rows, width in ((1, 7), (17, 128), (129, 1408)):
                with self.subTest(dtype=dtype, rows=rows, width=width):
                    source = torch.randn(
                        rows, 2 * width, device="cuda", dtype=dtype, generator=generator
                    )
                    output = torch.empty(rows, width, device="cuda", dtype=dtype)
                    self.check_result(source, output)

    def test_row_offsets_exceed_signed_int32(self):
        # A 163834-token top-6 MoE with intermediate=1408 has 2,768,139,264
        # gate/up elements. Sparse row strides exercise the same offset overflow
        # while computing only three rows. No giant reference tensor is needed.
        width, rows = 1408, 3
        stride = 2**30 + 2 * width
        self.assertGreater((rows - 1) * stride, 2**31 - 1)
        for large_input in (True, False):
            with self.subTest(large_input=large_input):
                source = torch.empty_strided(
                    (rows, 2 * width),
                    (stride if large_input else 2 * width, 1),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                output = torch.empty_strided(
                    (rows, width),
                    (width if large_input else stride, 1),
                    device="cuda",
                    dtype=torch.bfloat16,
                )
                for row in range(rows):
                    source[row, :width].fill_(row + 1)
                    source[row, width:].fill_((row - 1) * 0.5)
                self.check_result(source, output)
                del source, output


if __name__ == "__main__":
    unittest.main()
