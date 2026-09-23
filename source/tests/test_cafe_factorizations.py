"""Mathematical, Siren-parity, dense-regression, and model smoke tests."""

from __future__ import annotations

import copy
import unittest

import torch

from models.Cafe_Plus_FNO1D import (
    CAFEPlusFNO1D,
    CAFEPlusKernelGenerator1D,
    _cp_reconstruct_1d,
    _tt_reconstruct_1d,
    _tucker_reconstruct_1d,
    midpoint_axis_coordinates,
)
from models.Cafe_Plus_FNO2D import (
    CAFEPlusFNO2D,
    CAFEPlusKernelGenerator2D,
    _cp_reconstruct_2d,
    _tt_reconstruct_2d,
    _tucker_reconstruct_2d,
)
from experiments.common.sirenfno_backend import bootstrap_sirenfno_backend

bootstrap_sirenfno_backend()

from SirenFNO1D import SpectralConv1d_Siren
from SirenFNO2D import SpectralConv2d_Siren


def _tensor(shape: tuple[int, ...], start: int = 1) -> torch.Tensor:
    count = 1
    for size in shape:
        count *= size
    return torch.arange(start, start + count, dtype=torch.float32).reshape(shape) / 10


class ExactContractionTests(unittest.TestCase):
    def test_midpoint_coordinates_match_siren_sampling(self) -> None:
        actual = midpoint_axis_coordinates(
            4, device=torch.device("cpu"), dtype=torch.float64
        ).squeeze(-1)
        expected = torch.tensor([-0.75, -0.25, 0.25, 0.75], dtype=torch.float64)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_1d_cp_tt_tucker_match_official_equations(self) -> None:
        cin, cout, length, rank = 2, 3, 4, 2
        input_factor = _tensor((cin, rank))
        output_factor = _tensor((cout, rank), 5)

        frequency = _tensor((length, rank), 11)
        cp_expected = torch.einsum(
            "ir,jr,lr->ijl", input_factor, output_factor, frequency
        )
        torch.testing.assert_close(
            _cp_reconstruct_1d(input_factor, output_factor, frequency), cp_expected
        )

        frequency_raw = _tensor((length, rank * rank), 21)
        frequency_core = frequency_raw.reshape(length, rank, rank).permute(1, 0, 2)
        tt_partial = torch.einsum("ir,rlq->ilq", input_factor, frequency_core)
        tt_expected = torch.einsum("ilq,qj->ijl", tt_partial, output_factor.t())
        torch.testing.assert_close(
            _tt_reconstruct_1d(input_factor, output_factor, frequency_raw),
            tt_expected,
        )

        core = _tensor((rank, rank, rank), 41)
        tucker_expected = torch.einsum(
            "abx,ia,jb,lx->ijl",
            core,
            input_factor,
            output_factor,
            frequency,
        )
        torch.testing.assert_close(
            _tucker_reconstruct_1d(core, input_factor, output_factor, frequency),
            tucker_expected,
        )

    def test_2d_cp_tt_tucker_match_official_equations(self) -> None:
        cin, cout, height, width, rank = 2, 3, 4, 5, 2
        input_factor = _tensor((cin, rank))
        output_factor = _tensor((cout, rank), 5)
        kx = _tensor((height, rank), 11)
        ky = _tensor((width, rank), 19)

        cp_expected = torch.einsum(
            "ir,jr,hr,wr->ijhw", input_factor, output_factor, kx, ky
        )
        torch.testing.assert_close(
            _cp_reconstruct_2d(input_factor, output_factor, kx, ky), cp_expected
        )

        kx_raw = _tensor((height, rank * rank), 29)
        ky_raw = _tensor((width, rank * rank), 45)
        kx_core = kx_raw.reshape(height, rank, rank).permute(1, 0, 2)
        ky_core = ky_raw.reshape(width, rank, rank).permute(1, 0, 2)
        tt_partial = torch.einsum("ir,rhq->ihq", input_factor, kx_core)
        tt_partial = torch.einsum("ihq,qwr->ihwr", tt_partial, ky_core)
        tt_expected = torch.einsum("ihwr,rj->ijhw", tt_partial, output_factor.t())
        torch.testing.assert_close(
            _tt_reconstruct_2d(input_factor, output_factor, kx_raw, ky_raw),
            tt_expected,
        )

        core = _tensor((rank, rank, rank, rank), 65)
        tucker_expected = torch.einsum(
            "abxy,ia,jb,hx,wy->ijhw",
            core,
            input_factor,
            output_factor,
            kx,
            ky,
        )
        torch.testing.assert_close(
            _tucker_reconstruct_2d(
                core, input_factor, output_factor, kx, ky
            ),
            tucker_expected,
        )


class SirenStructuralParityTests(unittest.TestCase):
    common = {
        "in_channels": 3,
        "out_channels": 4,
        "rff_basis": 3,
        "cheb_basis": 4,
        "num_branches": 2,
        "branch_dim": 5,
        "hidden_dim": 6,
        "rank": 2,
    }

    def test_1d_factor_axes_shapes_and_seed_roles_match_siren(self) -> None:
        full_length = 10
        half_length = full_length // 2 + 1
        for factorization in ("cp", "tt", "tucker"):
            with self.subTest(factorization=factorization):
                cafe = CAFEPlusKernelGenerator1D(
                    factorization=factorization, **self.common
                )
                siren = SpectralConv1d_Siren(
                    self.common["in_channels"],
                    self.common["out_channels"],
                    hidden_dim=6,
                    siren_dim_in=4,
                    factorization=factorization,
                    rank=self.common["rank"],
                )
                factors = cafe.factor_tensors((full_length,))
                spatial_dim = 4 if factorization == "tt" else 2
                expected = {
                    "input_real": (3, 2),
                    "input_imag": (3, 2),
                    "output_real": (4, 2),
                    "output_imag": (4, 2),
                    "frequency_real": (half_length, spatial_dim),
                    "frequency_imag": (half_length, spatial_dim),
                }
                self.assertEqual(
                    {role: tuple(value.shape) for role, value in factors.items()},
                    expected,
                )
                with torch.no_grad():
                    device = torch.device("cpu")
                    siren_factors = {
                        "input_real": siren.cin_r(L=[3], dev=device).squeeze(0),
                        "input_imag": siren.cin_i(L=[3], dev=device).squeeze(0),
                        "output_real": siren.cout_r(L=[4], dev=device).squeeze(0),
                        "output_imag": siren.cout_i(L=[4], dev=device).squeeze(0),
                        "frequency_real": siren.sG_r(
                            L=[half_length], dev=device
                        ).squeeze(0),
                        "frequency_imag": siren.sG_i(
                            L=[half_length], dev=device
                        ).squeeze(0),
                    }
                self.assertEqual(
                    {
                        role: tuple(value.shape)
                        for role, value in siren_factors.items()
                    },
                    expected,
                )
                self.assertEqual(cafe.num_kernel_mlps, 6)
                self.assertEqual(
                    cafe.rff_rng_metadata,
                    {
                        "rff_rng": "torch_global",
                        "rff_seed_source": "global_experiment_seed",
                    },
                )
                if factorization == "tucker":
                    self.assertEqual(tuple(cafe.core_real.shape), tuple(siren.core_r.shape))
                siren_layout = cafe._reconstruct_factorized_siren_layout(
                    factors, "real"
                )
                if factorization == "cp":
                    official_layout = _cp_reconstruct_1d(
                        siren_factors["input_real"],
                        siren_factors["output_real"],
                        siren_factors["frequency_real"],
                    )
                elif factorization == "tt":
                    official_layout = _tt_reconstruct_1d(
                        siren_factors["input_real"],
                        siren_factors["output_real"],
                        siren_factors["frequency_real"],
                    )
                else:
                    official_layout = _tucker_reconstruct_1d(
                        siren.core_r,
                        siren_factors["input_real"],
                        siren_factors["output_real"],
                        siren_factors["frequency_real"],
                    )
                self.assertEqual(tuple(siren_layout.shape), (3, 4, half_length))
                self.assertEqual(tuple(official_layout.shape), tuple(siren_layout.shape))
                self.assertEqual(tuple(cafe((full_length,)).shape), (4, 3, half_length))

    def test_2d_factor_axes_shapes_and_seed_roles_match_siren(self) -> None:
        height, full_width = 5, 8
        half_width = full_width // 2 + 1
        for factorization in ("cp", "tt", "tucker"):
            with self.subTest(factorization=factorization):
                cafe = CAFEPlusKernelGenerator2D(
                    spatial_dim=2,
                    factorization=factorization,
                    **self.common,
                )
                siren = SpectralConv2d_Siren(
                    self.common["in_channels"],
                    self.common["out_channels"],
                    hidden_dim=6,
                    siren_dim_in=4,
                    factorization=factorization,
                    rank=self.common["rank"],
                )
                factors = cafe.factor_tensors((height, full_width))
                spatial_dim = 4 if factorization == "tt" else 2
                expected = {
                    "input_real": (3, 2),
                    "input_imag": (3, 2),
                    "output_real": (4, 2),
                    "output_imag": (4, 2),
                    "kx_real": (height, spatial_dim),
                    "kx_imag": (height, spatial_dim),
                    "ky_real": (half_width, spatial_dim),
                    "ky_imag": (half_width, spatial_dim),
                }
                self.assertEqual(
                    {role: tuple(value.shape) for role, value in factors.items()},
                    expected,
                )
                with torch.no_grad():
                    device = torch.device("cpu")
                    siren_factors = {
                        "input_real": siren.cin_r(L=[3], dev=device).squeeze(0),
                        "input_imag": siren.cin_i(L=[3], dev=device).squeeze(0),
                        "output_real": siren.cout_r(L=[4], dev=device).squeeze(0),
                        "output_imag": siren.cout_i(L=[4], dev=device).squeeze(0),
                        "kx_real": siren.sx_r(L=[height], dev=device).squeeze(0),
                        "kx_imag": siren.sx_i(L=[height], dev=device).squeeze(0),
                        "ky_real": siren.sy_r(L=[half_width], dev=device).squeeze(0),
                        "ky_imag": siren.sy_i(L=[half_width], dev=device).squeeze(0),
                    }
                self.assertEqual(
                    {
                        role: tuple(value.shape)
                        for role, value in siren_factors.items()
                    },
                    expected,
                )
                self.assertEqual(cafe.num_kernel_mlps, 8)
                self.assertEqual(
                    cafe.rff_rng_metadata,
                    {
                        "rff_rng": "torch_global",
                        "rff_seed_source": "global_experiment_seed",
                    },
                )
                if factorization == "tucker":
                    self.assertEqual(tuple(cafe.core_real.shape), tuple(siren.core_r.shape))
                siren_layout = cafe._reconstruct_factorized_siren_layout(
                    factors, "real"
                )
                if factorization == "cp":
                    official_layout = _cp_reconstruct_2d(
                        siren_factors["input_real"],
                        siren_factors["output_real"],
                        siren_factors["kx_real"],
                        siren_factors["ky_real"],
                    )
                elif factorization == "tt":
                    official_layout = _tt_reconstruct_2d(
                        siren_factors["input_real"],
                        siren_factors["output_real"],
                        siren_factors["kx_real"],
                        siren_factors["ky_real"],
                    )
                else:
                    official_layout = _tucker_reconstruct_2d(
                        siren.core_r,
                        siren_factors["input_real"],
                        siren_factors["output_real"],
                        siren_factors["kx_real"],
                        siren_factors["ky_real"],
                    )
                self.assertEqual(
                    tuple(siren_layout.shape), (3, 4, height, half_width)
                )
                self.assertEqual(tuple(official_layout.shape), tuple(siren_layout.shape))
                self.assertEqual(
                    tuple(cafe((height, full_width)).shape),
                    (4, 3, height, half_width),
                )

class DenseRegressionTests(unittest.TestCase):
    cases = (
        (
            CAFEPlusFNO1D,
            (2, 1, 16),
            669797,
        ),
        (
            CAFEPlusFNO2D,
            (2, 1, 8, 10),
            735621,
        ),
    )

    def test_dense_state_including_g_strictly_restores_output(self) -> None:
        for model_class, input_shape, expected_count in self.cases:
            with self.subTest(model=model_class.__name__):
                torch.manual_seed(20260902)
                source = model_class(factorization="dense").eval()
                sample = torch.randn(input_shape)
                state = copy.deepcopy(source.state_dict())

                torch.manual_seed(73)
                restored = model_class(factorization="dense").eval()
                restored.load_state_dict(state, strict=True)
                with torch.no_grad():
                    source_output = source(sample).detach().contiguous()
                    restored_output = restored(sample).detach().contiguous()
                torch.testing.assert_close(
                    source_output, restored_output, rtol=0, atol=0
                )
                self.assertTrue(any(key.endswith(".G") for key in state))
                self.assertEqual(
                    sum(parameter.numel() for parameter in source.parameters()),
                    expected_count,
                )


class EightVariantForwardSmokeTests(unittest.TestCase):
    def test_all_1d_and_2d_variants_forward_backward(self) -> None:
        for dimension, model_class, sample in (
            (1, CAFEPlusFNO1D, torch.randn(2, 2, 9)),
            (2, CAFEPlusFNO2D, torch.randn(2, 2, 5, 7)),
        ):
            for factorization in ("dense", "cp", "tt", "tucker"):
                with self.subTest(dimension=dimension, factorization=factorization):
                    model = model_class(
                        width=4,
                        input_dim=2,
                        output_dim=1,
                        num_layers=1,
                        ffn_expansion=2,
                        rff_basis=3,
                        cheb_basis=4,
                        cafe_branches=2,
                        cafe_branch_dim=5,
                        kernel_hidden_dim=6,
                        factorization=factorization,
                        rank=2,
                        input_layout="channels_first",
                        output_layout="channels_first",
                    ).train()
                    model.zero_grad(set_to_none=True)
                    output = model(sample)
                    expected_shape = (2, 1, 9) if dimension == 1 else (2, 1, 5, 7)
                    self.assertEqual(tuple(output.shape), expected_shape)
                    self.assertTrue(torch.isfinite(output).all().item())
                    loss = output.square().mean()
                    loss.backward()
                    gradients = [
                        parameter.grad
                        for parameter in model.parameters()
                        if parameter.requires_grad and parameter.grad is not None
                    ]
                    self.assertTrue(gradients)
                    self.assertTrue(
                        all(torch.isfinite(gradient).all().item() for gradient in gradients)
                    )
                    parameter_count = sum(
                        parameter.numel() for parameter in model.parameters()
                    )
                    self.assertGreater(parameter_count, 0)


if __name__ == "__main__":
    unittest.main()
