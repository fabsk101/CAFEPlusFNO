"""Regression tests for compact independent CAFE+ factor generators."""

from __future__ import annotations

import ast
import copy
import inspect
import math
import textwrap
import unittest
from unittest import mock

import torch

from experiments.common.seed import set_seed
from experiments.configs.darcy import (
    CAFEPLUSFNO_COMMON_CONFIG,
    CAFEPLUSFNO_FACTOR_COMMON_CONFIG,
    CAFEPLUSFNO_FACTORIZATIONS,
    model_constructor_kwargs,
)
from models.Cafe_Plus_FNO1D import (
    CAFEPlusFNO1D,
    CAFEPlusKernelGenerator1D,
    __version__ as VERSION_1D,
    enforce_rfft1_kernel_hermitian,
)
from models.Cafe_Plus_FNO2D import (
    CAFEPlusFNO2D,
    CAFEPlusKernelGenerator2D,
    __version__ as VERSION_2D,
    enforce_rfft2_kernel_hermitian,
)


CAFE_DARCY_MODELS = (
    "cafe_plus_fno",
    "cp_cafe_plus_fno",
    "tt_cafe_plus_fno",
    "tucker_cafe_plus_fno",
)
EXPECTED_COUNTS = {
    "cafe_plus_fno": 345_601,
    "cp_cafe_plus_fno": 80_517,
    "tt_cafe_plus_fno": 108_293,
    "tucker_cafe_plus_fno": 161_605,
}
EXPECTED_HIDDEN = {"cp": 16, "tt": 24, "tucker": 16}


def _small_model(model_class, factorization: str, **overrides):
    kwargs = {
        "width": 4,
        "input_dim": 2,
        "output_dim": 1,
        "num_layers": 1,
        "ffn_expansion": 2,
        "rff_basis": 3,
        "cheb_basis": 4,
        "cafe_branches": 2,
        "cafe_branch_dim": 5,
        "kernel_hidden_dim": 6,
        "factor_rff_basis": 16,
        "factor_cheb_basis": 8,
        "factor_branch_dim": 12,
        "factor_hidden_dim": EXPECTED_HIDDEN.get(factorization, 16),
        "factor_output_init_mode": "xavier",
        "factor_kernel_target_rms": 1e-3,
        "factorization": factorization,
        "rank": 2,
        "input_layout": "channels_first",
        "output_layout": "channels_first",
        **overrides,
    }
    return model_class(**kwargs)


def _generator(model):
    return model.operator_layers[0].spectral.kernel_generator


def _projected_raw(generator, shape: tuple[int, ...]) -> torch.Tensor:
    real, imag = generator._raw_factorized_components(shape)
    raw = torch.complex(real.float(), imag.float())
    if len(shape) == 1:
        return enforce_rfft1_kernel_hermitian(raw, shape[0])
    return enforce_rfft2_kernel_hermitian(raw, shape[1])


class CompactConfigurationTests(unittest.TestCase):
    def test_checkpoint_correctness_patch_version(self) -> None:
        self.assertEqual(VERSION_1D, "0.5.1")
        self.assertEqual(VERSION_2D, "0.5.1")

    def test_darcy_factor_settings_are_explicit_and_dense_common_is_clean(self) -> None:
        expected_common = {
            "factor_rff_basis": 16,
            "factor_cheb_basis": 8,
            "factor_branch_dim": 12,
            "factor_output_init_mode": "xavier",
            "factor_kernel_target_rms": 1e-3,
        }
        self.assertEqual(CAFEPLUSFNO_FACTOR_COMMON_CONFIG, expected_common)
        self.assertTrue(expected_common.keys().isdisjoint(CAFEPLUSFNO_COMMON_CONFIG))

        expected_variants = {
            "cp_cafe_plus_fno": {
                **expected_common,
                "factorization": "cp",
                "rank": 8,
                "factor_hidden_dim": 16,
            },
            "tt_cafe_plus_fno": {
                **expected_common,
                "factorization": "tt",
                "rank": 8,
                "factor_hidden_dim": 24,
            },
            "tucker_cafe_plus_fno": {
                **expected_common,
                "factorization": "tucker",
                "rank": 10,
                "factor_hidden_dim": 16,
            },
        }
        self.assertEqual(
            CAFEPLUSFNO_FACTORIZATIONS["cafe_plus_fno"],
            {
                "factorization": "dense",
                "rank": 16,
                "cafe_branch_dim": 32,
                "kernel_hidden_dim": 32,
            },
        )
        for model_name, expected in expected_variants.items():
            with self.subTest(model=model_name):
                configured = CAFEPLUSFNO_FACTORIZATIONS[model_name]
                self.assertEqual(configured, expected)
                self.assertTrue(expected_common.keys() <= configured.keys())
                self.assertIn("factor_hidden_dim", configured)
                kwargs = model_constructor_kwargs(model_name)
                model = CAFEPlusFNO2D(**kwargs)
                resolved = model.get_config()
                for key, value in expected.items():
                    self.assertEqual(kwargs[key], value)
                    self.assertEqual(resolved[key], value)

    def test_public_factor_api_defaults_are_explicit(self) -> None:
        expected = {
            "factor_rff_basis": 16,
            "factor_cheb_basis": 8,
            "factor_branch_dim": 12,
            "factor_hidden_dim": 16,
            "factor_output_init_mode": "xavier",
            "factor_kernel_target_rms": 1e-3,
        }
        for model_class in (CAFEPlusFNO1D, CAFEPlusFNO2D):
            signature = inspect.signature(model_class)
            self.assertEqual(
                {
                    name: signature.parameters[name].default
                    for name in expected
                },
                expected,
            )

    def test_dense_ignores_factor_only_arguments_exactly(self) -> None:
        sample_1d = torch.linspace(-1.0, 1.0, 18).reshape(1, 2, 9)
        sample_2d = torch.linspace(-1.0, 1.0, 70).reshape(1, 2, 5, 7)
        for model_class, sample in (
            (CAFEPlusFNO1D, sample_1d),
            (CAFEPlusFNO2D, sample_2d),
        ):
            with self.subTest(model=model_class.__name__):
                set_seed(20260905)
                reference = _small_model(model_class, "dense").eval()
                set_seed(20260905)
                candidate = _small_model(
                    model_class,
                    "dense",
                    factor_rff_basis=-1,
                    factor_cheb_basis=-1,
                    factor_branch_dim=-1,
                    factor_hidden_dim=-1,
                    factor_output_init_mode="ignored-on-dense",
                    factor_kernel_target_rms=-1.0,
                ).eval()
                self.assertEqual(tuple(reference.state_dict()), tuple(candidate.state_dict()))
                for key in reference.state_dict():
                    self.assertTrue(
                        torch.equal(reference.state_dict()[key], candidate.state_dict()[key])
                    )
                with torch.no_grad():
                    self.assertTrue(torch.equal(reference(sample), candidate(sample)))
                config = candidate.get_config()
                for key in (
                    "factor_rff_basis",
                    "factor_cheb_basis",
                    "factor_embedding_dim",
                    "factor_branch_dim",
                    "factor_hidden_dim",
                    "factor_output_init_mode",
                    "factor_kernel_target_rms",
                ):
                    self.assertNotIn(key, config)
                self.assertIsNone(
                    candidate.architecture_summary()["factor_embedding_dim"]
                )

    def test_darcy_compact_dimensions_and_exact_parameter_counts(self) -> None:
        for model_name in CAFE_DARCY_MODELS:
            with self.subTest(model=model_name):
                set_seed(42)
                model = CAFEPlusFNO2D(**model_constructor_kwargs(model_name))
                self.assertEqual(
                    sum(parameter.numel() for parameter in model.parameters()),
                    EXPECTED_COUNTS[model_name],
                )
                config = model.get_config()
                summary = model.architecture_summary()
                if model_name == "cafe_plus_fno":
                    self.assertNotIn("factor_embedding_dim", config)
                    self.assertIsNone(summary["factor_embedding_dim"])
                    generator = _generator(model)
                    self.assertEqual(config["cafe_branch_dim"], 32)
                    self.assertEqual(config["kernel_hidden_dim"], 32)
                    self.assertEqual(generator.encoder.branch_dim, 32)
                    self.assertEqual(generator.head.joint_mlp.input_dim, 32)
                    self.assertEqual(generator.head.joint_mlp.hidden_dim, 32)
                else:
                    factorization = str(config["factorization"])
                    self.assertEqual(config["factor_rff_basis"], 16)
                    self.assertEqual(config["factor_cheb_basis"], 8)
                    self.assertNotIn("factor_embedding_dim", config)
                    self.assertEqual(summary["factor_embedding_dim"], 40)
                    self.assertEqual(config["factor_branch_dim"], 12)
                    self.assertEqual(
                        config["factor_hidden_dim"], EXPECTED_HIDDEN[factorization]
                    )
                    self.assertEqual(config["factor_output_init_mode"], "xavier")
                    self.assertEqual(config["factor_kernel_target_rms"], 1e-3)
                    gain_names = [
                        name
                        for name, _ in model.named_parameters()
                        if name.endswith("factor_log_gain")
                    ]
                    self.assertEqual(len(gain_names), model.num_layers)

    def test_1d_and_2d_factor_blocks_use_the_same_compact_dimensions(self) -> None:
        for model_class in (CAFEPlusFNO1D, CAFEPlusFNO2D):
            for factorization in ("cp", "tt", "tucker"):
                with self.subTest(
                    model=model_class.__name__, factorization=factorization
                ):
                    model = _small_model(model_class, factorization)
                    generator = _generator(model)
                    for block in generator.factor_blocks.values():
                        self.assertEqual(block.embedding_dim, 40)
                        self.assertEqual(block.branch_dim, 12)
                        self.assertEqual(
                            block.mlp.hidden_dim, EXPECTED_HIDDEN[factorization]
                        )


class IndependenceAndInitializationTests(unittest.TestCase):
    def test_2d_keeps_exactly_eight_independent_factor_generators(self) -> None:
        for factorization in ("cp", "tt", "tucker"):
            with self.subTest(factorization=factorization):
                set_seed(42)
                generator = _generator(_small_model(CAFEPlusFNO2D, factorization))
                blocks = tuple(generator.factor_blocks.values())
                self.assertEqual(len(blocks), 8)
                self.assertEqual(len({id(block) for block in blocks}), 8)
                self.assertEqual(len({block.encoder.G.data_ptr() for block in blocks}), 8)
                self.assertEqual(
                    len({block.mlp.linear_out.weight.data_ptr() for block in blocks}),
                    8,
                )
                self.assertIsNot(
                    generator.factor_blocks["input_real"],
                    generator.factor_blocks["input_imag"],
                )

    def test_dense_output_is_small_normal_and_factor_outputs_are_xavier(self) -> None:
        set_seed(42)
        dense = CAFEPlusFNO2D(**model_constructor_kwargs("cafe_plus_fno"))
        dense_output = _generator(dense).head.joint_mlp.linear_out.weight
        self.assertEqual(_generator(dense).head.joint_mlp.output_init_mode, "normal")
        self.assertLess(abs(float(dense_output.std().item()) - 1e-3), 5e-5)

        for factorization in ("cp", "tt", "tucker"):
            with self.subTest(factorization=factorization):
                set_seed(42)
                generator = _generator(_small_model(CAFEPlusFNO2D, factorization))
                for block in generator.factor_blocks.values():
                    weight = block.mlp.linear_out.weight
                    fan_in, fan_out = block.mlp.hidden_dim, block.mlp.output_dim
                    bound = math.sqrt(6.0 / (fan_in + fan_out))
                    self.assertEqual(block.mlp.output_init_mode, "xavier")
                    self.assertLessEqual(float(weight.abs().max().item()), bound + 1e-7)
                    self.assertGreater(float(weight.std().item()), 1e-2)
                    self.assertTrue(torch.count_nonzero(block.mlp.linear_out.bias) == 0)

    def test_gain_and_base_scale_exist_only_for_factorized_generators(self) -> None:
        dense = _generator(_small_model(CAFEPlusFNO2D, "dense"))
        self.assertFalse(hasattr(dense, "factor_log_gain"))
        self.assertFalse(hasattr(dense, "factor_kernel_base_scale"))
        self.assertFalse(hasattr(dense, "_factor_kernel_calibrated"))
        for factorization in ("cp", "tt", "tucker"):
            generator = _generator(_small_model(CAFEPlusFNO2D, factorization))
            self.assertTrue(hasattr(generator, "factor_log_gain"))
            self.assertTrue(hasattr(generator, "factor_kernel_base_scale"))
            self.assertFalse(generator._factor_kernel_calibrated)
            self.assertEqual(float(generator.factor_log_gain.item()), 0.0)
            self.assertEqual(float(generator.factor_kernel_base_scale.item()), 0.0)


class CalibrationAndStateTests(unittest.TestCase):
    def test_fresh_generators_calibrate_once_across_repeated_forwards(self) -> None:
        for model_class, shape in (
            (CAFEPlusFNO1D, (9,)),
            (CAFEPlusFNO2D, (5, 7)),
        ):
            for factorization in ("cp", "tt", "tucker"):
                with self.subTest(
                    model=model_class.__name__, factorization=factorization
                ):
                    set_seed(42)
                    generator = _generator(
                        _small_model(model_class, factorization)
                    ).eval()
                    self.assertFalse(generator._factor_kernel_calibrated)
                    with mock.patch.object(
                        generator,
                        "_calibrate_factor_kernel",
                        wraps=generator._calibrate_factor_kernel,
                    ) as calibrate:
                        outputs = [generator(shape) for _ in range(21)]
                    self.assertEqual(calibrate.call_count, 1)
                    self.assertTrue(generator._factor_kernel_calibrated)
                    for output in outputs[1:]:
                        self.assertTrue(torch.equal(outputs[0], output))

    def test_uncalibrated_checkpoint_loads_as_uncalibrated_then_calibrates(self) -> None:
        cases = (
            (
                CAFEPlusFNO1D,
                (9,),
                torch.linspace(-1.0, 1.0, 18).reshape(1, 2, 9),
            ),
            (
                CAFEPlusFNO2D,
                (5, 7),
                torch.linspace(-1.0, 1.0, 70).reshape(1, 2, 5, 7),
            ),
        )
        for model_class, shape, sample in cases:
            for factorization in ("cp", "tt", "tucker"):
                with self.subTest(
                    model=model_class.__name__, factorization=factorization
                ):
                    set_seed(42)
                    source = _small_model(model_class, factorization).eval()
                    source_generator = _generator(source)
                    self.assertEqual(
                        float(source_generator.factor_kernel_base_scale.item()), 0.0
                    )
                    self.assertFalse(source_generator._factor_kernel_calibrated)
                    state = copy.deepcopy(source.state_dict())

                    set_seed(73)
                    restored = _small_model(model_class, factorization).eval()
                    restored.load_state_dict(state, strict=True)
                    restored_generator = _generator(restored)
                    self.assertEqual(
                        float(restored_generator.factor_kernel_base_scale.item()), 0.0
                    )
                    self.assertFalse(restored_generator._factor_kernel_calibrated)
                    with mock.patch.object(
                        restored_generator,
                        "_calibrate_factor_kernel",
                        wraps=restored_generator._calibrate_factor_kernel,
                    ) as calibrate:
                        with torch.no_grad():
                            output = restored(sample)
                    self.assertEqual(calibrate.call_count, 1)
                    self.assertTrue(restored_generator._factor_kernel_calibrated)
                    self.assertGreater(
                        float(restored_generator.factor_kernel_base_scale.item()), 0.0
                    )
                    kernel = restored_generator(shape)
                    kernel_rms = kernel.abs().square().mean().sqrt()
                    self.assertAlmostEqual(float(kernel_rms.item()), 1e-3, places=8)
                    self.assertTrue(torch.isfinite(output).all().item())

    def test_corrupt_calibration_scales_are_rejected(self) -> None:
        for model_class in (CAFEPlusFNO1D, CAFEPlusFNO2D):
            for factorization in ("cp", "tt", "tucker"):
                for invalid_scale in (float("nan"), float("inf"), -1.0):
                    with self.subTest(
                        model=model_class.__name__,
                        factorization=factorization,
                        invalid_scale=invalid_scale,
                    ):
                        source = _small_model(model_class, factorization)
                        state = copy.deepcopy(source.state_dict())
                        calibration_keys = [
                            key
                            for key in state
                            if key.endswith("factor_kernel_base_scale")
                        ]
                        self.assertEqual(len(calibration_keys), 1)
                        state[calibration_keys[0]].fill_(invalid_scale)
                        restored = _small_model(model_class, factorization)
                        with self.assertRaisesRegex(
                            RuntimeError, "finite non-negative scalar"
                        ):
                            restored.load_state_dict(state, strict=True)

    def test_steady_state_forward_has_no_tensor_scalar_extraction(self) -> None:
        forbidden_methods = {"item", "cpu", "numpy"}
        for generator_class in (
            CAFEPlusKernelGenerator1D,
            CAFEPlusKernelGenerator2D,
        ):
            with self.subTest(generator=generator_class.__name__):
                forward_source = textwrap.dedent(
                    inspect.getsource(generator_class.forward)
                )
                tree = ast.parse(forward_source)
                called_attributes = {
                    node.func.attr
                    for node in ast.walk(tree)
                    if isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                }
                self.assertTrue(forbidden_methods.isdisjoint(called_attributes))
                self.assertIn("_factor_kernel_calibrated", forward_source)
                calibration_source = inspect.getsource(
                    generator_class._calibrate_factor_kernel
                )
                self.assertNotIn(
                    "factor_kernel_base_scale.item", calibration_source
                )

    def test_initial_projected_kernel_rms_matches_target_in_1d_and_2d(self) -> None:
        for model_class, shape in (
            (CAFEPlusFNO1D, (10,)),
            (CAFEPlusFNO2D, (6, 8)),
        ):
            for factorization in ("cp", "tt", "tucker"):
                with self.subTest(
                    model=model_class.__name__, factorization=factorization
                ):
                    set_seed(42)
                    generator = _generator(_small_model(model_class, factorization))
                    raw_rms = _projected_raw(generator, shape).abs().square().mean().sqrt()
                    self.assertTrue(torch.isfinite(raw_rms).item())
                    self.assertGreater(float(raw_rms.item()), 0.0)
                    scaled = generator(shape)
                    scaled_rms = scaled.abs().square().mean().sqrt()
                    self.assertTrue(torch.isfinite(scaled_rms).item())
                    self.assertAlmostEqual(float(scaled_rms.item()), 1e-3, places=8)
                    scale = float(generator.factor_kernel_base_scale.item())
                    generator(shape)
                    self.assertEqual(
                        float(generator.factor_kernel_base_scale.item()), scale
                    )

    def test_calibration_is_independent_of_first_train_or_eval_call(self) -> None:
        set_seed(42)
        eval_model = _small_model(
            CAFEPlusFNO2D, "cp", mlp_dropout=0.25
        ).eval()
        set_seed(42)
        train_model = _small_model(
            CAFEPlusFNO2D, "cp", mlp_dropout=0.25
        ).train()
        eval_generator = _generator(eval_model)
        train_generator = _generator(train_model)
        eval_generator((5, 7))
        train_generator((5, 7))
        self.assertTrue(
            torch.equal(
                eval_generator.factor_kernel_base_scale,
                train_generator.factor_kernel_base_scale,
            )
        )

    def test_state_load_preserves_calibration_and_exact_output(self) -> None:
        cases = (
            (
                CAFEPlusFNO1D,
                torch.linspace(-1.0, 1.0, 18).reshape(1, 2, 9),
            ),
            (
                CAFEPlusFNO2D,
                torch.linspace(-1.0, 1.0, 70).reshape(1, 2, 5, 7),
            ),
        )
        for model_class, sample in cases:
            for factorization in ("cp", "tt", "tucker"):
                with self.subTest(
                    model=model_class.__name__, factorization=factorization
                ):
                    set_seed(42)
                    source = _small_model(model_class, factorization).eval()
                    with torch.no_grad():
                        expected = source(sample)
                    state = copy.deepcopy(source.state_dict())
                    source_generator = _generator(source)
                    self.assertTrue(source_generator._factor_kernel_calibrated)
                    self.assertGreater(
                        float(source_generator.factor_kernel_base_scale.item()), 0.0
                    )

                    set_seed(73)
                    restored = _small_model(model_class, factorization).eval()
                    restored_generator = _generator(restored)
                    self.assertFalse(restored_generator._factor_kernel_calibrated)
                    restored.load_state_dict(state, strict=True)
                    self.assertTrue(restored_generator._factor_kernel_calibrated)
                    with mock.patch.object(
                        restored_generator,
                        "_calibrate_factor_kernel",
                        side_effect=AssertionError(
                            "loaded calibrated state must not recalibrate"
                        ),
                    ) as calibrate:
                        with torch.no_grad():
                            actual = restored(sample)
                    self.assertEqual(calibrate.call_count, 0)
                    self.assertTrue(torch.equal(expected, actual))
                    self.assertTrue(
                        torch.equal(
                            source_generator.factor_kernel_base_scale,
                            restored_generator.factor_kernel_base_scale,
                        )
                    )


class ConstructorAndPortableCheckpointTests(unittest.TestCase):
    cases = (
        (
            CAFEPlusFNO1D,
            torch.linspace(-1.0, 1.0, 18).reshape(1, 2, 9),
            {"variant", "rff_rng", "rff_seed_source", "factor_embedding_dim"},
        ),
        (
            CAFEPlusFNO2D,
            torch.linspace(-1.0, 1.0, 70).reshape(1, 2, 5, 7),
            {
                "variant",
                "tensorization",
                "rff_rng",
                "rff_seed_source",
                "factor_embedding_dim",
            },
        ),
    )

    def test_get_config_is_constructor_only_and_round_trips(self) -> None:
        for model_class, _, derived_keys in self.cases:
            accepted = set(inspect.signature(model_class.__init__).parameters) - {
                "self"
            }
            for factorization in ("dense", "cp", "tt", "tucker"):
                with self.subTest(
                    model=model_class.__name__, factorization=factorization
                ):
                    set_seed(42)
                    model = _small_model(model_class, factorization)
                    config = model.get_config()
                    self.assertTrue(set(config) <= accepted)
                    self.assertTrue(derived_keys.isdisjoint(config))
                    if factorization == "dense":
                        self.assertTrue(
                            {
                                "factor_rff_basis",
                                "factor_cheb_basis",
                                "factor_branch_dim",
                                "factor_hidden_dim",
                                "factor_output_init_mode",
                                "factor_kernel_target_rms",
                            }.isdisjoint(config)
                        )
                    rebuilt = model_class(**config)
                    self.assertEqual(rebuilt.get_config(), config)
                    summary = model.architecture_summary()
                    for key in derived_keys:
                        self.assertIn(key, summary)

    def test_checkpoint_dict_reconstructs_every_1d_and_2d_variant(self) -> None:
        for model_class, sample, _ in self.cases:
            for factorization in ("dense", "cp", "tt", "tucker"):
                with self.subTest(
                    model=model_class.__name__, factorization=factorization
                ):
                    set_seed(42)
                    source = _small_model(model_class, factorization).eval()
                    with torch.no_grad():
                        expected = source(sample)
                    checkpoint = copy.deepcopy(source.checkpoint_dict())
                    self.assertEqual(
                        checkpoint["metadata"], source.architecture_summary()
                    )
                    rebuilt = model_class(**checkpoint["config"]).eval()
                    rebuilt.load_state_dict(checkpoint["state_dict"], strict=True)
                    with torch.no_grad():
                        actual = rebuilt(sample)
                    self.assertTrue(torch.equal(expected, actual))


class CompactGradientDiagnosticTests(unittest.TestCase):
    def test_factor_gradients_remain_finite_nonzero_and_above_collapse_floor(self) -> None:
        sample = torch.linspace(-0.75, 0.75, 140).reshape(2, 2, 5, 7)
        target = torch.linspace(0.25, 1.0, 70).reshape(2, 1, 5, 7)
        for factorization in ("cp", "tt", "tucker"):
            with self.subTest(factorization=factorization):
                set_seed(42)
                model = _small_model(
                    CAFEPlusFNO2D,
                    factorization,
                    learnable_sigma=True,
                ).train()
                generator = _generator(model)
                factors = generator.factor_tensors((5, 7))
                factor_rms = torch.cat(
                    [factor.reshape(-1) for factor in factors.values()]
                ).square().mean().sqrt()
                raw_rms = _projected_raw(generator, (5, 7)).abs().square().mean().sqrt()

                output = model(sample)
                loss = (output - target).square().mean()
                loss.backward()
                scaled = generator((5, 7))
                scaled_rms = scaled.abs().square().mean().sqrt()
                negative_x = torch.remainder(
                    -torch.arange(scaled.shape[-2]), scaled.shape[-2]
                ).long()
                boundary = scaled[..., 0]
                hermitian_error = (
                    boundary
                    - boundary.index_select(-1, negative_x).conj()
                ).abs().max()
                block = generator.factor_blocks["input_real"]
                diagnostics = {
                    "raw_kernel_rms": raw_rms,
                    "scaled_kernel_rms": scaled_rms,
                    "factor_output_rms": factor_rms,
                    "factor_linear_out_grad": block.mlp.linear_out.weight.grad.norm(),
                    "branch_grad": block.encoder.branches[0].weight.grad.norm(),
                    "sigma_grad": block.encoder.log_sigma.grad.abs(),
                    "layer_gain_grad": generator.factor_log_gain.grad.abs(),
                }
                print(
                    "COMPACT_GRADIENT_DIAGNOSTIC "
                    + factorization
                    + " "
                    + " ".join(
                        f"{key}={float(value.detach().item()):.9e}"
                        for key, value in diagnostics.items()
                    )
                    + f" hermitian_error={float(hermitian_error.item()):.9e}"
                )
                for key, value in diagnostics.items():
                    self.assertTrue(torch.isfinite(value).item(), key)
                    self.assertGreater(float(value.detach().item()), 0.0, key)
                self.assertTrue(torch.isfinite(hermitian_error).item())
                self.assertLessEqual(float(hermitian_error.item()), 1e-7)
                self.assertGreater(
                    float(diagnostics["factor_linear_out_grad"].item()), 1e-14
                )


if __name__ == "__main__":
    unittest.main()
