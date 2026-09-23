"""Role-based tests consolidated from 7 legacy modules."""

from __future__ import annotations

# Migrated from tests/test_airfoil_config.py
import inspect

import unittest

from unittest.mock import patch

from experiments.configs import airfoil

class AirfoilConfigurationTests(unittest.TestCase):

    def test_canonical_model_roster_has_exactly_eleven_models(self) -> None:
        self.assertEqual(airfoil.MODEL_CHOICES, ('fno', 'tfno_cp', 'amfno', 'sirenfno', 'cpsirenfno', 'ttsirenfno', 'tuckersirenfno', 'cafe_plus_fno', 'cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'))
        self.assertEqual(len(airfoil.MODEL_CHOICES), 11)
        self.assertNotIn('ufno', airfoil.MODEL_CHOICES)
        self.assertFalse(hasattr(airfoil, 'UFNO_CONFIG'))

    def test_protocol_is_fixed(self) -> None:
        self.assertEqual(airfoil.RESOLUTION, (221, 51))
        self.assertEqual((airfoil.INPUT_DIM, airfoil.OUTPUT_DIM), (2, 1))
        self.assertEqual(airfoil.TARGET_FIELD_INDEX, 4)
        self.assertEqual((airfoil.N_TRAIN, airfoil.N_VAL, airfoil.N_TEST), (1000, 0, 200))
        self.assertEqual(airfoil.BATCH_SIZE, 8)
        self.assertEqual(airfoil.EPOCHS, 500)
        self.assertEqual((airfoil.LEARNING_RATE, airfoil.WEIGHT_DECAY), (0.001, 0.0001))
        self.assertEqual(airfoil.EXPECTED_TRAIN_BATCHES_PER_EPOCH, 125)
        self.assertEqual(airfoil.EXPECTED_SCHEDULER_T_MAX, 62500)
        self.assertEqual(airfoil.NORMALIZATION_POLICY, 'none')
        self.assertEqual(airfoil.SELECTION_POLICY, 'fixed_final_epoch_no_test_selection')

    def test_neuraloperator_constructor_api_and_mode_translation(self) -> None:
        self.assertEqual(airfoil.EFFECTIVE_RETAINED_MODES, (12, 12))
        self.assertEqual(airfoil.NEURALOP_N_MODES, (24, 22))
        for config in (airfoil.FNO_CONFIG, airfoil.TFNO_CP_CONFIG):
            self.assertEqual(config['lifting_channel_ratio'], 2)
            self.assertEqual(config['projection_channel_ratio'], 2)
            self.assertEqual(config['positional_embedding'], 'grid')
            self.assertNotIn('lifting_channels', config)
            self.assertNotIn('projection_channels', config)
            self.assertEqual(config['domain_padding_mode'], 'one-sided')
        self.assertEqual(airfoil.TFNO_CP_CONFIG['factorization'], 'cp')
        self.assertEqual(airfoil.TFNO_CP_CONFIG['rank'], 0.05)

    def test_siren_airfoil_kwargs_are_explicit_and_equal_capacity(self) -> None:
        expected_common = {'width': 32, 'input_dim': 2, 'output_dim': 1, 'padding': 8, 'mlp_dropout': 0.0, 'add_grid': True, 'hidden_dim': 32, 'omega': 30.0, 'n_hidden': 1, 'siren_dim_in': 32, 'ff_sigma': 512.0, 'learnable_ff': True}
        self.assertEqual(airfoil.SIRENFNO_AIRFOIL_COMMON_CONFIG, expected_common)
        variants = {'sirenfno': ('dense', None), 'cpsirenfno': ('cp', 8), 'ttsirenfno': ('tt', 8), 'tuckersirenfno': ('tucker', 8)}
        for name, (factorization, rank) in variants.items():
            with self.subTest(model=name):
                kwargs = airfoil.model_constructor_kwargs(name)
                for key, value in expected_common.items():
                    self.assertEqual(kwargs[key], value)
                self.assertEqual(kwargs['factorization'], factorization)
                if rank is None:
                    self.assertNotIn('rank', kwargs)
                else:
                    self.assertEqual(kwargs['rank'], rank)

    def test_cafe_factorization_policy_is_fixed(self) -> None:
        variants = airfoil.CAFEPLUSFNO_FACTORIZATIONS
        self.assertEqual(variants['cp_cafe_plus_fno']['rank'], 8)
        self.assertEqual(variants['tt_cafe_plus_fno']['rank'], 8)
        self.assertEqual(variants['tucker_cafe_plus_fno']['rank'], 8)
        self.assertEqual(variants['cp_cafe_plus_fno']['factor_hidden_dim'], 16)
        self.assertEqual(variants['tt_cafe_plus_fno']['factor_hidden_dim'], 24)
        self.assertEqual(variants['tucker_cafe_plus_fno']['factor_hidden_dim'], 16)
        for name in ('cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'):
            self.assertEqual(variants[name]['factor_rff_basis'], 16)
            self.assertEqual(variants[name]['factor_cheb_basis'], 8)
            self.assertEqual(variants[name]['factor_branch_dim'], 12)
            self.assertEqual(variants[name]['factor_output_init_mode'], 'xavier')
            self.assertEqual(variants[name]['factor_kernel_target_rms'], 0.001)

    def test_parameter_count_guards_cover_exactly_the_roster(self) -> None:
        counts = {**airfoil.EXPECTED_PINNED_PARAMETER_COUNTS, **airfoil.EXPECTED_CAFE_PARAMETER_COUNTS}
        self.assertEqual(set(counts), set(airfoil.MODEL_CHOICES))
        self.assertNotIn('ufno', counts)
        self.assertEqual(counts, {'fno': 2372513, 'tfno_cp': 131993, 'amfno': 1136673, 'sirenfno': 308897, 'cpsirenfno': 80545, 'ttsirenfno': 109217, 'tuckersirenfno': 113313, 'cafe_plus_fno': 345633, 'cp_cafe_plus_fno': 80549, 'tt_cafe_plus_fno': 108325, 'tucker_cafe_plus_fno': 113317})

    def test_ufno_exclusion_is_non_performance_based(self) -> None:
        exclusion = airfoil.AIRFOIL_EXCLUDED_BASELINES['ufno']
        self.assertFalse(exclusion['performance_based_exclusion'])
        self.assertTrue(exclusion['excluded_before_paper_training'])
        self.assertFalse(exclusion['published_result_imported'])
        self.assertIn('FNOs.py', exclusion['reason'])

    def test_provenance_avoids_unsupported_reproduction_claims(self) -> None:
        amfno = airfoil.model_configuration_provenance('amfno')
        self.assertEqual(amfno['policy'], 'amfno_airfoil_author_protocol_on_pinned_sirenfno_implementation')
        self.assertFalse(amfno['amfno_table_2_numerical_reproduction_claimed'])
        for name in airfoil.MODEL_CHOICES:
            provenance = airfoil.model_configuration_provenance(name)
            self.assertFalse(provenance['sirenfno_author_airfoil_configuration'])
            self.assertTrue(provenance['configuration_selected_before_airfoil_results'])
            self.assertFalse(provenance['validation_or_test_used_for_configuration'])

class AirfoilSeedPolicyTests(unittest.TestCase):

    def test_main_uses_one_shared_seed_after_verification(self) -> None:
        from experiments import train_airfoil
        source = inspect.getsource(train_airfoil.main)
        self.assertEqual(source.count('set_seed(args.seed)'), 1)
        self.assertLess(source.index('verify_airfoil_dataset(data_root)'), source.index('set_seed(args.seed)'))
        self.assertLess(source.index('set_seed(args.seed)'), source.index('load_airfoil(data_root)'))
        self.assertLess(source.index('load_airfoil(data_root)'), source.index('build_model(args.model, device)'))
        for forbidden in ('random.seed', 'np.random.seed', 'torch.manual_seed', 'torch.cuda.manual_seed_all', 'torch.Generator('):
            self.assertNotIn(forbidden, source)

    def test_runtime_event_order(self) -> None:
        from experiments import train_airfoil
        events: list[str] = []

        class StopAfterModel(RuntimeError):
            pass

        class FakeData:
            train_loader = [None] * 125
        with patch.object(train_airfoil, 'resolve_device', side_effect=lambda _value: events.append('resolve') or train_airfoil.torch.device('cpu')), patch.object(train_airfoil, 'source_repository_state', side_effect=lambda **_kwargs: events.append('source') or {}), patch.object(train_airfoil, 'verify_airfoil_dataset', side_effect=lambda _root: events.append('verify') or {}), patch.object(train_airfoil, 'set_seed', side_effect=lambda _seed: events.append('seed')), patch.object(train_airfoil, 'load_airfoil', side_effect=lambda _root: events.append('loader') or FakeData()), patch.object(train_airfoil, 'build_model', side_effect=lambda *_args: (events.append('model'), (_ for _ in ()).throw(StopAfterModel()))[1]):
            with self.assertRaises(StopAfterModel):
                train_airfoil.main(['--model', 'fno', '--allow-dirty-source'])
        self.assertEqual(events, ['resolve', 'source', 'verify', 'seed', 'loader', 'model'])

# Migrated from tests/test_burgers_config.py

import tempfile


from pathlib import Path

from experiments import train_burgers as train_burgers_module

from experiments import train_darcy as train_darcy_module

from experiments import train_ns2d as train_ns_module

from experiments.common.sirenfno_backend import verify_pinned_source_blobs

from experiments.configs.burgers1d import AMFNO_CONFIG as _bgcfg_AMFNO_CONFIG, AMFNO_PROVENANCE_DISCREPANCY, BATCH_SIZE as _bgcfg_BATCH_SIZE, BURGERS_SOURCE_BLOBS, CAFEPLUSFNO_COMMON_CONFIG as _bgcfg_CAFEPLUSFNO_COMMON_CONFIG, CAFEPLUSFNO_FACTOR_COMMON_CONFIG as _bgcfg_CAFEPLUSFNO_FACTOR_COMMON_CONFIG, CAFEPLUSFNO_FACTORIZATIONS as _bgcfg_CAFEPLUSFNO_FACTORIZATIONS, CANONICAL_SEED as _bgcfg_CANONICAL_SEED, EPOCHS as _bgcfg_EPOCHS, EVAL_DROP_LAST as _bgcfg_EVAL_DROP_LAST, EXPECTED_PINNED_PARAMETER_COUNTS as _bgcfg_EXPECTED_PINNED_PARAMETER_COUNTS, FNO_CONFIG as _bgcfg_FNO_CONFIG, INPUT_STEPS as _bgcfg_INPUT_STEPS, LEARNING_RATE as _bgcfg_LEARNING_RATE, MODEL_CHOICES as _bgcfg_MODEL_CHOICES, N_TEST as _bgcfg_N_TEST, N_TRAIN as _bgcfg_N_TRAIN, N_VAL as _bgcfg_N_VAL, NUM_WORKERS as _bgcfg_NUM_WORKERS, PIN_MEMORY as _bgcfg_PIN_MEMORY, PUSHFORWARD_DETACH as _bgcfg_PUSHFORWARD_DETACH, RESOLUTION as _bgcfg_RESOLUTION, ROLLOUT as _bgcfg_ROLLOUT, SCHEDULER_T_MAX as _bgcfg_SCHEDULER_T_MAX, SELECTION_POLICY as _bgcfg_SELECTION_POLICY, SIRENFNO_COMMON_CONFIG as _bgcfg_SIRENFNO_COMMON_CONFIG, SIRENFNO_FACTORIZATIONS as _bgcfg_SIRENFNO_FACTORIZATIONS, TFNO_CP_CONFIG as _bgcfg_TFNO_CP_CONFIG, TRAIN_DROP_LAST as _bgcfg_TRAIN_DROP_LAST, UFNO_CONFIG as _bgcfg_UFNO_CONFIG, UFNO_PROVENANCE_DISCREPANCY, USE_AMP as _bgcfg_USE_AMP, WEIGHT_DECAY as _bgcfg_WEIGHT_DECAY, model_configuration_provenance, model_constructor_kwargs as _bgcfg_model_constructor_kwargs

from experiments.train_burgers import parse_args as _bgcfg_parse_args, prepare_run_paths as _bgcfg_prepare_run_paths, rff_metadata as _bgcfg_rff_metadata

class BurgersConfigurationTests(unittest.TestCase):

    def test_burgers1d_is_the_only_canonical_config(self) -> None:
        repository_root = Path(__file__).resolve().parents[1]
        config_root = repository_root / 'experiments' / 'configs'
        self.assertTrue((config_root / 'burgers1d.py').is_file())
        self.assertFalse((config_root / ('burgers' + '.py')).exists())
        legacy_module = 'experiments.configs.' + 'burgers'
        stale: list[str] = []
        for path in repository_root.rglob('*.py'):
            relative = path.relative_to(repository_root).as_posix()
            if relative.startswith(('third_party/', 'dist/')):
                continue
            source = path.read_text(encoding='utf-8')
            if legacy_module in source.replace(legacy_module + '1d', ''):
                stale.append(relative)
        self.assertEqual(stale, [])

    def test_burgers_upstream_blob_provenance_matches_pinned_commit(self) -> None:
        expected = {'SirenFNO1D.py': 'e05c7fce2c72d206345acc9754615b0eb393ad3a', 'neuralop/utils.py': 'f4fa635f4726b98daefcca71d580a2beb6fd4594', 'train_Burgers.py': '099d8f7d886de62aa1911af4fb0cfbe8e867fdc6', 'utils.py': '240a272db4d1b3f96b7fa4023a581c3876613d9b'}
        self.assertEqual(BURGERS_SOURCE_BLOBS, expected)
        self.assertEqual(verify_pinned_source_blobs(expected), expected)

    def test_pinned_data_and_training_protocol(self) -> None:
        self.assertEqual((_bgcfg_RESOLUTION, _bgcfg_INPUT_STEPS, _bgcfg_ROLLOUT), (1024, 10, 10))
        self.assertEqual((_bgcfg_N_TRAIN, _bgcfg_N_VAL, _bgcfg_N_TEST), (1000, 0, 200))
        self.assertEqual((_bgcfg_BATCH_SIZE, _bgcfg_EPOCHS), (32, 500))
        self.assertEqual((_bgcfg_LEARNING_RATE, _bgcfg_WEIGHT_DECAY), (0.001, 0.0001))
        self.assertEqual(_bgcfg_SCHEDULER_T_MAX, 500)
        self.assertFalse(_bgcfg_PUSHFORWARD_DETACH)
        self.assertFalse(_bgcfg_USE_AMP)
        self.assertEqual((_bgcfg_NUM_WORKERS, _bgcfg_PIN_MEMORY), (0, True))
        self.assertTrue(_bgcfg_TRAIN_DROP_LAST)
        self.assertFalse(_bgcfg_EVAL_DROP_LAST)
        self.assertEqual(_bgcfg_SELECTION_POLICY, 'fixed_final_epoch_no_test_selection')

    def test_model_choices_are_exactly_the_main_comparison(self) -> None:
        self.assertEqual(len(_bgcfg_MODEL_CHOICES), 12)
        self.assertEqual(_bgcfg_MODEL_CHOICES, ('fno', 'ufno', 'tfno_cp', 'amfno', 'sirenfno', 'cpsirenfno', 'ttsirenfno', 'tuckersirenfno', 'cafe_plus_fno', 'cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'))

    def test_fno_tfno_and_sirenfno_match_the_pinned_source(self) -> None:
        self.assertEqual(_bgcfg_FNO_CONFIG, {'n_modes_height': 1024, 'hidden_channels': 32, 'in_channels': 10, 'out_channels': 1, 'n_layers': 4, 'lifting_channels': 64, 'projection_channels': 64})
        self.assertEqual(_bgcfg_TFNO_CP_CONFIG['factorization'], 'cp')
        self.assertEqual(_bgcfg_TFNO_CP_CONFIG['rank'], 0.05)
        self.assertEqual(_bgcfg_SIRENFNO_COMMON_CONFIG['width'], 32)
        self.assertEqual(_bgcfg_SIRENFNO_COMMON_CONFIG['ff_sigma'], 1024)
        self.assertEqual({name: (variant['factorization'], variant.get('rank')) for name, variant in _bgcfg_SIRENFNO_FACTORIZATIONS.items()}, {'sirenfno': ('dense', None), 'cpsirenfno': ('cp', 8), 'ttsirenfno': ('tt', 8), 'tuckersirenfno': ('tucker', 8)})

    def test_amfno_matches_the_pinned_released_constructor(self) -> None:
        self.assertEqual(_bgcfg_AMFNO_CONFIG, {'width': 64, 'n1': 10, 'padding': 0, 'input_dim': 10, 'output_dim': 1, 'mlp_dropout': 0})
        self.assertEqual(_bgcfg_EXPECTED_PINNED_PARAMETER_COUNTS['amfno'], 823073)
        provenance = model_configuration_provenance('amfno')
        self.assertEqual(provenance['policy'], 'pinned_released_constructor')
        self.assertEqual(provenance['released_constructor_width'], 64)
        self.assertEqual(provenance['resolved_constructor_width'], 64)
        self.assertFalse(provenance['table_3_architecture_parity_claimed'])
        self.assertFalse(provenance['prior_width_32_checkpoints_compatible'])
        self.assertIn('distinct from prior width 32', AMFNO_PROVENANCE_DISCREPANCY)

    def test_ufno_retains_released_constructor_and_discloses_discrepancy(self) -> None:
        self.assertEqual(_bgcfg_UFNO_CONFIG, {'n_modes_height': 32, 'hidden_channels': 64, 'in_channels': 10, 'out_channels': 1, 'n_layers': 6, 'lifting_channels': 64, 'projection_channels': 64, 'positional_embedding': 'grid'})
        self.assertEqual(_bgcfg_EXPECTED_PINNED_PARAMETER_COUNTS['ufno'], 1883073)
        provenance = model_configuration_provenance('ufno')
        self.assertFalse(provenance['table_3_architecture_parity_claimed'])
        self.assertIn('does not reproduce', UFNO_PROVENANCE_DISCREPANCY)
        self.assertIn('without reverse-engineering', UFNO_PROVENANCE_DISCREPANCY)

    def test_cafe_compact_factorizations_are_explicit(self) -> None:
        self.assertEqual(_bgcfg_CAFEPLUSFNO_COMMON_CONFIG['width'], 32)
        self.assertEqual(_bgcfg_CAFEPLUSFNO_COMMON_CONFIG['input_dim'], 10)
        self.assertFalse(_bgcfg_CAFEPLUSFNO_COMMON_CONFIG['learnable_sigma'])
        self.assertEqual(_bgcfg_CAFEPLUSFNO_COMMON_CONFIG['sigma_init'], 1.0)
        self.assertEqual(_bgcfg_CAFEPLUSFNO_FACTOR_COMMON_CONFIG, {'factor_rff_basis': 16, 'factor_cheb_basis': 8, 'factor_branch_dim': 12, 'factor_output_init_mode': 'xavier', 'factor_kernel_target_rms': 0.001})
        self.assertEqual(_bgcfg_CAFEPLUSFNO_FACTORIZATIONS['cp_cafe_plus_fno']['factor_hidden_dim'], 16)
        self.assertEqual(_bgcfg_CAFEPLUSFNO_FACTORIZATIONS['tt_cafe_plus_fno']['factor_hidden_dim'], 24)
        self.assertEqual(_bgcfg_CAFEPLUSFNO_FACTORIZATIONS['tucker_cafe_plus_fno']['factor_hidden_dim'], 16)
        self.assertTrue(all((_bgcfg_CAFEPLUSFNO_FACTORIZATIONS[name]['rank'] == 8 for name in ('cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'))))

    def test_constructor_configs_are_independent(self) -> None:
        first = _bgcfg_model_constructor_kwargs('tt_cafe_plus_fno')
        first['width'] = -1
        self.assertEqual(_bgcfg_model_constructor_kwargs('tt_cafe_plus_fno')['width'], 32)

    def test_cli_defaults_and_required_model(self) -> None:
        parsed = _bgcfg_parse_args(['--model', 'fno'])
        self.assertEqual(parsed.seed, _bgcfg_CANONICAL_SEED)
        self.assertEqual(parsed.data_root, Path('data'))
        self.assertEqual(parsed.results_root, Path('results') / 'burgers1024_sirenfno_81918ec_protocol_v2')
        with self.assertRaises(SystemExit):
            _bgcfg_parse_args([])

    def test_seed_policy_matches_current_darcy_and_ns_order(self) -> None:
        sources = {'burgers': inspect.getsource(train_burgers_module.main), 'darcy': inspect.getsource(train_darcy_module.main), 'ns': inspect.getsource(train_ns_module.main)}
        for source in sources.values():
            self.assertEqual(source.count('set_seed(args.seed)'), 1)
        burgers = sources['burgers']
        self.assertLess(burgers.index('verify_burgers_dataset(data_root)'), burgers.index('set_seed(args.seed)'))
        self.assertLess(burgers.index('set_seed(args.seed)'), burgers.index('load_burgers(data_root)'))
        self.assertLess(burgers.index('load_burgers(data_root)'), burgers.index('build_model(args.model, device)'))
        for forbidden in ('torch.Generator(', 'manual_seed(', 'torch.seed(', 'rff_seed='):
            self.assertNotIn(forbidden, burgers)

    def test_rff_metadata_matches_every_model_family(self) -> None:
        for name in _bgcfg_MODEL_CHOICES:
            with self.subTest(model=name):
                metadata = _bgcfg_rff_metadata(name)
                if 'sirenfno' in name or 'cafe_plus_fno' in name:
                    self.assertEqual(metadata['rff_rng_policy'], 'global_torch_rng')
                    self.assertEqual(metadata['rff_seed_source'], 'global_experiment_seed')
                else:
                    self.assertEqual(metadata['rff_rng_policy'], 'not_applicable')

    def test_output_paths_are_one_model_by_one_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _bgcfg_prepare_run_paths(root, 'ufno', 73, overwrite=False)
            self.assertEqual(paths[0], root / 'ufno' / 'seed_73')
            self.assertEqual(tuple((path.name for path in paths[1:])), ('training_log.csv', 'summary.json', 'final_checkpoint.pt'))

# Migrated from tests/test_cfd1d_config.py





from experiments import train_cfd1d as train_cfd1d_module




from experiments.configs.cfd1d import AMFNO_CONFIG as _c1cfg_AMFNO_CONFIG, BATCH_SIZE as _c1cfg_BATCH_SIZE, CAFEPLUSFNO_COMMON_CONFIG as _c1cfg_CAFEPLUSFNO_COMMON_CONFIG, CAFEPLUSFNO_FACTOR_COMMON_CONFIG as _c1cfg_CAFEPLUSFNO_FACTOR_COMMON_CONFIG, CAFEPLUSFNO_FACTORIZATIONS as _c1cfg_CAFEPLUSFNO_FACTORIZATIONS, CANONICAL_SEED as _c1cfg_CANONICAL_SEED, CFD1D_SOURCE_BLOBS, DATASET_FIELD_POLICY, EPOCHS as _c1cfg_EPOCHS, EVAL_DROP_LAST as _c1cfg_EVAL_DROP_LAST, EXPERIMENT_ID, EXPECTED_PINNED_PARAMETER_COUNTS as _c1cfg_EXPECTED_PINNED_PARAMETER_COUNTS, EXPECTED_TIME_STEPS as _c1cfg_EXPECTED_TIME_STEPS, FIELD_DIM, FNO_CONFIG as _c1cfg_FNO_CONFIG, INPUT_DIM, INPUT_STEPS as _c1cfg_INPUT_STEPS, LEARNING_RATE as _c1cfg_LEARNING_RATE, MAIN_REPORTING_METRIC, MODEL_CHOICES as _c1cfg_MODEL_CHOICES, NORMALIZATION_POLICY as _c1cfg_NORMALIZATION_POLICY, N_TEST as _c1cfg_N_TEST, N_TRAIN as _c1cfg_N_TRAIN, N_VAL as _c1cfg_N_VAL, NUM_WORKERS as _c1cfg_NUM_WORKERS, OUTPUT_DIM, PHYSICAL_CHANNELS, PIN_MEMORY as _c1cfg_PIN_MEMORY, PUSHFORWARD_DETACH as _c1cfg_PUSHFORWARD_DETACH, RESOLUTION as _c1cfg_RESOLUTION, ROLLOUT as _c1cfg_ROLLOUT, SCHEDULER_T_MAX as _c1cfg_SCHEDULER_T_MAX, SECONDARY_REPORTING_METRICS, SELECTED_FIELD, SELECTION_POLICY as _c1cfg_SELECTION_POLICY, SIRENFNO_COMMON_CONFIG as _c1cfg_SIRENFNO_COMMON_CONFIG, SIRENFNO_FACTORIZATIONS as _c1cfg_SIRENFNO_FACTORIZATIONS, TFNO_CP_CONFIG as _c1cfg_TFNO_CP_CONFIG, TRAIN_DROP_LAST as _c1cfg_TRAIN_DROP_LAST, UFNO_CONFIG as _c1cfg_UFNO_CONFIG, USE_AMP as _c1cfg_USE_AMP, WEIGHT_DECAY as _c1cfg_WEIGHT_DECAY, model_constructor_kwargs as _c1cfg_model_constructor_kwargs

from experiments.train_cfd1d import parse_args as _c1cfg_parse_args, prepare_run_paths as _c1cfg_prepare_run_paths, rff_metadata as _c1cfg_rff_metadata

from scripts.run_paper_sweep import DATASETS

class CFD1DConfigurationTests(unittest.TestCase):

    def test_upstream_blob_provenance_matches_pinned_commit(self) -> None:
        expected = {'SirenFNO1D.py': 'e05c7fce2c72d206345acc9754615b0eb393ad3a', 'neuralop/utils.py': 'f4fa635f4726b98daefcca71d580a2beb6fd4594', 'train_CFD.py': '5656a35914271031ec2aee600d658daedc3f7be5', 'utils.py': '240a272db4d1b3f96b7fa4023a581c3876613d9b'}
        self.assertEqual(CFD1D_SOURCE_BLOBS, expected)
        self.assertEqual(verify_pinned_source_blobs(expected), expected)

    def test_released_runtime_data_and_training_protocol(self) -> None:
        self.assertEqual((_c1cfg_RESOLUTION, _c1cfg_EXPECTED_TIME_STEPS), (1024, 101))
        self.assertEqual((SELECTED_FIELD, DATASET_FIELD_POLICY), ('Vx', 'released_sirenfno_single_field'))
        self.assertEqual((PHYSICAL_CHANNELS, INPUT_DIM, OUTPUT_DIM, FIELD_DIM), (1, 10, 1, 1))
        self.assertEqual((_c1cfg_INPUT_STEPS, _c1cfg_ROLLOUT), (10, 10))
        self.assertEqual((_c1cfg_N_TRAIN, _c1cfg_N_VAL, _c1cfg_N_TEST), (1800, 0, 200))
        self.assertEqual((_c1cfg_BATCH_SIZE, _c1cfg_EPOCHS), (32, 500))
        self.assertEqual((_c1cfg_LEARNING_RATE, _c1cfg_WEIGHT_DECAY), (0.001, 0.0001))
        self.assertEqual(_c1cfg_SCHEDULER_T_MAX, 500)
        self.assertEqual(_c1cfg_NORMALIZATION_POLICY, 'none')
        self.assertFalse(_c1cfg_PUSHFORWARD_DETACH)
        self.assertFalse(_c1cfg_USE_AMP)
        self.assertEqual((_c1cfg_NUM_WORKERS, _c1cfg_PIN_MEMORY), (0, True))
        self.assertTrue(_c1cfg_TRAIN_DROP_LAST)
        self.assertFalse(_c1cfg_EVAL_DROP_LAST)
        self.assertEqual(_c1cfg_SELECTION_POLICY, 'fixed_final_epoch_no_test_selection')
        self.assertEqual(EXPERIMENT_ID, 'cfd1d1024_released_sirenfno_vx')
        self.assertEqual(MAIN_REPORTING_METRIC, 'final_test_corrected_trajectory_relative_l2')
        self.assertEqual(SECONDARY_REPORTING_METRICS, ('final_test_corrected_step_relative_l2',))

    def test_reporting_policy_is_written_through_shared_run_metadata(self) -> None:
        source = inspect.getsource(train_cfd1d_module.main)
        self.assertIn('"main_reporting_metric": MAIN_REPORTING_METRIC', source)
        self.assertIn('"secondary_reporting_metrics": SECONDARY_REPORTING_METRICS', source)
        self.assertIn('checkpoint = prepare_weights_only_checkpoint({', source)
        self.assertIn('summary = {', source)
        self.assertGreaterEqual(source.count('**run_metadata'), 2)

    def test_model_choices_are_exactly_the_main_comparison(self) -> None:
        self.assertEqual(_c1cfg_MODEL_CHOICES, ('fno', 'ufno', 'tfno_cp', 'amfno', 'sirenfno', 'cpsirenfno', 'ttsirenfno', 'tuckersirenfno', 'cafe_plus_fno', 'cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'))

    def test_author_baseline_constructors_are_exact(self) -> None:
        self.assertEqual(_c1cfg_FNO_CONFIG, {'n_modes_height': 1024, 'hidden_channels': 32, 'in_channels': 10, 'out_channels': 1, 'n_layers': 4, 'lifting_channels': 64, 'projection_channels': 64})
        self.assertEqual(_c1cfg_TFNO_CP_CONFIG['factorization'], 'cp')
        self.assertEqual(_c1cfg_TFNO_CP_CONFIG['rank'], 0.05)
        self.assertEqual(_c1cfg_AMFNO_CONFIG, {'width': 64, 'n1': 10, 'padding': 0, 'input_dim': 10, 'output_dim': 1, 'mlp_dropout': 0})
        self.assertEqual(_c1cfg_UFNO_CONFIG, {'n_modes_height': 32, 'hidden_channels': 64, 'in_channels': 10, 'out_channels': 1, 'n_layers': 6, 'lifting_channels': 128, 'projection_channels': 128, 'positional_embedding': 'grid'})
        self.assertEqual(_c1cfg_SIRENFNO_COMMON_CONFIG['omega'], 15.0)
        self.assertEqual(_c1cfg_SIRENFNO_COMMON_CONFIG['ff_sigma'], 256)
        self.assertEqual(_c1cfg_SIRENFNO_COMMON_CONFIG['siren_dim_in'], 16)
        self.assertEqual({name: (item['factorization'], item.get('rank')) for name, item in _c1cfg_SIRENFNO_FACTORIZATIONS.items()}, {'sirenfno': ('dense', None), 'cpsirenfno': ('cp', 8), 'ttsirenfno': ('tt', 8), 'tuckersirenfno': ('tucker', 8)})
        self.assertEqual(_c1cfg_EXPECTED_PINNED_PARAMETER_COUNTS['fno'], 4216161)
        self.assertEqual(_c1cfg_EXPECTED_PINNED_PARAMETER_COUNTS['ufno'], 1892161)

    def test_cafe_compact_factorizations_are_explicit(self) -> None:
        self.assertEqual(_c1cfg_CAFEPLUSFNO_COMMON_CONFIG['input_dim'], 10)
        self.assertEqual(_c1cfg_CAFEPLUSFNO_COMMON_CONFIG['output_dim'], 1)
        self.assertFalse(_c1cfg_CAFEPLUSFNO_COMMON_CONFIG['learnable_sigma'])
        self.assertEqual(_c1cfg_CAFEPLUSFNO_FACTOR_COMMON_CONFIG, {'factor_rff_basis': 16, 'factor_cheb_basis': 8, 'factor_branch_dim': 12, 'factor_output_init_mode': 'xavier', 'factor_kernel_target_rms': 0.001})
        expected_hidden = {'cp_cafe_plus_fno': 16, 'tt_cafe_plus_fno': 24, 'tucker_cafe_plus_fno': 16}
        for name, hidden in expected_hidden.items():
            self.assertEqual(_c1cfg_CAFEPLUSFNO_FACTORIZATIONS[name]['rank'], 8)
            self.assertEqual(_c1cfg_CAFEPLUSFNO_FACTORIZATIONS[name]['factor_hidden_dim'], hidden)

    def test_constructor_configs_are_independent(self) -> None:
        first = _c1cfg_model_constructor_kwargs('tt_cafe_plus_fno')
        first['width'] = -1
        self.assertEqual(_c1cfg_model_constructor_kwargs('tt_cafe_plus_fno')['width'], 32)

    def test_cli_and_sweep_use_canonical_result_root(self) -> None:
        parsed = _c1cfg_parse_args(['--model', 'fno'])
        self.assertEqual(parsed.seed, _c1cfg_CANONICAL_SEED)
        self.assertEqual(parsed.data_root, Path('data'))
        expected = Path('results') / 'cfd1d1024_sirenfno_81918ec'
        self.assertEqual(parsed.results_root, expected)
        self.assertEqual(DATASETS['cfd1d1024']['default_results'], expected)
        self.assertEqual(DATASETS['cfd1d1024']['models'], _c1cfg_MODEL_CHOICES)
        with self.assertRaises(SystemExit):
            _c1cfg_parse_args([])

    def test_seed_policy_matches_current_darcy_and_ns_order(self) -> None:
        sources = {'cfd': inspect.getsource(train_cfd1d_module.main), 'darcy': inspect.getsource(train_darcy_module.main), 'ns': inspect.getsource(train_ns_module.main)}
        for source in sources.values():
            self.assertEqual(source.count('set_seed(args.seed)'), 1)
        cfd = sources['cfd']
        self.assertLess(cfd.index('verify_cfd1d_dataset(data_root)'), cfd.index('set_seed(args.seed)'))
        self.assertLess(cfd.index('set_seed(args.seed)'), cfd.index('load_cfd1d(data_root)'))
        self.assertLess(cfd.index('load_cfd1d(data_root)'), cfd.index('build_model(args.model, device)'))
        for forbidden in ('torch.Generator(', 'manual_seed(', 'torch.seed(', 'rff_seed='):
            self.assertNotIn(forbidden, cfd)

    def test_rff_metadata_matches_every_model_family(self) -> None:
        for name in _c1cfg_MODEL_CHOICES:
            metadata = _c1cfg_rff_metadata(name)
            expected = 'global_torch_rng' if 'sirenfno' in name or 'cafe_plus_fno' in name else 'not_applicable'
            self.assertEqual(metadata['rff_rng_policy'], expected)

    def test_output_paths_are_one_model_by_one_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _c1cfg_prepare_run_paths(root, 'ufno', 73, overwrite=False)
            self.assertEqual(paths[0], root / 'ufno' / 'seed_73')
            self.assertEqual(tuple((path.name for path in paths[1:])), ('training_log.csv', 'summary.json', 'final_checkpoint.pt'))

# Migrated from tests/test_cfd2d_config.py





from experiments import train_cfd2d as _c2cfg_train_module


from experiments.configs import cfd2d


class CFD2DConfigurationTests(unittest.TestCase):

    def test_upstream_blob_provenance_matches_pinned_commit(self) -> None:
        self.assertEqual(
            verify_pinned_source_blobs(cfd2d.CFD2D_SOURCE_BLOBS),
            cfd2d.CFD2D_SOURCE_BLOBS,
        )

    def test_released_runtime_protocol_is_fully_explicit(self) -> None:
        self.assertEqual(cfd2d.RESOLUTION, (128, 128))
        self.assertEqual(cfd2d.EXPECTED_TIME_STEPS, 21)
        self.assertEqual(cfd2d.SELECTED_HDF5_KEY, '/Vx')
        self.assertEqual(cfd2d.CHANNEL_SEMANTICS, ('Vx velocity component',))
        self.assertEqual((cfd2d.PHYSICAL_CHANNELS, cfd2d.INPUT_DIM, cfd2d.OUTPUT_DIM, cfd2d.FIELD_DIM), (1, 5, 1, 1))
        self.assertEqual((cfd2d.INPUT_STEPS, cfd2d.ROLLOUT), (5, 5))
        self.assertEqual((cfd2d.N_TRAIN, cfd2d.N_VAL, cfd2d.N_TEST), (1800, 0, 200))
        self.assertEqual((cfd2d.BATCH_SIZE, cfd2d.EPOCHS), (32, 500))
        self.assertEqual((cfd2d.LEARNING_RATE, cfd2d.WEIGHT_DECAY), (0.001, 0.0001))
        self.assertEqual(cfd2d.SCHEDULER_T_MAX, 500)
        self.assertEqual(cfd2d.NORMALIZATION_POLICY, 'none')
        self.assertFalse(cfd2d.PUSHFORWARD_DETACH)
        self.assertFalse(cfd2d.USE_AMP)
        self.assertEqual((cfd2d.NUM_WORKERS, cfd2d.PIN_MEMORY), (0, True))
        self.assertTrue(cfd2d.TRAIN_DROP_LAST)
        self.assertFalse(cfd2d.EVAL_DROP_LAST)
        self.assertEqual(cfd2d.SELECTION_POLICY, 'fixed_final_epoch_no_test_selection')
        self.assertEqual(cfd2d.MAIN_REPORTING_METRIC, 'final_test_corrected_trajectory_relative_l2')
        self.assertEqual(cfd2d.SECONDARY_REPORTING_METRICS, ('final_test_corrected_step_relative_l2',))

    def test_model_choices_and_author_constructors_are_exact(self) -> None:
        self.assertEqual(len(cfd2d.MODEL_CHOICES), 12)
        self.assertEqual(cfd2d.FNO_CONFIG['n_modes'], (32, 32))
        self.assertEqual(cfd2d.TFNO_CP_CONFIG['rank'], 0.05)
        self.assertEqual(cfd2d.AMFNO_CONFIG, {'width': 32, 'n1': 10, 'n2': 10, 'padding': 0, 'input_dim': 5, 'output_dim': 1, 'mlp_dropout': 0})
        self.assertEqual(cfd2d.UFNO_CONFIG, {'n_modes': (12, 12), 'hidden_channels': 32, 'in_channels': 5, 'out_channels': 1, 'n_layers': 4, 'positional_embedding': 'grid', 'use_channel_mlp': True, 'use_unet_from': 2, 'unet_dropout': 0.0, 'domain_padding': None, 'fno_block_precision': 'full'})
        self.assertEqual(cfd2d.SIRENFNO_COMMON_CONFIG['omega'], 30.0)
        self.assertEqual(cfd2d.SIRENFNO_COMMON_CONFIG['ff_sigma'], 256)
        for name in ('cpsirenfno', 'ttsirenfno', 'tuckersirenfno'):
            self.assertEqual(cfd2d.SIRENFNO_FACTORIZATIONS[name]['rank'], 8)

    def test_parameter_counts_and_compact_cafe_policy_are_pinned(self) -> None:
        self.assertEqual(cfd2d.EXPECTED_PINNED_PARAMETER_COUNTS, {'fno': 4469857, 'ufno': 991009, 'tfno_cp': 237761, 'amfno': 385601, 'sirenfno': 304769, 'cpsirenfno': 64001, 'ttsirenfno': 92673, 'tuckersirenfno': 96769})
        self.assertEqual(cfd2d.EXPECTED_CAFE_PARAMETER_COUNTS, {'cafe_plus_fno': 327169, 'cp_cafe_plus_fno': 80645, 'tt_cafe_plus_fno': 108421, 'tucker_cafe_plus_fno': 113413})
        self.assertFalse(cfd2d.CAFEPLUSFNO_COMMON_CONFIG['learnable_sigma'])
        expected_hidden = {'cp_cafe_plus_fno': 16, 'tt_cafe_plus_fno': 24, 'tucker_cafe_plus_fno': 16}
        for name, hidden in expected_hidden.items():
            config = cfd2d.CAFEPLUSFNO_FACTORIZATIONS[name]
            self.assertEqual(config['rank'], 8)
            self.assertEqual(config['factor_hidden_dim'], hidden)
            self.assertEqual(config['factor_output_init_mode'], 'xavier')
            self.assertEqual(config['factor_kernel_target_rms'], 0.001)

    def test_seed_order_and_metadata_policy_are_explicit(self) -> None:
        source = inspect.getsource(_c2cfg_train_module.main)
        self.assertEqual(source.count('set_seed(args.seed)'), 1)
        self.assertLess(source.index('verify_cfd2d_dataset(data_root)'), source.index('set_seed(args.seed)'))
        self.assertLess(source.index('set_seed(args.seed)'), source.index('load_cfd2d(data_root)'))
        self.assertLess(source.index('load_cfd2d(data_root)'), source.index('build_model(args.model, device)'))
        for forbidden in ('torch.Generator(', 'manual_seed(', 'torch.seed(', 'rff_seed='):
            self.assertNotIn(forbidden, source)
        self.assertIn('"main_reporting_metric": MAIN_REPORTING_METRIC', source)
        self.assertGreaterEqual(source.count('**run_metadata'), 2)

    def test_cli_sweep_and_fail_closed_output_layout(self) -> None:
        parsed = _c2cfg_train_module.parse_args(['--model', 'fno'])
        expected = Path('results') / 'cfd2d128_sirenfno_81918ec'
        self.assertEqual(parsed.results_root, expected)
        self.assertEqual(DATASETS['cfd2d128']['default_results'], expected)
        self.assertEqual(DATASETS['cfd2d128']['experiment'], cfd2d.EXPERIMENT_ID)
        with tempfile.TemporaryDirectory() as directory:
            paths = _c2cfg_train_module.prepare_run_paths(Path(directory), 'fno', 42, overwrite=False)
            paths[1].write_text('existing', encoding='utf-8')
            with self.assertRaises(FileExistsError):
                _c2cfg_train_module.prepare_run_paths(Path(directory), 'fno', 42, overwrite=False)

# Migrated from tests/test_darcy_models.py
import gc

import json

import os




import torch


from neuralop import LpLoss

from neuralop.training import AdamW

from neuralop.utils import count_model_params

from experiments.common.seed import set_seed

from experiments.configs.darcy import AMFNO_CONFIG as _darcy_AMFNO_CONFIG, BATCH_SIZE as _darcy_BATCH_SIZE, CANONICAL_SEED as _darcy_CANONICAL_SEED, EPOCHS as _darcy_EPOCHS, EVAL_INTERVAL as _darcy_EVAL_INTERVAL, FNO_CONFIG as _darcy_FNO_CONFIG, LEARNING_RATE as _darcy_LEARNING_RATE, MODEL_CHOICES as _darcy_MODEL_CHOICES, N_TEST as _darcy_N_TEST, N_TRAIN as _darcy_N_TRAIN, RESOLUTION as _darcy_RESOLUTION, SCHEDULER_T_MAX as _darcy_SCHEDULER_T_MAX, SIRENFNO_COMMON_CONFIG as _darcy_SIRENFNO_COMMON_CONFIG, SIRENFNO_FACTORIZATIONS as _darcy_SIRENFNO_FACTORIZATIONS, TEST_BATCH_SIZE as _darcy_TEST_BATCH_SIZE, TFNO_CP_CONFIG as _darcy_TFNO_CP_CONFIG, UFNO_CONFIG as _darcy_UFNO_CONFIG, WEIGHT_DECAY as _darcy_WEIGHT_DECAY, model_constructor_kwargs as _darcy_model_constructor_kwargs

from experiments.train_darcy import AUTHOR_LOCAL_IMPORT_PATHS, FNO, author_local_import_audit, build_model, load_darcy, resolve_device, rff_metadata as _darcy_rff_metadata, upstream_versions

class DarcyConfigurationTests(unittest.TestCase):

    def test_main_comparison_model_choices_are_exact(self) -> None:
        self.assertEqual(_darcy_MODEL_CHOICES, ('fno', 'ufno', 'tfno_cp', 'amfno', 'sirenfno', 'cpsirenfno', 'ttsirenfno', 'tuckersirenfno', 'cafe_plus_fno', 'cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'))

    def test_reference_training_protocol_is_fixed(self) -> None:
        self.assertEqual((_darcy_N_TRAIN, _darcy_N_TEST), (1000, 200))
        self.assertEqual((_darcy_RESOLUTION, _darcy_BATCH_SIZE, _darcy_TEST_BATCH_SIZE), (128, 32, 32))
        self.assertEqual((_darcy_EPOCHS, _darcy_EVAL_INTERVAL), (500, 1))
        self.assertEqual((_darcy_LEARNING_RATE, _darcy_WEIGHT_DECAY), (0.001, 0.0001))
        self.assertEqual(_darcy_SCHEDULER_T_MAX, 500)

    def test_loader_arguments_exactly_match_reference_protocol(self) -> None:
        expected_result = object()
        with patch.object(train_darcy_module, 'expected_darcy_files', return_value=()), patch.object(train_darcy_module, 'load_darcy_flow_small', return_value=expected_result) as mocked_loader:
            actual = load_darcy(Path('unused-data-root'))
        self.assertIs(actual, expected_result)
        mocked_loader.assert_called_once_with(n_train=1000, batch_size=32, train_resolution=128, test_resolutions=[128], n_tests=[200], test_batch_sizes=[32], data_root='unused-data-root', download=False)

    def test_rff_metadata_matches_each_model_family(self) -> None:
        rff_models = {'cafe_plus_fno', 'cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno', 'sirenfno', 'cpsirenfno', 'ttsirenfno', 'tuckersirenfno'}
        non_rff_models = {'fno', 'tfno_cp', 'amfno', 'ufno'}
        for model_name in rff_models:
            with self.subTest(model=model_name):
                self.assertEqual(_darcy_rff_metadata(model_name), {'rff_rng_policy': 'global_torch_rng', 'rff_seed_source': 'global_experiment_seed'})
        for model_name in non_rff_models:
            with self.subTest(model=model_name):
                self.assertEqual(_darcy_rff_metadata(model_name), {'rff_rng_policy': 'not_applicable', 'rff_seed_source': 'not_applicable'})
        with self.assertRaises(KeyError):
            _darcy_rff_metadata('unknown-model')

    def test_fno_and_tfno_cp_match_reference_channels(self) -> None:
        self.assertEqual(_darcy_FNO_CONFIG['n_modes'], (16, 16))
        self.assertEqual(_darcy_FNO_CONFIG['hidden_channels'], 32)
        self.assertEqual(_darcy_FNO_CONFIG['n_layers'], 4)
        self.assertEqual(_darcy_FNO_CONFIG['lifting_channels'], 64)
        self.assertEqual(_darcy_FNO_CONFIG['projection_channels'], 64)
        self.assertNotIn('lifting_channel_ratio', _darcy_FNO_CONFIG)
        self.assertNotIn('projection_channel_ratio', _darcy_FNO_CONFIG)
        self.assertEqual(_darcy_TFNO_CP_CONFIG['implementation'], 'factorized')
        self.assertEqual(_darcy_TFNO_CP_CONFIG['factorization'], 'cp')
        self.assertEqual(_darcy_TFNO_CP_CONFIG['rank'], 0.05)

    def test_author_local_import_sources_are_isolated(self) -> None:
        audited = author_local_import_audit()
        self.assertEqual(audited, AUTHOR_LOCAL_IMPORT_PATHS)
        for name in ('neuralop', 'FNO', 'Trainer', 'LpLoss', 'H1Loss', 'load_darcy_flow_small', 'AdamW', 'count_model_params'):
            with self.subTest(component=name):
                self.assertTrue(audited[name].startswith('third_party/SirenFNO/neuralop/'))
        for name in ('SirenFNO2d', 'FNO2dMLP', 'UFNO'):
            with self.subTest(component=name):
                self.assertTrue(audited[name].startswith('third_party/SirenFNO/'))
        self.assertEqual(audited['CAFEPlusFNO2D'], 'models/Cafe_Plus_FNO2D.py')
        self.assertNotIn('site-packages', audited['neuralop'].lower())

    def test_pinned_author_source_identity(self) -> None:
        provenance = upstream_versions()
        sirenfno = provenance['SirenFNO']
        self.assertEqual(sirenfno['bundled_neuraloperator']['declared_version'], '1.0.2')
        self.assertEqual(sirenfno['commit'], '81918ecce323a2fd5c5a54db917598bda088574b')
        self.assertEqual(sirenfno['bundled_neuraloperator']['git_tree_sha1'], 'fbc6aa738d6ffc2e2ca6808e724b0083c3a688eb')
        self.assertEqual({name: identity['git_blob_sha1'] for name, identity in sirenfno['source_files'].items()}, {'SirenFNO2D.py': 'b0a987e42cd00547dfa64d0e4dfa619ee7106730', 'baseline/AMFNO.py': '70b2affe7237e0123ef0baa0bd970b62ec87a020', 'baseline/UFNO.py': '594dfbc7147ffe737930eaa12e12330c085448d3'})

    def test_fno_resolves_author_architecture_and_parameter_count(self) -> None:
        set_seed(_darcy_CANONICAL_SEED)
        built = build_model('fno', torch.device('cpu'))
        model = built.model
        self.assertIsInstance(model, FNO)
        self.assertEqual(tuple(model.n_modes), (16, 16))
        self.assertEqual(model.hidden_channels, 32)
        self.assertEqual(model.n_layers, 4)
        self.assertEqual(model.lifting_channels, 64)
        self.assertEqual(model.projection_channels, 64)
        self.assertEqual(count_model_params(model), 1192801)

    def test_local_author_model_configs_match_reference(self) -> None:
        self.assertEqual(_darcy_AMFNO_CONFIG, {'width': 32, 'n1': 10, 'n2': 10, 'padding': 0, 'input_dim': 1, 'output_dim': 1, 'mlp_dropout': 0})
        self.assertEqual(_darcy_UFNO_CONFIG['n_modes'], (12, 12))
        self.assertEqual(_darcy_UFNO_CONFIG['hidden_channels'], 64)
        self.assertEqual(_darcy_UFNO_CONFIG['n_layers'], 6)
        self.assertEqual(_darcy_UFNO_CONFIG['use_unet_from'], 3)
        self.assertEqual(_darcy_UFNO_CONFIG['unet_dropout'], 0)
        self.assertEqual(_darcy_SIRENFNO_COMMON_CONFIG['width'], 32)
        self.assertEqual(_darcy_SIRENFNO_COMMON_CONFIG['hidden_dim'], 32)
        self.assertEqual(_darcy_SIRENFNO_COMMON_CONFIG['siren_dim_in'], 32)
        self.assertEqual(_darcy_SIRENFNO_COMMON_CONFIG['ff_sigma'], 512.0)
        self.assertEqual(_darcy_SIRENFNO_FACTORIZATIONS, {'sirenfno': {'factorization': 'dense'}, 'cpsirenfno': {'factorization': 'cp', 'rank': 8}, 'ttsirenfno': {'factorization': 'tt', 'rank': 8}, 'tuckersirenfno': {'factorization': 'tucker', 'rank': 10}})

    def test_all_twelve_model_constructors_match_config_metadata(self) -> None:
        set_seed(_darcy_CANONICAL_SEED)
        for model_name in _darcy_MODEL_CHOICES:
            with self.subTest(model=model_name):
                built = build_model(model_name, torch.device('cpu'))
                requested = _darcy_model_constructor_kwargs(model_name)
                for key, expected in requested.items():
                    if key == 'padding' and isinstance(expected, int):
                        resolved = built.configuration.get(key)
                        if isinstance(resolved, tuple):
                            expected = (expected, expected)
                    self.assertEqual(built.configuration.get(key), expected)
                print('CONSTRUCTOR_AUDIT ' + json.dumps({'model': model_name, 'model_class': f'{built.model.__class__.__module__}.{built.model.__class__.__qualname__}', 'parameter_count': count_model_params(built.model), 'factorization': built.factorization, 'rank': built.rank}, sort_keys=True))
                del built
                gc.collect()

# Migrated from tests/test_ns_config.py






from torch.utils.data import SequentialSampler, TensorDataset



from experiments.configs.darcy import CAFEPLUSFNO_COMMON_CONFIG as APPROVED_CAFE_CONFIG

from experiments.configs.ns2d import AMFNO_CONFIG as _nscfg_AMFNO_CONFIG, BATCH_SIZE as _nscfg_BATCH_SIZE, CAFEPLUSFNO_COMMON_CONFIG as _nscfg_CAFEPLUSFNO_COMMON_CONFIG, CAFEPLUSFNO_FACTOR_COMMON_CONFIG as _nscfg_CAFEPLUSFNO_FACTOR_COMMON_CONFIG, CAFEPLUSFNO_FACTORIZATIONS as _nscfg_CAFEPLUSFNO_FACTORIZATIONS, CANONICAL_SEED as _nscfg_CANONICAL_SEED, DATASET_SOURCE, EPOCHS as _nscfg_EPOCHS, EVAL_INTERVAL as _nscfg_EVAL_INTERVAL, FNO_CONFIG as _nscfg_FNO_CONFIG, LEARNING_RATE as _nscfg_LEARNING_RATE, MODEL_CHOICES as _nscfg_MODEL_CHOICES, N_TEST as _nscfg_N_TEST, N_TRAIN as _nscfg_N_TRAIN, RESOLUTION as _nscfg_RESOLUTION, SCHEDULER_T_MAX as _nscfg_SCHEDULER_T_MAX, SEED_POLICY, SELECTION_POLICY as _nscfg_SELECTION_POLICY, SIRENFNO_COMMON_CONFIG as _nscfg_SIRENFNO_COMMON_CONFIG, SIRENFNO_VARIANT_CONFIGS, TEST_BATCH_SIZE as _nscfg_TEST_BATCH_SIZE, TFNO_CP_CONFIG as _nscfg_TFNO_CP_CONFIG, UFNO_CONFIG as _nscfg_UFNO_CONFIG, WEIGHT_DECAY as _nscfg_WEIGHT_DECAY, model_constructor_kwargs as _nscfg_model_constructor_kwargs

from experiments.train_ns2d import CsvLoggingTrainer, load_ns, parse_args as _nscfg_parse_args, prepare_run_paths as _nscfg_prepare_run_paths, rff_metadata as _nscfg_rff_metadata

class NSConfigurationTests(unittest.TestCase):

    def test_official_dataset_and_training_protocol(self) -> None:
        self.assertEqual((_nscfg_N_TRAIN, _nscfg_N_TEST), (1000, 200))
        self.assertEqual((_nscfg_RESOLUTION, _nscfg_BATCH_SIZE, _nscfg_TEST_BATCH_SIZE), (128, 32, 32))
        self.assertEqual((_nscfg_EPOCHS, _nscfg_EVAL_INTERVAL), (500, 1))
        self.assertEqual((_nscfg_LEARNING_RATE, _nscfg_WEIGHT_DECAY), (0.001, 0.0001))
        self.assertEqual(_nscfg_SCHEDULER_T_MAX, 500)
        self.assertEqual(_nscfg_SELECTION_POLICY, 'fixed_final_epoch_no_test_selection')
        self.assertIn('multi_seed', SEED_POLICY)
        self.assertIn('offline pipeline', DATASET_SOURCE)

    def test_model_choices_are_exactly_the_main_comparison(self) -> None:
        self.assertEqual(_nscfg_MODEL_CHOICES, ('fno', 'ufno', 'tfno_cp', 'amfno', 'sirenfno', 'cpsirenfno', 'ttsirenfno', 'tuckersirenfno', 'cafe_plus_fno', 'cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'))

    def test_official_fno_tfno_amfno_and_ufno_configs(self) -> None:
        self.assertEqual(_nscfg_FNO_CONFIG, {'n_modes': (32, 32), 'hidden_channels': 32, 'in_channels': 1, 'out_channels': 1, 'n_layers': 4, 'lifting_channels': 64, 'projection_channels': 64})
        self.assertEqual(_nscfg_TFNO_CP_CONFIG['n_modes'], (32, 32))
        self.assertEqual(_nscfg_TFNO_CP_CONFIG['factorization'], 'cp')
        self.assertEqual(_nscfg_TFNO_CP_CONFIG['rank'], 0.05)
        self.assertEqual((_nscfg_AMFNO_CONFIG['width'], _nscfg_AMFNO_CONFIG['n1'], _nscfg_AMFNO_CONFIG['n2']), (64, 32, 32))
        self.assertEqual(_nscfg_UFNO_CONFIG['hidden_channels'], 32)
        self.assertEqual(_nscfg_UFNO_CONFIG['n_layers'], 4)
        self.assertEqual(_nscfg_UFNO_CONFIG['use_unet_from'], 2)

    def test_official_siren_variants_keep_variant_specific_dimensions(self) -> None:
        self.assertNotIn('hidden_dim', _nscfg_SIRENFNO_COMMON_CONFIG)
        self.assertNotIn('siren_dim_in', _nscfg_SIRENFNO_COMMON_CONFIG)
        expected = {'sirenfno': (64, 32, 'dense', None), 'cpsirenfno': (64, 32, 'cp', 16), 'ttsirenfno': (32, 32, 'tt', 16), 'tuckersirenfno': (32, 16, 'tucker', 16)}
        for name, values in expected.items():
            with self.subTest(model=name):
                config = _nscfg_model_constructor_kwargs(name)
                actual = (config['hidden_dim'], config['siren_dim_in'], config['factorization'], config.get('rank'))
                self.assertEqual(actual, values)

    def test_cafe_uses_approved_fixed_sigma_config_and_ns_ranks(self) -> None:
        self.assertEqual(_nscfg_CAFEPLUSFNO_COMMON_CONFIG, APPROVED_CAFE_CONFIG)
        self.assertFalse(_nscfg_CAFEPLUSFNO_COMMON_CONFIG['learnable_sigma'])
        self.assertEqual(_nscfg_CAFEPLUSFNO_COMMON_CONFIG['sigma_init'], 1.0)
        expected_factor_common = {'factor_rff_basis': 16, 'factor_cheb_basis': 8, 'factor_branch_dim': 12, 'factor_output_init_mode': 'xavier', 'factor_kernel_target_rms': 0.001}
        self.assertEqual(_nscfg_CAFEPLUSFNO_FACTOR_COMMON_CONFIG, expected_factor_common)
        self.assertTrue(expected_factor_common.keys().isdisjoint(_nscfg_CAFEPLUSFNO_COMMON_CONFIG))
        self.assertEqual(_nscfg_CAFEPLUSFNO_FACTORIZATIONS, {'cafe_plus_fno': {'factorization': 'dense', 'rank': 16, 'cafe_branch_dim': 32, 'kernel_hidden_dim': 64}, 'cp_cafe_plus_fno': {**expected_factor_common, 'factorization': 'cp', 'rank': 16, 'factor_hidden_dim': 16}, 'tt_cafe_plus_fno': {**expected_factor_common, 'factorization': 'tt', 'rank': 16, 'factor_hidden_dim': 24}, 'tucker_cafe_plus_fno': {**expected_factor_common, 'factorization': 'tucker', 'rank': 16, 'factor_hidden_dim': 16}})
        dense = _nscfg_model_constructor_kwargs('cafe_plus_fno')
        self.assertTrue(expected_factor_common.keys().isdisjoint(dense))
        self.assertNotIn('factor_hidden_dim', dense)
        self.assertEqual((dense['cafe_branch_dim'], dense['kernel_hidden_dim']), (32, 64))
        built = train_ns_module.build_model('cafe_plus_fno', torch.device('cpu'))
        generator = built.model.operator_layers[0].spectral.kernel_generator
        self.assertEqual((built.configuration['cafe_branch_dim'], built.configuration['kernel_hidden_dim']), (32, 64))
        self.assertEqual((generator.encoder.branch_dim, generator.head.joint_mlp.input_dim, generator.head.joint_mlp.hidden_dim), (32, 32, 64))
        for name in ('cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'):
            with self.subTest(model=name):
                configured = _nscfg_model_constructor_kwargs(name)
                self.assertTrue(expected_factor_common.keys() <= configured.keys())
                self.assertIn('factor_hidden_dim', configured)

    def test_constructor_configs_are_independent(self) -> None:
        first = _nscfg_model_constructor_kwargs('cp_cafe_plus_fno')
        first['width'] = -1
        self.assertEqual(_nscfg_model_constructor_kwargs('cp_cafe_plus_fno')['width'], 32)

    def test_cli_defaults_and_required_model(self) -> None:
        parsed = _nscfg_parse_args(['--model', 'fno'])
        self.assertEqual(parsed.seed, _nscfg_CANONICAL_SEED)
        self.assertEqual(parsed.data_root, Path('data'))
        self.assertEqual(parsed.results_root, Path('results') / 'ns128_sirenfno_81918ec')
        with self.assertRaises(SystemExit):
            _nscfg_parse_args([])

    def test_offline_loader_exactly_matches_official_construction(self) -> None:
        processor = object()
        train_db = TensorDataset(torch.arange(4))
        test_db = TensorDataset(torch.arange(2))
        dataset = type('DatasetFixture', (), {'train_db': train_db, 'test_dbs': {128: test_db}, 'data_processor': processor})()
        with patch.object(train_ns_module, 'expected_ns_files', return_value=()), patch.object(train_ns_module, 'NavierStokesDataset', return_value=dataset) as mocked_dataset:
            train_loader, test_loaders, actual_processor = load_ns(Path('unused-data-root'))
        mocked_dataset.assert_called_once_with(root_dir='unused-data-root', n_train=1000, n_tests=[200], batch_size=32, test_batch_sizes=[32], train_resolution=128, test_resolutions=[128], encode_input=True, encode_output=True, encoding='channel-wise', channel_dim=1, subsampling_rate=None, download=False)
        self.assertIs(train_loader.dataset, train_db)
        self.assertIs(test_loaders[128].dataset, test_db)
        self.assertIs(actual_processor, processor)
        for loader in (train_loader, test_loaders[128]):
            self.assertEqual(loader.batch_size, 32)
            self.assertEqual(loader.num_workers, 0)
            self.assertTrue(loader.pin_memory)
            self.assertFalse(loader.persistent_workers)
            self.assertFalse(loader.drop_last)
            self.assertIsInstance(loader.sampler, SequentialSampler)
            self.assertIsNone(loader.generator)
            self.assertIsNone(loader.worker_init_fn)
            self.assertIsNone(loader.prefetch_factor)

    def test_seed_order_and_no_private_rff_seed(self) -> None:
        source = inspect.getsource(train_ns_module.main)
        darcy_source = inspect.getsource(train_darcy_module.main)
        self.assertEqual(source.count('set_seed(args.seed)'), 1)
        self.assertEqual(darcy_source.count('set_seed(args.seed)'), 1)
        self.assertLess(source.index('verify_ns_dataset(data_root)'), source.index('set_seed(args.seed)'))
        self.assertLess(source.index('set_seed(args.seed)'), source.index('load_ns(data_root)'))
        self.assertLess(source.index('set_seed(args.seed)'), source.index('build_model(args.model, device)'))
        self.assertLess(darcy_source.index('verify_dataset("darcy128", data_root)'), darcy_source.index('set_seed(args.seed)'))
        self.assertLess(darcy_source.index('set_seed(args.seed)'), darcy_source.index('load_darcy(data_root)'))
        self.assertLess(darcy_source.index('load_darcy(data_root)'), darcy_source.index('build_model(args.model, device)'))
        for forbidden in ('torch.Generator(', 'manual_seed(', 'torch.seed(', 'rff_seed='):
            self.assertNotIn(forbidden, source)

    def test_rff_metadata_matches_every_model_family(self) -> None:
        for name in _nscfg_MODEL_CHOICES:
            with self.subTest(model=name):
                metadata = _nscfg_rff_metadata(name)
                if 'sirenfno' in name or 'cafe_plus_fno' in name:
                    self.assertEqual(metadata['rff_rng_policy'], 'global_torch_rng')
                    self.assertEqual(metadata['rff_seed_source'], 'global_experiment_seed')
                else:
                    self.assertEqual(metadata['rff_rng_policy'], 'not_applicable')
                    self.assertEqual(metadata['rff_seed_source'], 'not_applicable')

    def test_csv_logger_delegates_each_numerical_method_once(self) -> None:
        train_source = inspect.getsource(CsvLoggingTrainer.train_one_epoch)
        eval_source = inspect.getsource(CsvLoggingTrainer.evaluate_all)
        self.assertEqual(train_source.count('super().train_one_epoch('), 1)
        self.assertEqual(eval_source.count('super().evaluate_all('), 1)

    def test_output_paths_are_one_model_by_one_seed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = _nscfg_prepare_run_paths(root, 'fno', 42, overwrite=False)
            self.assertEqual(paths[0], root / 'fno' / 'seed_42')
            self.assertEqual(tuple((path.name for path in paths[1:])), ('training_log.csv', 'summary.json', 'final_checkpoint.pt'))

# Migrated from tests/test_reacdiff_config.py




from experiments import train_reacdiff as _rdcfg_train_module

from experiments.configs.reacdiff1d import AMFNO_CONFIG as _rdcfg_AMFNO_CONFIG, BATCH_SIZE as _rdcfg_BATCH_SIZE, CAFEPLUSFNO_COMMON_CONFIG as _rdcfg_CAFEPLUSFNO_COMMON_CONFIG, CAFEPLUSFNO_FACTOR_COMMON_CONFIG as _rdcfg_CAFEPLUSFNO_FACTOR_COMMON_CONFIG, CAFEPLUSFNO_FACTORIZATIONS as _rdcfg_CAFEPLUSFNO_FACTORIZATIONS, EPOCHS as _rdcfg_EPOCHS, EXPECTED_CAFE_PARAMETER_COUNTS, EXPECTED_PINNED_PARAMETER_COUNTS as _rdcfg_EXPECTED_PINNED_PARAMETER_COUNTS, EXPECTED_TIME_STEPS as _rdcfg_EXPECTED_TIME_STEPS, FNO_CONFIG as _rdcfg_FNO_CONFIG, INPUT_STEPS as _rdcfg_INPUT_STEPS, LEARNING_RATE as _rdcfg_LEARNING_RATE, MODEL_CHOICES as _rdcfg_MODEL_CHOICES, NORMALIZATION_POLICY as _rdcfg_NORMALIZATION_POLICY, N_TEST as _rdcfg_N_TEST, N_TRAIN as _rdcfg_N_TRAIN, N_VAL as _rdcfg_N_VAL, REACDIFF_SOURCE_BLOBS, RESOLUTION as _rdcfg_RESOLUTION, ROLLOUT as _rdcfg_ROLLOUT, SCHEDULER_T_MAX as _rdcfg_SCHEDULER_T_MAX, SIRENFNO_COMMON_CONFIG as _rdcfg_SIRENFNO_COMMON_CONFIG, SIRENFNO_FACTORIZATIONS as _rdcfg_SIRENFNO_FACTORIZATIONS, TFNO_CP_CONFIG as _rdcfg_TFNO_CP_CONFIG, UFNO_CONFIG as _rdcfg_UFNO_CONFIG, WEIGHT_DECAY as _rdcfg_WEIGHT_DECAY, model_constructor_kwargs as _rdcfg_model_constructor_kwargs



class ReacDiffConfigurationTests(unittest.TestCase):

    def test_released_data_and_training_protocol(self) -> None:
        self.assertEqual((_rdcfg_RESOLUTION, _rdcfg_EXPECTED_TIME_STEPS), (1024, 101))
        self.assertEqual((_rdcfg_INPUT_STEPS, _rdcfg_ROLLOUT), (10, 10))
        self.assertEqual((_rdcfg_N_TRAIN, _rdcfg_N_VAL, _rdcfg_N_TEST), (1000, 0, 200))
        self.assertEqual((_rdcfg_BATCH_SIZE, _rdcfg_EPOCHS), (32, 500))
        self.assertEqual((_rdcfg_LEARNING_RATE, _rdcfg_WEIGHT_DECAY), (0.001, 0.0001))
        self.assertEqual(_rdcfg_SCHEDULER_T_MAX, 500)
        self.assertEqual(_rdcfg_NORMALIZATION_POLICY, 'none')

    def test_model_roster_is_exactly_twelve(self) -> None:
        self.assertEqual(_rdcfg_MODEL_CHOICES, ('fno', 'ufno', 'tfno_cp', 'amfno', 'sirenfno', 'cpsirenfno', 'ttsirenfno', 'tuckersirenfno', 'cafe_plus_fno', 'cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'))

    def test_released_constructor_configs_are_exact(self) -> None:
        self.assertEqual(_rdcfg_FNO_CONFIG, {'n_modes_height': 1024, 'hidden_channels': 32, 'in_channels': 10, 'out_channels': 1, 'n_layers': 4, 'lifting_channels': 64, 'projection_channels': 64})
        self.assertEqual(_rdcfg_TFNO_CP_CONFIG['factorization'], 'cp')
        self.assertEqual(_rdcfg_TFNO_CP_CONFIG['rank'], 0.05)
        self.assertEqual(_rdcfg_AMFNO_CONFIG, {'width': 64, 'n1': 10, 'padding': 0, 'input_dim': 10, 'output_dim': 1, 'mlp_dropout': 0})
        self.assertEqual(_rdcfg_UFNO_CONFIG, {'n_modes_height': 32, 'hidden_channels': 64, 'in_channels': 10, 'out_channels': 1, 'n_layers': 6, 'lifting_channels': 128, 'projection_channels': 128, 'positional_embedding': 'grid'})
        self.assertEqual(_rdcfg_SIRENFNO_COMMON_CONFIG['siren_dim_in'], 16)
        self.assertEqual(_rdcfg_SIRENFNO_COMMON_CONFIG['hidden_dim'], 32)
        self.assertEqual(_rdcfg_SIRENFNO_COMMON_CONFIG['omega'], 15.0)
        self.assertEqual(_rdcfg_SIRENFNO_COMMON_CONFIG['ff_sigma'], 256)
        self.assertEqual({name: (item['factorization'], item.get('rank')) for name, item in _rdcfg_SIRENFNO_FACTORIZATIONS.items()}, {'sirenfno': ('dense', None), 'cpsirenfno': ('cp', 8), 'ttsirenfno': ('tt', 8), 'tuckersirenfno': ('tucker', 8)})

    def test_cafe_policy_and_counts_are_precommitted(self) -> None:
        self.assertEqual(_rdcfg_CAFEPLUSFNO_COMMON_CONFIG['input_dim'], 10)
        self.assertFalse(_rdcfg_CAFEPLUSFNO_COMMON_CONFIG['learnable_sigma'])
        self.assertEqual(_rdcfg_CAFEPLUSFNO_FACTOR_COMMON_CONFIG, {'factor_rff_basis': 16, 'factor_cheb_basis': 8, 'factor_branch_dim': 12, 'factor_output_init_mode': 'xavier', 'factor_kernel_target_rms': 0.001})
        self.assertEqual(EXPECTED_CAFE_PARAMETER_COUNTS, {'cafe_plus_fno': 323201, 'cp_cafe_plus_fno': 70149, 'tt_cafe_plus_fno': 85381, 'tucker_cafe_plus_fno': 74245})
        self.assertTrue(all((_rdcfg_CAFEPLUSFNO_FACTORIZATIONS[name]['rank'] == 8 for name in ('cp_cafe_plus_fno', 'tt_cafe_plus_fno', 'tucker_cafe_plus_fno'))))

    def test_exact_released_counts_are_pinned(self) -> None:
        self.assertEqual(_rdcfg_EXPECTED_PINNED_PARAMETER_COUNTS, {'fno': 4216161, 'ufno': 1892161, 'tfno_cp': 226369, 'amfno': 823073, 'sirenfno': 304833, 'cpsirenfno': 57665, 'ttsirenfno': 72001, 'tuckersirenfno': 61761})

    def test_runtime_blob_ids_match_pinned_checkout(self) -> None:
        self.assertEqual(
            verify_pinned_source_blobs(REACDIFF_SOURCE_BLOBS),
            REACDIFF_SOURCE_BLOBS,
        )

    def test_seed_order_and_no_private_reseed(self) -> None:
        source = inspect.getsource(_rdcfg_train_module.main)
        self.assertEqual(source.count('set_seed(args.seed)'), 1)
        self.assertLess(source.index('verify_reacdiff_dataset(data_root)'), source.index('set_seed(args.seed)'))
        self.assertLess(source.index('set_seed(args.seed)'), source.index('load_reacdiff(data_root)'))
        self.assertLess(source.index('load_reacdiff(data_root)'), source.index('build_model(args.model, device)'))
        for forbidden in ('manual_seed(', 'torch.seed(', 'torch.Generator(', 'rff_seed='):
            self.assertNotIn(forbidden, source)

    def test_sweep_configuration_is_guarded_60_runs(self) -> None:
        spec = DATASETS['reacdiff1024']
        self.assertEqual(spec['models'], _rdcfg_MODEL_CHOICES)
        self.assertEqual(spec['experiment'], 'reacdiff1024_released_sirenfno')
        self.assertIs(spec['require_current_source_commit'], True)
        self.assertEqual(len(spec['models']) * 5, 60)
        self.assertEqual(spec['default_results'], Path('results') / 'reacdiff1024_sirenfno_81918ec')

    def test_constructor_configs_are_independent(self) -> None:
        first = _rdcfg_model_constructor_kwargs('tt_cafe_plus_fno')
        first['width'] = -1
        self.assertEqual(_rdcfg_model_constructor_kwargs('tt_cafe_plus_fno')['width'], 32)
