"""Explicit future CUDA arithmetic or Darcy official loader/evaluator smoke.

No training, download, checkpoint, or formal result is produced. This command
must be invoked deliberately after the user is ready to use a device/data copy.
"""
from __future__ import annotations
import importlib
import json
import math
import stat
from pathlib import Path

from .contracts import validate_condition_variant
from .contracts import ABLATION_CONDITIONS, DATASETS, MODEL_VARIANTS, seed_value
from .public import SafeArgumentParser, PublicError, emit_json, public_main, public_payload


def parse_args(argv=None):
    parser = SafeArgumentParser(description=__doc__)
    parser.add_argument('--kind', required=True, choices=('cuda', 'official-data'))
    parser.add_argument('--dataset', required=True, choices=sorted(DATASETS))
    parser.add_argument('--model-variant', required=True, choices=sorted(MODEL_VARIANTS))
    parser.add_argument('--condition', required=True, choices=sorted(ABLATION_CONDITIONS))
    parser.add_argument('--seed', required=True, type=seed_value)
    parser.add_argument('--device', required=True, choices=('cpu', 'cuda'))
    parser.add_argument('--data-root', type=Path)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        validate_condition_variant(args.condition, args.model_variant)
    except ValueError as exc:
        parser.error(str(exc))
    if args.kind == 'cuda' and (args.device != 'cuda' or args.data_root is not None):
        parser.error('CUDA_SMOKE_REQUIRES_CUDA_AND_NO_DATA_ROOT')
    if args.kind == 'official-data' and (args.dataset != 'darcy' or args.data_root is None):
        parser.error('OFFICIAL_LOADER_SMOKE_SUPPORTS_DARCY_WITH_DATA_ROOT')
    return args


def _preflight_official_files(paths, data_root):
    """Check each required independent regular file before its verifier reads."""
    from .runtime import contained
    for path in paths:
        checked = contained(path, data_root)
        info = checked.lstat()
        if not stat.S_ISREG(info.st_mode) or info.st_nlink != 1:
            raise PublicError('OFFICIAL_SMOKE_SHARED_OR_NONREGULAR_FILE')


def main(argv=None):
    args = parse_args(argv)
    from .repository import resolve_repository_root, source_state, activate_repository, import_audit
    from .runtime import configure_runtime, apply_torch_runtime, output_path, workspace_root, contained
    root = resolve_repository_root()
    activate_repository(root)
    source = source_state(root, formal=True)
    report_path = output_path(args.report)
    if report_path.exists():
        raise PublicError('SMOKE_REPORT_EXISTS')
    # This is deliberately a diagnostic numerical policy and cannot write a
    # formal summary. Device use is explicit and never inferred from a wheel.
    configure_runtime(cpu=args.device == 'cpu', profile='diagnostic', threads=1, cache_area='results')
    import torch
    from .models import build_ablation_model, describe_model
    apply_torch_runtime()
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise PublicError('CUDA_SMOKE_DEVICE_UNAVAILABLE')
    spec = DATASETS[args.dataset]
    module = importlib.import_module(spec.train_module)
    module.set_seed(args.seed)
    built = build_ablation_model(args.dataset, args.model_variant, args.condition, device=args.device)
    record = {'status': 'PASS', 'result_kind': 'diagnostic_smoke', 'kind': args.kind,
        'dataset': args.dataset, 'model_variant': args.model_variant, 'condition': args.condition,
        'seed': args.seed, 'device': args.device, 'source': source, 'configuration': built.configuration,
        'model': describe_model(built.model), 'training': 'NOT RUN', 'formal_result': False}
    if args.kind == 'cuda':
        config = built.model.get_config()
        spatial = (17,) if spec.spatial_dim == 1 else (7, 9)
        channels = int(config['input_dim'])
        shape = ((1, *spatial, channels) if config['input_layout'] == 'channels_last'
                 else (1, channels, *spatial))
        x = torch.rand(shape, device='cuda') + 0.5
        prediction = built.model(x)
        loss = prediction.square().mean()
        if not torch.isfinite(loss).all().item():
            raise PublicError('CUDA_SMOKE_NONFINITE')
        loss.backward()
        gradients = [p.grad for p in built.model.parameters() if p.grad is not None]
        if (not gradients or not torch.isfinite(prediction).all().item()
                or not all(torch.isfinite(g).all().item() for g in gradients)):
            raise PublicError('CUDA_SMOKE_NONFINITE')
        torch.cuda.synchronize()
        record.update(input_shape=list(x.shape), output_shape=list(prediction.shape),
                      finite_forward_backward=True, official_data='NOT RUN',
                      loss_scope='synthetic squared-output arithmetic smoke; not native training')
    else:
        from torch.utils.data import DataLoader, Subset
        from .train import _verify_dataset
        data_root = output_path(contained(args.data_root, workspace_root()))
        # Inspect each required file without following links before the parent
        # verifier reads it; the native loader explicitly uses download=False.
        _preflight_official_files(module.expected_darcy_files(data_root), data_root)
        identity = _verify_dataset(module, spec, data_root)
        train_loader, test_loaders, processor = module.load_darcy(data_root)
        if len(test_loaders) != 1:
            raise PublicError('OFFICIAL_SMOKE_TEST_SPLIT_INVALID')
        native_loader = next(iter(test_loaders.values()))
        if not len(native_loader.dataset):
            raise PublicError('OFFICIAL_SMOKE_EMPTY_DATASET')
        loader = DataLoader(Subset(native_loader.dataset, [0]), batch_size=1, shuffle=False,
                            num_workers=0, pin_memory=False, drop_last=False)
        trainer = module.Trainer(model=built.model, n_epochs=1, device=args.device,
            data_processor=processor.to(args.device), wandb_log=False, use_distributed=False, verbose=False)
        metrics = trainer.evaluate({'l2': module.LpLoss(d=2, p=2), 'h1': module.H1Loss(d=2)},
                                   loader, log_prefix='official_smoke')
        metrics = {key: float(value.detach().cpu().item() if torch.is_tensor(value) else value)
                   for key, value in metrics.items()}
        if not metrics or not all(math.isfinite(value) for value in metrics.values()):
            raise PublicError('OFFICIAL_SMOKE_NONFINITE')
        record.update(dataset_identity=identity, metrics=metrics, evaluated_samples=1,
                      loaded_training_samples=len(train_loader.dataset),
                      loaded_test_samples=len(native_loader.dataset), evaluation_decoded=True,
                      weights='seeded initialization; not a trained checkpoint',
                      loader_scope='native complete file load and normalization; evaluation limited to one sample')
    record['imports'] = import_audit(root)
    from .runtime_metadata import capture_runtime_metadata
    record['runtime'] = capture_runtime_metadata(profile='diagnostic', device=args.device, seed=args.seed)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open('x', encoding='utf-8') as handle:
        json.dump(public_payload(record), handle, indent=2, sort_keys=True)
        handle.write('\n')
    emit_json({'status': record['status'], 'kind': args.kind, 'dataset': args.dataset,
               'device': args.device, 'result_kind': record['result_kind']})


if __name__ == '__main__':
    public_main(main)
