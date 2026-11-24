#!/usr/bin/env python3
"""Real native-FSDP inference/master/export boundary regression (one process).

NPU execution must receive one explicit ASCEND_RT_VISIBLE_DEVICES integer before
Python starts and must enumerate exactly one device; only logical npu:0 is used.
The caller chooses the assigned physical device. This script never changes device
visibility. The CPU path exercises native FSDP without Turbo.
"""
import argparse
from datetime import timedelta
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import traceback

os.environ.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
import torch
import torch.distributed as dist


TESTED_REGRESSION_SOURCE_SHA256 = '72d95e1fabc079b552179b9492da7d000e98ecff030c2179fcdc2eed64151c8d'


def single_visible_npu(environment):
    value = environment.get('ASCEND_RT_VISIBLE_DEVICES', '')
    if not value.isascii() or not value.isdecimal():
        raise RuntimeError('Set ASCEND_RT_VISIBLE_DEVICES to exactly one assigned device integer before Python starts')
    return int(value)


def cpu(value):
    value = value.detach()
    if hasattr(value, 'to_local'):
        value = value.to_local()
    return value.to('cpu', copy=True).contiguous()


def digest(value):
    return hashlib.sha256(memoryview(cpu(value).reshape(-1).view(torch.uint8).numpy())).hexdigest()


def write(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    os.replace(temporary, path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'npu'], required=True)
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--export-module', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    result = {'passed': False, 'device_mode': args.device, 'torch': torch.__version__,
              'scope': 'Real one-device native-FSDP boundary regression; not eight-rank Slime training',
              'tested_regression_source_sha256': TESTED_REGRESSION_SOURCE_SHA256,
              'executed_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    torch.set_num_threads(1)
    spec = importlib.util.spec_from_file_location('boundary_export', args.export_module)
    export = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(export)
    sys.path.insert(0, str(args.repo / 'slime'))
    from slime.backends.fsdp2_utils.loss import backward_scale, grpo_contribution, response_log_probs
    from transformers import AutoModelForCausalLM, Qwen3Config, Qwen3ForCausalLM
    config = json.loads(args.config.read_text())
    initialized = False
    try:
        if args.device == 'npu':
            result['caller_selected_visible_device'] = single_visible_npu(os.environ)
            import torch_npu  # noqa: F401
            from fsdp_turbo.utils.device import set_accelerator_compatible
            count = torch.npu.device_count()
            result.update(visible=os.environ.get('ASCEND_RT_VISIBLE_DEVICES'), enumerated_devices=count)
            if count != 1:
                raise RuntimeError('The explicitly single-device process must enumerate exactly one logical device')
            if set_accelerator_compatible() != 'hccl':
                raise RuntimeError('Expected the pinned Ascend runtime')
            torch.npu.set_device(0)
            device = torch.device('npu', 0)
            backend = 'cpu:gloo,npu:hccl'
        else:
            device, backend = torch.device('cpu'), 'gloo'
        dist.init_process_group(backend, init_method='file://' + str((args.output / 'rendezvous').resolve()),
                                rank=0, world_size=1, timeout=timedelta(seconds=120))
        initialized = True
        torch.manual_seed(42)
        hf_config = Qwen3Config.from_dict(config)
        hf_config._attn_implementation = 'eager'
        hf_config.use_cache = False
        hf = Qwen3ForCausalLM(hf_config).float().eval()
        with torch.no_grad():
            for name, parameter in hf.named_parameters():
                if name.endswith('norm.weight'):
                    parameter.copy_(torch.linspace(.9137, 1.0873, parameter.numel()).reshape(parameter.shape))
        initial = {name: cpu(value) for name, value in hf.named_parameters()}
        assert len(initial) == 25
        norm = initial['model.norm.weight']
        assert digest(norm) != digest(norm.bfloat16().float()), 'Test norm must expose BF16 rounding'
        if args.device == 'npu':
            from fsdp_turbo.fsdp_turbo import FSDPTurbo
            from fsdp_turbo.fsdp_turbo_config import DistributedConfig, FSDPPlanConfig, FSDPTurboConfig, MemoryConfig
            plan = FSDPTurboConfig(distributed=DistributedConfig(fully_shard_parallel_size=1,
                tensor_parallel_size=1, expert_parallel_size=1, fsdp_plan=FSDPPlanConfig(
                    apply_modules={'model.layers.{*}': {}, 'model.embed_tokens': {}, 'lm_head': {}},
                    param_dtype='bf16', reduce_dtype='fp32', output_dtype='bf16', fsdp_implementation='native',
                    num_to_forward_prefetch=0, num_to_backward_prefetch=0)), memory=MemoryConfig(recompute=False))
            wrapped = FSDPTurbo(plan, hf)
            model = wrapped.model
        else:
            from torch.distributed.device_mesh import init_device_mesh
            from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard
            mesh = init_device_mesh('cpu', (1, 1), mesh_dim_names=('dp', 'fsdp'))
            policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32,
                                          output_dtype=torch.bfloat16)
            for module in [*hf.model.layers, hf.model.embed_tokens, hf.lm_head]:
                fully_shard(module, mesh=mesh, mp_policy=policy)
            model = fully_shard(hf, mesh=mesh, mp_policy=policy)
        from torch.distributed.checkpoint.state_dict import StateDictOptions, get_model_state_dict
        from torch.distributed.tensor import DTensor
        masters = dict(model.named_parameters())
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0., foreach=False)
        tokens = torch.tensor([[1, 2, 3, 4, 5, 6], [1, 2, 3, 7, 8, 9]], device=device)
        def forward():
            logits = model(input_ids=tokens, use_cache=False).logits
            return response_log_probs(logits, tokens, [6, 6], [3, 3], .7)
        with torch.no_grad():
            forward()
        view = model.model.norm.weight
        result['after_no_grad_visible_norm'] = {'dtype': str(view.dtype), 'is_dtensor': isinstance(view, DTensor),
                                               'same_master_parameter_object': view is masters['model.norm.weight']}
        legacy = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
        result['legacy_getter_norm'] = {'dtype': str(legacy['model.norm.weight'].dtype),
                                       'master_bytes_preserved': digest(legacy['model.norm.weight']) == digest(norm)}
        del legacy
        with torch.no_grad():
            old = [value.detach().cpu() for value in forward()]
        result['restored_before_current'] = export.reshard_to_master_shards(model)
        if any(parameter is not masters[name] or digest(parameter) != digest(initial[name])
               for name, parameter in model.named_parameters()):
            raise AssertionError('Public reshard did not restore the original master objects and bytes')
        exports = []
        def check_export(version):
            with torch.no_grad():
                probe = model(input_ids=tokens[:1, :4], use_cache=False).logits[0, -1, :16].float().cpu().tolist()
            restore = export.reshard_to_master_shards(model)
            live = {name: cpu(value) for name, value in model.named_parameters()}
            state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
            if set(state) != set(live) or any(digest(state[name]) != digest(value) for name, value in live.items()):
                raise AssertionError('Gathered state changed real master parameter bytes')
            publication = export.write_export(model, state, args.output / 'weights', version,
                {'input_ids': tokens[:1, :4].cpu().tolist(), 'logits': probe, 'forward_dtype': 'torch.bfloat16',
                 'atol': .02, 'rtol': .02}, {'diagnostic': True})
            receipt = export.verify_exported_model(**publication)
            cold = AutoModelForCausalLM.from_pretrained(publication['model_path'], dtype=torch.float32,
                                                      local_files_only=True, attn_implementation='eager')
            loaded = cold.state_dict()
            if set(loaded) != set(live) or any(digest(loaded[name]) != digest(value) for name, value in live.items()):
                raise AssertionError('HF save/cold-load changed real master bytes')
            exports.append({'version': version, 'all_25_master_bytes_equal': True, 'restore': restore,
                            'norm_sha256': digest(live['model.norm.weight']), 'receipt_verified': receipt['verified']})
        check_export(0)
        # A second independent no-grad snapshot exercises the exact train boundary.
        with torch.no_grad():
            old = [value.detach().cpu() for value in forward()]
        export.reshard_to_master_shards(model)
        current = forward()
        result['snapshot_current_logprobs_equal'] = all(torch.equal(cpu(a), cpu(b)) for a, b in zip(current, old, strict=True))
        if not result['snapshot_current_logprobs_equal']:
            raise AssertionError('Reshard boundary changed the step-start policy logprobs')
        contribution, _ = grpo_contribution(current, old, [1., -1.], [[1, 1, 1], [1, 1, 0]], [3., 2.])
        (contribution * backward_scale(1, 2)).backward()
        for name, parameter in model.named_parameters():
            if parameter.grad is None or parameter.grad.dtype != torch.float32 or not torch.isfinite(cpu(parameter.grad)).all():
                raise AssertionError('Invalid actual FP32 gradient: ' + name)
        optimizer.step()
        result['norm_update_nonzero'] = digest(masters['model.norm.weight']) != digest(norm)
        if not result['norm_update_nonzero']:
            raise AssertionError('The regression did not update the non-representable FP32 norm')
        check_export(1)
        result.update(passed=True, parameter_count=25, exports=exports,
                      final_all_master_objects_retained=all(parameter is masters[name] for name, parameter in model.named_parameters()))
        if args.device == 'npu':
            result['actual_logical_current_device'] = torch.npu.current_device()
        write(args.output / 'summary.json', result)
        print(json.dumps(result, indent=2), flush=True)
    except Exception as error:
        result.update(error=repr(error), traceback=traceback.format_exc())
        write(args.output / 'summary.json', result)
        raise
    finally:
        if initialized:
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
