"""
Pipeline Dataflow Framework
基于数据流动方式的模型层级流水线切分框架

将模型的 forward() 拆解为若干 PipelineStage，
每个阶段声明「输入 key → 输出 key」的依赖关系，
DataFlowPipeline 编排执行顺序，支持：
  - 顺序执行 (调试/单卡)
  - Profile (每阶段计时/显存)
  - 中间结果检查/断点续推
  - 多卡流水线并行 (micro-batch)
  - 流式执行 (每阶段完成后回调，用于 API 增量响应)
  - 中间状态序列化/反序列化 (用于断点续推和 API 传输)
"""

import time
import base64
import io
import numpy as np
import torch
import torch.nn as nn
from datetime import datetime
from typing import Dict, List, Optional, Callable, Any, Tuple, Generator


# ============================================================
# 1. DataRegistry — 全局数据注册表
# ============================================================

class DataRegistry:
    """
    中间数据注册表
    
    所有阶段产出的张量都注册到全局字典中，
    后续阶段通过 key 读取所需输入。
    """

    def __init__(self):
        self._data: Dict[str, Any] = {}
        self._meta: Dict[str, dict] = {}

    def register(self, key: str, value: Any, source: str = 'unknown'):
        self._data[key] = value
        meta = {'source': source}
        if isinstance(value, torch.Tensor):
            meta['shape'] = tuple(value.shape)
            meta['dtype'] = str(value.dtype)
            meta['device'] = str(value.device)
            meta['numel'] = value.numel()
        elif isinstance(value, (list, tuple)):
            meta['length'] = len(value)
            if value and isinstance(value[0], torch.Tensor):
                meta['element_shape'] = tuple(value[0].shape)
        self._meta[key] = meta

    def get(self, key: str) -> Any:
        if key not in self._data:
            raise KeyError(f"DataRegistry: key '{key}' not found. Available: {list(self._data.keys())}")
        return self._data[key]

    def has(self, key: str) -> bool:
        return key in self._data

    def keys(self) -> List[str]:
        return list(self._data.keys())

    def meta(self, key: str) -> dict:
        return self._meta.get(key, {})

    def clear(self):
        self._data.clear()
        self._meta.clear()

    def to_dict(self) -> Dict[str, Any]:
        return dict(self._data)

    def summary(self) -> str:
        lines = ["DataRegistry Summary:"]
        lines.append(f"  Total keys: {len(self._data)}")
        total_memory = 0
        for key, meta in self._meta.items():
            shape_str = str(meta.get('shape', '?'))
            numel = meta.get('numel', 0)
            mem_mb = numel * 4 / 1024 / 1024 if numel else 0
            total_memory += mem_mb
            lines.append(f"  {key:30s} shape={shape_str:30s} "
                         f"src={meta.get('source', '?'):15s} "
                         f"mem~{mem_mb:.2f}MB")
        lines.append(f"  Total estimated memory: {total_memory:.2f}MB")
        return '\n'.join(lines)


# ============================================================
# 2. PipelineStage — 流水线阶段
# ============================================================

class PipelineStage:
    """
    流水线的一个阶段

    每个阶段包含:
    - name: 阶段名称
    - module: 该阶段包含的 nn.Module (可为 None)
    - input_keys: 需要从 DataRegistry 读取的 key 列表
    - output_keys: 产出到 DataRegistry 的 key 列表
    - execute_fn: 自定义执行函数 (覆盖默认的 module forward)
    - description: 阶段描述
    - api_response_fn: 阶段完成后生成 API 响应数据的函数
    """

    def __init__(
        self,
        name: str,
        module: Optional[nn.Module] = None,
        input_keys: List[str] = None,
        output_keys: List[str] = None,
        execute_fn: Optional[Callable] = None,
        description: str = "",
        device: Optional[str] = None,
        api_response_fn: Optional[Callable] = None,
    ):
        self.name = name
        self.module = module
        self.input_keys = input_keys or []
        self.output_keys = output_keys or []
        self.execute_fn = execute_fn
        self.description = description
        self.device = device
        self.api_response_fn = api_response_fn

        # Profile 统计
        self._profile = {
            'call_count': 0,
            'total_time_ms': 0.0,
            'total_memory_mb': 0.0,
        }

    def execute(self, registry: DataRegistry) -> DataRegistry:
        inputs = {key: registry.get(key) for key in self.input_keys}

        if self.device and self.module is not None:
            self.module.to(self.device)
            inputs = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in inputs.items()
            }

        if self.execute_fn is not None:
            outputs = self.execute_fn(**inputs)
        elif self.module is not None:
            input_values = [inputs[k] for k in self.input_keys]
            outputs = self.module(*input_values)
        else:
            raise ValueError(f"Stage '{self.name}': must provide either execute_fn or module")

        # 注册输出
        if isinstance(outputs, dict):
            for key in self.output_keys:
                if key in outputs:
                    registry.register(key, outputs[key], source=self.name)
        elif isinstance(outputs, (tuple, list)):
            for key, val in zip(self.output_keys, outputs):
                registry.register(key, val, source=self.name)
        else:
            if len(self.output_keys) == 1:
                registry.register(self.output_keys[0], outputs, source=self.name)
            else:
                raise ValueError(f"Stage '{self.name}': output is single value but has {len(self.output_keys)} output_keys")

        return registry

    def execute_with_profile(self, registry: DataRegistry) -> DataRegistry:
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        mem_before = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        registry = self.execute(registry)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        mem_after = torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0

        self._profile['call_count'] += 1
        self._profile['total_time_ms'] += (t1 - t0) * 1000
        self._profile['total_memory_mb'] += max(0, mem_after - mem_before)
        return registry

    @property
    def profile(self) -> dict:
        avg_time = self._profile['total_time_ms'] / max(self._profile['call_count'], 1)
        return {**self._profile, 'avg_time_ms': avg_time}

    def param_count(self) -> int:
        if self.module is None:
            return 0
        return sum(p.numel() for p in self.module.parameters())

    def build_api_response(self, registry: DataRegistry) -> Dict[str, Any]:
        """构建该阶段的 API 响应数据。

        如果设置了 api_response_fn，则调用它生成响应；
        否则返回默认的元信息（输出 key 的形状/统计值）。
        """
        if self.api_response_fn is not None:
            outputs = {key: registry.get(key) for key in self.output_keys}
            return self.api_response_fn(**outputs)

        # 默认：返回输出张量的元信息
        response = {}
        for key in self.output_keys:
            val = registry.get(key)
            if isinstance(val, torch.Tensor):
                response[key] = {
                    'shape': list(val.shape),
                    'dtype': str(val.dtype),
                    'min': float(val.min()),
                    'max': float(val.max()),
                    'mean': float(val.mean()),
                }
            elif isinstance(val, (list, tuple)):
                response[key] = {
                    'type': 'list',
                    'length': len(val),
                }
            else:
                response[key] = str(type(val).__name__)
        return response


# ============================================================
# 3. DataFlowPipeline — 数据流流水线
# ============================================================

class DataFlowPipeline:
    """
    基于数据流的流水线编排器

    管理多个 PipelineStage 的执行顺序，
    支持顺序执行、性能分析、中间结果检查、断点续推、micro-batch 并行。
    """

    def __init__(self, stages: List[PipelineStage], name: str = 'Pipeline'):
        self.stages = stages
        self.name = name
        self.registry = DataRegistry()
        self._stage_map = {s.name: s for s in stages}

    def run(self, inputs: Dict[str, Any], profile: bool = False) -> Dict[str, Any]:
        self.registry.clear()
        for key, value in inputs.items():
            self.registry.register(key, value, source='input')

        for stage in self.stages:
            missing = [k for k in stage.input_keys if not self.registry.has(k)]
            if missing:
                raise RuntimeError(f"Stage '{stage.name}' missing inputs: {missing}. Available: {self.registry.keys()}")

            if profile:
                stage.execute_with_profile(self.registry)
            else:
                stage.execute(self.registry)

        return self.registry.to_dict()

    def get_intermediate(self, key: str) -> Any:
        return self.registry.get(key)

    def run_until(self, stage_name: str, inputs: Dict[str, Any], profile: bool = False) -> Dict[str, Any]:
        self.registry.clear()
        for key, value in inputs.items():
            self.registry.register(key, value, source='input')

        for stage in self.stages:
            if profile:
                stage.execute_with_profile(self.registry)
            else:
                stage.execute(self.registry)
            if stage.name == stage_name:
                break

        return self.registry.to_dict()

    def run_from(self, stage_name: str, registry_state: Dict[str, Any], profile: bool = False) -> Dict[str, Any]:
        self.registry.clear()
        for key, value in registry_state.items():
            self.registry.register(key, value, source='checkpoint')

        started = False
        for stage in self.stages:
            if stage.name == stage_name:
                started = True
            if not started:
                continue
            if profile:
                stage.execute_with_profile(self.registry)
            else:
                stage.execute(self.registry)

        return self.registry.to_dict()

    def run_streaming(
        self,
        inputs: Dict[str, Any],
        profile: bool = False,
    ) -> Generator[Dict[str, Any], None, None]:
        """流式执行流水线，每完成一个阶段即 yield 当前阶段的响应数据。

        Yields:
            dict: 包含 stage, stage_index, total_stages, progress, data 等字段。
        """
        total = len(self.stages)
        self.registry.clear()
        for key, value in inputs.items():
            self.registry.register(key, value, source='input')

        for idx, stage in enumerate(self.stages):
            missing = [k for k in stage.input_keys if not self.registry.has(k)]
            if missing:
                raise RuntimeError(
                    f"Stage '{stage.name}' missing inputs: {missing}. "
                    f"Available: {self.registry.keys()}"
                )

            if profile:
                stage.execute_with_profile(self.registry)
            else:
                stage.execute(self.registry)

            # 构建该阶段的 API 响应
            stage_data = stage.build_api_response(self.registry)
            yield {
                'stage': stage.name,
                'stage_index': idx,
                'total_stages': total,
                'progress': (idx + 1) / total,
                'status': 'completed',
                'data': stage_data,
                'timestamp': datetime.now().isoformat(),
            }

    def profile_report(self) -> str:
        lines = [f"\n{'=' * 70}"]
        lines.append(f"Pipeline Profile Report: {self.name}")
        lines.append(f"{'=' * 70}")
        lines.append(f"{'Stage':<25s} {'Calls':>6s} {'Avg(ms)':>10s} {'Total(ms)':>10s} {'Mem(MB)':>10s} {'Params':>10s}")
        lines.append(f"{'-' * 75}")

        total_time = 0
        for stage in self.stages:
            p = stage.profile
            params_m = stage.param_count() / 1e6
            lines.append(
                f"{stage.name:<25s} {p['call_count']:>6d} "
                f"{p['avg_time_ms']:>10.2f} {p['total_time_ms']:>10.2f} "
                f"{p['total_memory_mb']:>10.2f} {params_m:>9.2f}M"
            )
            total_time += p['total_time_ms']

        lines.append(f"{'-' * 75}")
        lines.append(f"{'Total':<25s} {'':>6s} {'':>10s} {total_time:>10.2f}")
        lines.append(f"{'=' * 70}\n")
        return '\n'.join(lines)

    def dataflow_graph(self) -> str:
        lines = [f"\nData Flow Graph: {self.name}"]
        lines.append(f"{'=' * 70}")
        for stage in self.stages:
            in_str = ', '.join(stage.input_keys) if stage.input_keys else '(initial)'
            out_str = ', '.join(stage.output_keys) if stage.output_keys else '(terminal)'
            desc = f"  # {stage.description}" if stage.description else ""
            lines.append(f"  [{stage.name}]{desc}")
            lines.append(f"    reads:  [{in_str}]")
            lines.append(f"    writes: [{out_str}]")
            lines.append(f"")
        lines.append(f"{'=' * 70}\n")
        return '\n'.join(lines)

    def stage_info(self) -> List[dict]:
        infos = []
        for stage in self.stages:
            infos.append({
                'name': stage.name,
                'description': stage.description,
                'input_keys': stage.input_keys,
                'output_keys': stage.output_keys,
                'param_count': stage.param_count(),
                'profile': stage.profile,
            })
        return infos

    def total_params(self) -> int:
        return sum(s.param_count() for s in self.stages)


# ============================================================
# 4. MicroBatchPipeline — 多卡流水线并行
# ============================================================

class MicroBatchPipeline(DataFlowPipeline):
    """
    支持 micro-batch 流水线并行
    
    将一个 batch 切分为 M 个 micro-batch，
    不同 stage 可以同时处理不同的 micro-batch，
    各 stage 可分配到不同 GPU。
    """

    def __init__(self, stages: List[PipelineStage], name: str = 'MicroBatchPipeline',
                 num_microbatches: int = 4, devices: Optional[List[str]] = None):
        super().__init__(stages, name)
        self.num_microbatches = num_microbatches
        self.devices = devices or (['cuda:0', 'cuda:1'] if torch.cuda.device_count() >= 2 else ['cpu'])

    def run_parallel(self, inputs: Dict[str, Any], profile: bool = False) -> Dict[str, Any]:
        micro_inputs = self._split_microbatches(inputs)
        micro_outputs = [{} for _ in range(self.num_microbatches)]

        for i, stage in enumerate(self.stages):
            stage.device = self.devices[i % len(self.devices)]

        for stage in self.stages:
            for mb_idx in range(self.num_microbatches):
                reg = DataRegistry()
                for key, val in micro_inputs[mb_idx].items():
                    reg.register(key, val, source='input')
                reg._data.update(micro_outputs[mb_idx])

                if profile:
                    stage.execute_with_profile(reg)
                else:
                    stage.execute(reg)

                micro_outputs[mb_idx].update(reg.to_dict())

        return self._merge_microbatches(micro_outputs, inputs)

    def _split_microbatches(self, inputs: Dict[str, Any]) -> List[Dict[str, Any]]:
        B = None
        for v in inputs.values():
            if isinstance(v, torch.Tensor):
                B = v.size(0)
                break
        if B is None:
            return [inputs] * self.num_microbatches

        mb_size = max(1, B // self.num_microbatches)
        micro_inputs = []
        for mb_idx in range(self.num_microbatches):
            mb_input = {}
            start = mb_idx * mb_size
            end = start + mb_size if mb_idx < self.num_microbatches - 1 else B
            for key, val in inputs.items():
                if isinstance(val, torch.Tensor) and val.size(0) == B:
                    mb_input[key] = val[start:end]
                else:
                    mb_input[key] = val
            micro_inputs.append(mb_input)
        return micro_inputs

    def _merge_microbatches(self, micro_outputs: List[Dict[str, Any]], original_inputs: Dict[str, Any]) -> Dict[str, Any]:
        if not micro_outputs:
            return {}
        merged = {}
        for key in micro_outputs[0]:
            values = [mo[key] for mo in micro_outputs if key in mo and mo[key] is not None]
            if values and isinstance(values[0], torch.Tensor) and values[0].dim() > 0:
                try:
                    merged[key] = torch.cat(values, dim=0)
                except Exception:
                    merged[key] = values[-1]
            elif values:
                merged[key] = values[-1]
            else:
                merged[key] = None
        return merged


# ============================================================
# 5. 序列化工具 — 中间状态与 API 传输
# ============================================================

class PipelineSerializer:
    """将 DataRegistry 中的张量序列化为 JSON 兼容格式，以及反序列化恢复。

    用于 API 传输（将中间结果发送给前端）和断点续推
    （将中间结果保存/恢复，以便从某个阶段继续推理）。

    序列化格式:
        {
            "key_name": {
                "type": "tensor",
                "shape": [1, 512, 7, 7],
                "dtype": "torch.float32",
                "data": "<base64 编码的 numpy 数组>"
            },
            "another_key": {
                "type": "list_tensor",
                "items": [ ... ]
            },
            ...
        }
    """

    @staticmethod
    def serialize_registry(registry: DataRegistry, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """将 DataRegistry 序列化为 JSON 兼容的字典。"""
        result = {}
        target_keys = keys if keys is not None else registry.keys()

        for key in target_keys:
            if not registry.has(key):
                continue
            val = registry.get(key)
            result[key] = PipelineSerializer._serialize_value(val)
        return result

    @staticmethod
    def deserialize_registry(data: Dict[str, Any]) -> Dict[str, Any]:
        """将序列化的字典反序列化为 Python 对象（含 torch.Tensor）。"""
        result = {}
        for key, val in data.items():
            result[key] = PipelineSerializer._deserialize_value(val)
        return result

    @staticmethod
    def serialize_for_api(registry: DataRegistry, keys: Optional[List[str]] = None) -> Dict[str, Any]:
        """序列化为 API 友好的格式，只包含轻量元信息（不包含原始张量数据）。

        用于前端展示进度和状态，不传输大量数据。
        """
        result = {}
        target_keys = keys if keys is not None else registry.keys()

        for key in target_keys:
            if not registry.has(key):
                continue
            val = registry.get(key)
            meta = registry.meta(key)

            if isinstance(val, torch.Tensor):
                result[key] = {
                    'type': 'tensor',
                    'shape': list(val.shape),
                    'dtype': str(val.dtype),
                    'stats': {
                        'min': float(val.min()),
                        'max': float(val.max()),
                        'mean': float(val.mean()),
                        'std': float(val.std()) if val.numel() > 1 else 0.0,
                    },
                    'memory_mb': val.numel() * val.element_size() / 1024 / 1024,
                }
            elif isinstance(val, (list, tuple)):
                result[key] = {
                    'type': 'list',
                    'length': len(val),
                    'element_shapes': [
                        list(v.shape) if isinstance(v, torch.Tensor) else None
                        for v in val
                    ] if val and isinstance(val[0], torch.Tensor) else None,
                }
            else:
                result[key] = {
                    'type': type(val).__name__,
                    'value': str(val),
                }
        return result

    # ---- 内部序列化/反序列化方法 ----

    @staticmethod
    def _serialize_value(val: Any) -> Any:
        if isinstance(val, torch.Tensor):
            arr = val.cpu().numpy()
            buf = io.BytesIO()
            np.save(buf, arr)
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            return {
                'type': 'tensor',
                'shape': list(val.shape),
                'dtype': str(val.dtype),
                'data': b64,
            }
        elif isinstance(val, (list, tuple)):
            items = [PipelineSerializer._serialize_value(v) for v in val]
            return {'type': 'list', 'items': items}
        elif isinstance(val, np.ndarray):
            buf = io.BytesIO()
            np.save(buf, val)
            b64 = base64.b64encode(buf.getvalue()).decode('ascii')
            return {
                'type': 'ndarray',
                'shape': list(val.shape),
                'dtype': str(val.dtype),
                'data': b64,
            }
        elif isinstance(val, (int, float, str, bool)):
            return {'type': 'scalar', 'value': val}
        elif isinstance(val, dict):
            return {
                'type': 'dict',
                'items': {k: PipelineSerializer._serialize_value(v) for k, v in val.items()}
            }
        else:
            return {'type': 'unknown', 'value': str(val)}

    @staticmethod
    def _deserialize_value(val: Any) -> Any:
        if not isinstance(val, dict):
            return val

        vtype = val.get('type', 'unknown')

        if vtype == 'tensor':
            b64 = val['data']
            buf = io.BytesIO(base64.b64decode(b64))
            arr = np.load(buf)
            t = torch.from_numpy(arr)
            # 恢复到原始 dtype
            dtype_str = val.get('dtype', 'torch.float32')
            try:
                t = t.to(getattr(torch, dtype_str.replace('torch.', '')))
            except Exception:
                pass
            return t

        elif vtype == 'ndarray':
            b64 = val['data']
            buf = io.BytesIO(base64.b64decode(b64))
            arr = np.load(buf)
            return arr

        elif vtype == 'list':
            return [PipelineSerializer._deserialize_value(item) for item in val['items']]

        elif vtype == 'scalar':
            return val['value']

        elif vtype == 'dict':
            return {
                k: PipelineSerializer._deserialize_value(v)
                for k, v in val['items'].items()
            }

        else:
            return val.get('value')