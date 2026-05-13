"""
Pipeline API Server — 切片模型的 FastAPI 路由层

将 SlicedModel 的推理能力暴露为 REST API 和 WebSocket 端点，
供前端实时获取推理进度和结果。

提供的端点:
  - POST /api/pipeline/predict          — 一次性完整推理
  - POST /api/pipeline/predict/stream   — SSE 流式推理
  - POST /api/pipeline/predict/stage    — 推理到指定阶段
  - POST /api/pipeline/predict/resume   — 从中间状态恢复推理
  - POST /api/pipeline/predict/memory   — 记忆增强推理
  - GET  /api/pipeline/stages           — 获取可用阶段列表
  - GET  /api/pipeline/profile          — 获取性能分析数据
  - GET  /api/pipeline/dataflow         — 获取数据流图
  - POST /api/pipeline/load             — 加载模型检查点

WebSocket 推理命令 (通过主 /ws 端点):
  - type: "pipeline_predict_stream"     — 流式推理
  - type: "pipeline_predict"            — 一次性推理
  - type: "pipeline_stage_status"       — 查询阶段状态
"""

import os
import sys
import json
import base64
import asyncio
import numpy as np
from datetime import datetime
from typing import Optional, List, Dict, Any

import torch
from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

# 项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from model.sliced_model import SlicedModel, load_sliced_model_from_checkpoint

# ============================================================
# Pydantic 请求/响应模型
# ============================================================

class PredictRequest(BaseModel):
    """推理请求。"""
    image_base64: str = Field(..., description="Base64 编码的输入图像")
    pointcloud_base64: str = Field(..., description="Base64 编码的点云数据 (numpy array)")
    sub_box: List[float] = Field(..., description="主体边界框 [x1, y1, x2, y2]")
    obj_box: List[float] = Field(..., description="客体边界框 [x1, y1, x2, y2]")
    text_emb_base64: Optional[str] = Field(None, description="文本嵌入 (IAG_TextEmb)")
    model_type: str = Field("iag", description="模型类型: iag / iag_textemb")


class PredictStageRequest(BaseModel):
    """推理到指定阶段的请求。"""
    image_base64: str
    pointcloud_base64: str
    sub_box: List[float]
    obj_box: List[float]
    stage_name: str = Field(..., description="目标阶段名")
    return_intermediate: bool = Field(True, description="是否返回序列化的中间状态")
    text_emb_base64: Optional[str] = None
    model_type: str = "iag"


class PredictResumeRequest(BaseModel):
    """从中间状态恢复推理的请求。"""
    intermediate_state: Dict[str, Any] = Field(..., description="序列化的中间状态")
    stage_name: str = Field(..., description="起始阶段名")
    text_emb_base64: Optional[str] = None
    model_type: str = "iag"


class PredictMemoryRequest(BaseModel):
    """记忆增强推理请求。"""
    image_base64: str
    pointcloud_base64: str
    sub_box: List[float]
    obj_box: List[float]
    affordance_label: Optional[str] = None
    alpha: float = 0.3
    top_k: int = 5
    text_emb_base64: Optional[str] = None
    model_type: str = "iag"


class LoadModelRequest(BaseModel):
    """加载模型请求。"""
    checkpoint_path: str = Field(..., description="模型检查点路径")
    model_type: str = Field("iag", description="模型类型: iag / iag_textemb")
    device: str = Field("auto", description="推理设备: auto / cpu / cuda:0")
    emb_dim: int = 512
    N_p: int = 64
    N_raw: int = 2048
    num_affordance: int = 17
    text_dim: int = 300


class StageInfoResponse(BaseModel):
    """阶段信息响应。"""
    stages: List[Dict[str, Any]] = []
    model_type: str = ""
    device: str = ""


class ProfileResponse(BaseModel):
    """性能分析响应。"""
    report: str = ""
    stages: List[Dict[str, Any]] = []


# ============================================================
# 解码工具函数
# ============================================================

def decode_base64_array(b64_str: str) -> np.ndarray:
    """将 base64 编码的 numpy 数组解码。"""
    import io
    raw = base64.b64decode(b64_str)
    buf = io.BytesIO(raw)
    return np.load(buf, allow_pickle=True)


def decode_base64_image(b64_str: str) -> torch.Tensor:
    """将 base64 编码的图像解码为 [3, H, W] 的 Tensor。"""
    from PIL import Image
    import io

    raw = base64.b64decode(b64_str)
    img = Image.open(io.BytesIO(raw)).convert('RGB')
    arr = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]
    tensor = torch.from_numpy(arr).permute(2, 0, 1)  # [3, H, W]
    return tensor


# ============================================================
# Pipeline API Router
# ============================================================

# 全局切片模型管理器
_sliced_models: Dict[str, SlicedModel] = {}
_default_model_key = 'default'

router = APIRouter(prefix="/api/pipeline", tags=["pipeline"])


def _get_sliced_model(model_type: str = 'iag') -> SlicedModel:
    """获取当前加载的切片模型。"""
    key = f"{model_type}_{_default_model_key}"
    if key not in _sliced_models:
        raise HTTPException(
            status_code=404,
            detail=f"No {model_type} model loaded. Call /api/pipeline/load first."
        )
    return _sliced_models[key]


def _get_device(device_str: str) -> str:
    """解析设备字符串。"""
    if device_str == 'auto':
        return 'cuda:0' if torch.cuda.is_available() else 'cpu'
    return device_str


# ============================================================
# REST API 端点
# ============================================================

@router.post("/load")
async def load_model(req: LoadModelRequest):
    """加载模型检查点到切片推理服务中。"""
    try:
        device = _get_device(req.device)

        model_kwargs = {
            'emb_dim': req.emb_dim,
            'N_p': req.N_p,
            'N_raw': req.N_raw,
            'num_affordance': req.num_affordance,
        }
        if req.model_type == 'iag_textemb':
            model_kwargs['text_dim'] = req.text_dim

        sliced = load_sliced_model_from_checkpoint(
            checkpoint_path=req.checkpoint_path,
            device=device,
            model_type=req.model_type,
            **model_kwargs,
        )

        key = f"{req.model_type}_{_default_model_key}"
        _sliced_models[key] = sliced

        return {
            "success": True,
            "message": f"Model loaded: {req.model_type} on {device}",
            "stages": sliced.get_available_stages(),
            "device": str(sliced.device),
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")


@router.post("/predict")
async def predict(req: PredictRequest):
    """一次性完整推理，返回所有结果。"""
    try:
        sliced = _get_sliced_model(req.model_type)

        img = decode_base64_image(req.image_base64)
        xyz = decode_base64_array(req.pointcloud_base64)
        sub_box = torch.tensor(req.sub_box, dtype=torch.float32)
        obj_box = torch.tensor(req.obj_box, dtype=torch.float32)

        text_emb = None
        if req.text_emb_base64:
            text_emb = decode_base64_array(req.text_emb_base64)

        result = sliced.predict(img, xyz, sub_box, obj_box, text_emb)

        # 转换为 JSON 兼容格式
        response = {
            'affordance_3d': sliced.numpy_to_api_response(result['affordance_3d']),
            'classification': {
                'predicted_classes': result['predicted_class_names'],
                'confidence': result['confidence'],
                'predicted_indices': result['predicted_class_indices'],
            },
            'inference_time_ms': result['inference_time_ms'],
            'model_type': req.model_type,
            'timestamp': datetime.now().isoformat(),
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")


@router.post("/predict/stream")
async def predict_stream(req: PredictRequest):
    """SSE 流式推理，每完成一个阶段推送一次结果。"""
    try:
        sliced = _get_sliced_model(req.model_type)

        img = decode_base64_image(req.image_base64)
        xyz = decode_base64_array(req.pointcloud_base64)
        sub_box = torch.tensor(req.sub_box, dtype=torch.float32)
        obj_box = torch.tensor(req.obj_box, dtype=torch.float32)

        text_emb = None
        if req.text_emb_base64:
            text_emb = decode_base64_array(req.text_emb_base64)

        def event_generator():
            for stage_response in sliced.predict_streaming(img, xyz, sub_box, obj_box, text_emb):
                # 如果是最后一个阶段（decoder），添加完整预测结果
                if stage_response['stage_index'] == stage_response['total_stages'] - 1:
                    registry = sliced.pipeline.registry
                    if registry.has('_3daffordance'):
                        _3d = registry.get('_3daffordance')
                        stage_response['final_prediction'] = sliced.numpy_to_api_response(_3d)
                    if registry.has('logits'):
                        logits = registry.get('logits')
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        pred_class = torch.argmax(probs, dim=-1)
                        stage_response['final_classification'] = {
                            'predicted_classes': [
                                AFFORDANCE_LABELS[idx] if idx < len(AFFORDANCE_LABELS) else f"class_{idx}"
                                for idx in pred_class.cpu().tolist()
                            ],
                            'confidence': torch.max(probs, dim=-1).values.cpu().tolist(),
                        }

                yield f"data: {json.dumps(stage_response, default=str)}\n\n"

            yield f"data: {json.dumps({'type': 'stream_end', 'timestamp': datetime.now().isoformat()})}\n\n"

        return StreamingResponse(
            event_generator(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Streaming error: {str(e)}")


@router.post("/predict/stage")
async def predict_to_stage(req: PredictStageRequest):
    """推理到指定阶段后暂停，返回中间结果和可选的序列化状态。"""
    try:
        sliced = _get_sliced_model(req.model_type)

        img = decode_base64_image(req.image_base64)
        xyz = decode_base64_array(req.pointcloud_base64)
        sub_box = torch.tensor(req.sub_box, dtype=torch.float32)
        obj_box = torch.tensor(req.obj_box, dtype=torch.float32)

        text_emb = None
        if req.text_emb_base64:
            text_emb = decode_base64_array(req.text_emb_base64)

        result = sliced.predict_to_stage(
            stage_name=req.stage_name,
            img=img,
            xyz=xyz,
            sub_box=sub_box,
            obj_box=obj_box,
            text_emb=text_emb,
            return_serialized=req.return_intermediate,
        )

        return result

    except HTTPException:
        raise
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Unknown stage: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Stage inference error: {str(e)}")


@router.post("/predict/resume")
async def predict_from_stage(req: PredictResumeRequest):
    """从保存的中间状态恢复推理。"""
    try:
        sliced = _get_sliced_model(req.model_type)

        text_emb = None
        if req.text_emb_base64:
            text_emb = decode_base64_array(req.text_emb_base64)

        result = sliced.predict_from_stage(
            stage_name=req.stage_name,
            intermediate_state=req.intermediate_state,
            text_emb=text_emb,
        )

        response = {
            'affordance_3d': sliced.numpy_to_api_response(result['affordance_3d']),
            'classification': {
                'predicted_classes': result['predicted_class_names'],
                'confidence': result['confidence'],
                'predicted_indices': result['predicted_class_indices'],
            },
            'resumed_from_stage': result.get('resumed_from_stage'),
            'timestamp': datetime.now().isoformat(),
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Resume error: {str(e)}")


@router.post("/predict/memory")
async def predict_with_memory(req: PredictMemoryRequest):
    """带记忆增强的推理。"""
    try:
        sliced = _get_sliced_model(req.model_type)

        img = decode_base64_image(req.image_base64)
        xyz = decode_base64_array(req.pointcloud_base64)
        sub_box = torch.tensor(req.sub_box, dtype=torch.float32)
        obj_box = torch.tensor(req.obj_box, dtype=torch.float32)

        text_emb = None
        if req.text_emb_base64:
            text_emb = decode_base64_array(req.text_emb_base64)

        # 尝试获取 MemoryManager
        memory_manager = _get_memory_manager()

        result = sliced.predict_with_memory(
            img=img,
            xyz=xyz,
            sub_box=sub_box,
            obj_box=obj_box,
            memory_manager=memory_manager,
            affordance_label=req.affordance_label,
            alpha=req.alpha,
            top_k=req.top_k,
            text_emb=text_emb,
        )

        response = {
            'affordance_3d': sliced.numpy_to_api_response(result['affordance_3d']),
            'classification': {
                'predicted_classes': result['predicted_class_names'],
                'confidence': result['confidence'],
            },
            'memory_applied': result.get('memory_applied', False),
            'timestamp': datetime.now().isoformat(),
        }

        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Memory inference error: {str(e)}")


@router.get("/stages", response_model=StageInfoResponse)
async def get_stages(model_type: str = Query("iag")):
    """获取可用阶段列表和详情。"""
    try:
        sliced = _get_sliced_model(model_type)
        return StageInfoResponse(
            stages=sliced.get_stage_info(),
            model_type=model_type,
            device=str(sliced.device),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(model_type: str = Query("iag")):
    """获取性能分析数据。"""
    try:
        sliced = _get_sliced_model(model_type)
        return ProfileResponse(
            report=sliced.get_profile_report(),
            stages=sliced.get_stage_info(),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/dataflow")
async def get_dataflow(model_type: str = Query("iag")):
    """获取数据流图。"""
    try:
        sliced = _get_sliced_model(model_type)
        return {"dataflow": sliced.get_dataflow_graph()}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


# ============================================================
# Memory Manager 获取工具
# ============================================================

def _get_memory_manager():
    """尝试获取全局 MemoryManager 实例。"""
    try:
        # 尝试从 memory_system 模块获取
        sys.path.insert(0, PROJECT_ROOT)
        from memory_system.memory_manager import MemoryManager
        # 这里可以扩展为从全局状态获取已初始化的 manager
        # 暂时返回 None，由调用者决定是否使用记忆
        return None
    except ImportError:
        return None


# ============================================================
# WebSocket 集成辅助
# ============================================================

AFFORDANCE_LABELS = [
    "grasp", "contain", "lift", "open", "lay", "sit", "support",
    "wrapgrasp", "pour", "move", "display", "push", "listen",
    "wear", "press", "cut", "stab",
]


async def handle_pipeline_ws_command(
    command: str,
    data: dict,
    websocket: WebSocket,
    org: str,
    uid: str,
):
    """处理通过主 WebSocket 发来的流水线命令。

    这个函数可被集成到 fastapi_backend/main.py 的 WebSocket 处理器中，
    当收到 type 为 pipeline_* 的消息时调用。

    Parameters
    ----------
    command : str
        命令类型 (pipeline_predict, pipeline_predict_stream, 等)
    data : dict
        命令数据
    websocket : WebSocket
        WebSocket 连接
    org : str
        组织名
    uid : str
        用户 ID
    """
    model_type = data.get('model_type', 'iag')

    try:
        sliced = _get_sliced_model(model_type)
    except Exception:
        await websocket.send_json({
            "type": "pipeline_error",
            "message": f"No {model_type} model loaded. Call /api/pipeline/load first.",
        })
        return

    if command == "pipeline_predict_stream":
        # 流式推理
        try:
            img = decode_base64_image(data['image_base64'])
            xyz = decode_base64_array(data['pointcloud_base64'])
            sub_box = torch.tensor(data['sub_box'], dtype=torch.float32)
            obj_box = torch.tensor(data['obj_box'], dtype=torch.float32)
            text_emb = None
            if data.get('text_emb_base64'):
                text_emb = decode_base64_array(data['text_emb_base64'])

            for stage_response in sliced.predict_streaming(img, xyz, sub_box, obj_box, text_emb):
                # 在最后一个阶段添加最终预测
                if stage_response['stage_index'] == stage_response['total_stages'] - 1:
                    registry = sliced.pipeline.registry
                    if registry.has('_3daffordance'):
                        _3d = registry.get('_3daffordance')
                        stage_response['final_prediction'] = sliced.numpy_to_api_response(_3d)
                    if registry.has('logits'):
                        logits = registry.get('logits')
                        probs = torch.nn.functional.softmax(logits, dim=-1)
                        pred_class = torch.argmax(probs, dim=-1)
                        stage_response['final_classification'] = {
                            'predicted_classes': [
                                AFFORDANCE_LABELS[idx] if idx < len(AFFORDANCE_LABELS) else f"class_{idx}"
                                for idx in pred_class.cpu().tolist()
                            ],
                            'confidence': torch.max(probs, dim=-1).values.cpu().tolist(),
                        }

                await websocket.send_json({
                    "type": "pipeline_stage_update",
                    "scene_id": data.get('scene_id', ''),
                    "data": stage_response,
                })

            await websocket.send_json({
                "type": "pipeline_complete",
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            await websocket.send_json({
                "type": "pipeline_error",
                "message": str(e),
            })

    elif command == "pipeline_predict":
        # 一次性推理
        try:
            img = decode_base64_image(data['image_base64'])
            xyz = decode_base64_array(data['pointcloud_base64'])
            sub_box = torch.tensor(data['sub_box'], dtype=torch.float32)
            obj_box = torch.tensor(data['obj_box'], dtype=torch.float32)
            text_emb = None
            if data.get('text_emb_base64'):
                text_emb = decode_base64_array(data['text_emb_base64'])

            result = sliced.predict(img, xyz, sub_box, obj_box, text_emb)

            await websocket.send_json({
                "type": "pipeline_result",
                "data": {
                    'affordance_3d': sliced.numpy_to_api_response(result['affordance_3d']),
                    'classification': {
                        'predicted_classes': result['predicted_class_names'],
                        'confidence': result['confidence'],
                    },
                    'inference_time_ms': result['inference_time_ms'],
                },
                "timestamp": datetime.now().isoformat(),
            })

        except Exception as e:
            await websocket.send_json({
                "type": "pipeline_error",
                "message": str(e),
            })

    elif command == "pipeline_stage_status":
        # 查询当前模型阶段信息
        try:
            await websocket.send_json({
                "type": "pipeline_stage_info",
                "data": {
                    "stages": sliced.get_stage_info(),
                    "available_stages": sliced.get_available_stages(),
                },
            })
        except Exception as e:
            await websocket.send_json({
                "type": "pipeline_error",
                "message": str(e),
            })

    else:
        await websocket.send_json({
            "type": "pipeline_error",
            "message": f"Unknown pipeline command: {command}",
        })