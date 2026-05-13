"""
Annotation Tool - Streamlit Interface
标注工具界面

功能:
- 上传图像
- 自动检测主体和客体
- 预测动作类型
- 手动调整和导出标注
"""

import streamlit as st
import os
import sys
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib.patches as patches

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from annotation_model import build_annotation_model


# 可供性类别
AFFORDANCE_LABELS = [
    'grasp', 'contain', 'lift', 'open', 'lay', 'sit', 'support',
    'wrapgrasp', 'pour', 'move', 'display', 'push', 'listen',
    'wear', 'press', 'cut', 'stab'
]

# 颜色映射
COLORS = {
    'subject': '#FF6B6B',    # 红色 - 主体
    'object': '#4ECDC4',     # 青色 - 客体
    'prediction': '#FFE66D'  # 黄色 - 预测
}


def load_model(model_path, device):
    """加载模型"""
    model = build_annotation_model(num_interactions=17, pretrained=False)
    
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        st.success(f"模型加载成功: {model_path}")
    else:
        st.warning(f"模型文件不存在: {model_path}，使用未训练的模型")
    
    model = model.to(device)
    model.eval()
    return model


def preprocess_image(image, target_size=(224, 224)):
    """预处理图像"""
    # 保存原始尺寸
    original_size = image.size
    
    # 转换为RGB
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # 缩放
    image_resized = image.resize(target_size, Image.BILINEAR)
    
    # 转换为tensor
    image_array = np.array(image_resized).astype(np.float32) / 255.0
    
    # 标准化
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image_array = (image_array - mean) / std
    
    # 转换为tensor [C, H, W]
    image_tensor = torch.from_numpy(image_array.transpose(2, 0, 1))
    
    return image_tensor, original_size


def draw_boxes(image, subject_boxes, object_boxes, scores=None):
    """在图像上绘制边界框"""
    draw = ImageDraw.Draw(image)
    
    # 绘制主体边界框
    for i, box in enumerate(subject_boxes):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=COLORS['subject'], width=3)
        label = "Subject"
        if scores is not None and i < len(scores):
            label += f" ({scores[i]:.2f})"
        draw.text((x1, y1 - 15), label, fill=COLORS['subject'])
    
    # 绘制客体边界框
    for i, box in enumerate(object_boxes):
        x1, y1, x2, y2 = box.tolist()
        draw.rectangle([x1, y1, x2, y2], outline=COLORS['object'], width=3)
        label = "Object"
        if scores is not None and i < len(scores):
            label += f" ({scores[i]:.2f})"
        draw.text((x1, y1 - 15), label, fill=COLORS['object'])
    
    return image


def main():
    st.set_page_config(
        page_title="图像标注工具",
        page_icon="🏷️",
        layout="wide"
    )
    
    st.title("🏷️ 图像标注工具")
    st.markdown("""
    自动检测图像中的**主体**（交互发起者）和**客体**（被交互物体），
    并预测**动作类型**（17种可供性类别）。
    """)
    
    # ============ 侧边栏 ============
    st.sidebar.title("设置")
    
    # 模型设置
    model_path = st.sidebar.text_input(
        "模型路径",
        value="./checkpoints/annotation/best.pt"
    )
    
    # 设备选择
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    st.sidebar.info(f"设备: {device}")
    
    # 置信度阈值
    conf_threshold = st.sidebar.slider(
        "检测置信度阈值",
        min_value=0.1,
        max_value=0.9,
        value=0.5,
        step=0.05
    )
    
    # 加载模型
    @st.cache_resource
    def get_model():
        return load_model(model_path, device)
    
    model = get_model()
    
    # ============ 主界面 ============
    tab1, tab2 = st.tabs(["📤 上传图像", "📝 标注结果"])
    
    with tab1:
        st.header("上传图像")
        
        # 文件上传
        uploaded_file = st.file_uploader(
            "选择图像文件",
            type=['jpg', 'jpeg', 'png'],
            help="支持 JPG, JPEG, PNG 格式"
        )
        
        # 或使用示例图像
        use_example = st.checkbox("使用示例图像")
        
        if uploaded_file is not None or use_example:
            # 加载图像
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
            else:
                # 创建示例图像
                image = Image.new('RGB', (224, 224), color=(128, 128, 128))
            
            # 显示原图
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("原始图像")
                st.image(image, use_column_width=True)
            
            # 预处理
            image_tensor, original_size = preprocess_image(image)
            image_tensor = image_tensor.unsqueeze(0).to(device)
            
            # 运行推理
            if st.button("🔍 开始检测", type="primary"):
                with st.spinner("正在检测..."):
                    with torch.no_grad():
                        outputs = model(image_tensor)
                
                # 存储结果到session state
                st.session_state['outputs'] = outputs
                st.session_state['original_size'] = original_size
                st.session_state['original_image'] = image
                
                st.success("检测完成!")
                st.experimental_rerun()
    
    with tab2:
        st.header("标注结果")
        
        if 'outputs' not in st.session_state:
            st.info("请先上传图像并进行检测")
            return
        
        outputs = st.session_state['outputs']
        original_size = st.session_state['original_size']
        original_image = st.session_state['original_image'].copy()
        
        # 获取检测结果
        subject_boxes = outputs.get('subject_boxes', torch.tensor([]))
        object_boxes = outputs.get('object_boxes', torch.tensor([]))
        subject_scores = outputs.get('subject_scores', torch.tensor([]))
        object_scores = outputs.get('object_scores', torch.tensor([]))
        interaction_logits = outputs.get('interaction_logits', None)
        
        # 缩放回原始尺寸
        scale_x = original_size[0] / 224
        scale_y = original_size[1] / 224
        
        if len(subject_boxes) > 0:
            subject_boxes[:, 0] *= scale_x
            subject_boxes[:, 2] *= scale_x
            subject_boxes[:, 1] *= scale_y
            subject_boxes[:, 3] *= scale_y
        
        if len(object_boxes) > 0:
            object_boxes[:, 0] *= scale_x
            object_boxes[:, 2] *= scale_x
            object_boxes[:, 1] *= scale_y
            object_boxes[:, 3] *= scale_y
        
        # 绘制结果
        result_image = draw_boxes(
            original_image,
            subject_boxes.cpu() if len(subject_boxes) > 0 else [],
            object_boxes.cpu() if len(object_boxes) > 0 else [],
            scores=None
        )
        
        # 显示结果图像
        st.subheader("检测结果")
        st.image(result_image, use_column_width=True)
        
        # 图例
        st.markdown("""
        **图例:**
        - 🟥 <span style="color:#FF6B6B">**红色**</span>: 主体 (Subject)
        - 🟦 <span style="color:#4ECDC4">**青色**</span>: 客体 (Object)
        """, unsafe_allow_html=True)
        
        # 检测详情
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("主体检测")
            if len(subject_boxes) > 0:
                for i, (box, score) in enumerate(zip(subject_boxes, subject_scores)):
                    with st.expander(f"主体 #{i + 1} (置信度: {score:.2f})"):
                        st.write(f"边界框: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
                        
                        # 手动调整
                        new_box = st.text_input(
                            "调整边界框 (x1, y1, x2, y2)",
                            value=f"{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}",
                            key=f"subject_box_{i}"
                        )
            else:
                st.warning("未检测到主体")
        
        with col2:
            st.subheader("客体检测")
            if len(object_boxes) > 0:
                for i, (box, score) in enumerate(zip(object_boxes, object_scores)):
                    with st.expander(f"客体 #{i + 1} (置信度: {score:.2f})"):
                        st.write(f"边界框: [{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}]")
                        
                        # 手动调整
                        new_box = st.text_input(
                            "调整边界框 (x1, y1, x2, y2)",
                            value=f"{box[0]:.1f}, {box[1]:.1f}, {box[2]:.1f}, {box[3]:.1f}",
                            key=f"object_box_{i}"
                        )
            else:
                st.warning("未检测到客体")
        
        # 动作类型预测
        st.subheader("动作类型预测")
        
        if interaction_logits is not None and len(interaction_logits) > 0:
            # 获取top-5预测
            probs = torch.softmax(interaction_logits[0], dim=0)
            top5_probs, top5_indices = torch.topk(probs, 5)
            
            # 显示预测结果
            for prob, idx in zip(top5_probs, top5_indices):
                col_prob, col_label = st.columns([3, 1])
                with col_prob:
                    st.progress(prob.item())
                with col_label:
                    st.write(f"**{AFFORDANCE_LABELS[idx]}**: {prob.item():.2%}")
            
            # 选择最终动作类型
            selected_interaction = st.selectbox(
                "选择动作类型",
                options=AFFORDANCE_LABELS,
                index=top5_indices[0].item()
            )
        else:
            st.warning("无法预测动作类型（需要同时检测到主体和客体）")
            selected_interaction = st.selectbox(
                "选择动作类型",
                options=AFFORDANCE_LABELS
            )
        
        # 导出标注
        st.subheader("导出标注")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("📋 复制JSON格式", type="primary"):
                import json
                
                annotation = {
                    "image_width": original_size[0],
                    "image_height": original_size[1],
                    "subject_box": subject_boxes[0].tolist() if len(subject_boxes) > 0 else [],
                    "object_box": object_boxes[0].tolist() if len(object_boxes) > 0 else [],
                    "interaction": selected_interaction
                }
                
                json_str = json.dumps(annotation, indent=2)
                st.code(json_str, language='json')
                
                st.success("标注已生成!")
        
        with col2:
            if st.button("💾 保存标注文件"):
                st.info("标注文件保存功能待实现")


if __name__ == "__main__":
    main()
