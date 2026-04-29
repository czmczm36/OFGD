"""
主体排序器（Subject Ranker）
使用 YOLO + InstructBLIP 对图像中的主体进行重要性排序

使用方法:
    python subject_ranker.py --images_dir images --yolo_model yolov8s.pt --blip_model /path/to/instructblip --output_dir outputs
    
    支持断点续跑:
    python subject_ranker.py --resume

输出:
    - outputs/results.jsonl: 每张图的结果（jsonl格式，与原来一致）
    - outputs/results_full_objects.jsonl: 仅含每张图「全物体」列表，专供 eval_chair_benchmark / eval_chair 用 --results-file 指定
    - outputs/debug_boxes/: 检测框和排序可视化
    - outputs/debug_masks/: 遮挡图可视化
    - outputs/subject_ranker.log: 运行日志
"""

import os
import json
import logging
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
from ultralytics import YOLO
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration


# ==================== 配置 ====================
class Config:
    """配置类"""
    # 路径配置
    IMAGES_DIR = "images"
    YOLO_MODEL_PATH = "yolov8s.pt"
    INSTRUCTBLIP_MODEL_PATH = "/mnt/d5f4cfb6-8afe-40a4-8650-2965046cd208/shuiyiboy/instructblip-vicuna-7b"
    OUTPUT_DIR = "outputs"
    
    # YOLO 配置
    YOLO_CONF_THRESH = 0.25
    YOLO_IOU_THRESH = 0.45
    TARGET_CLASS = None  # None表示检测所有类别，也可以指定类别ID（如0=person）
    K1 = 10  # 初始候选数量
    K2 = 5   # 压缩后候选数量
    K_FULL = 500  # 每张图最多取多少检测框用于生成「全物体」列表（供 eval 使用，与 K1/K2 无关）
    # 候选压缩策略：'area'（面积最大，参考1_detect_mainobj.py）或 'combined'（面积+中心性+置信度）
    COMPRESS_METHOD = 'area'  # 使用面积最大法，更准确
    
    # InstructBLIP 配置
    # 改进：要求生成包含所有检测到的object的描述，用于全局对比
    ANCHOR_PROMPT_SINGLE = "Describe all the main objects in the image. Include all detected subjects and their key characteristics."
    ANCHOR_PROMPT_MULTIPLE = "Describe all the main objects in the image. Include all detected subjects and their key characteristics."
    ANCHOR_PROMPT_FALLBACK = "Describe the main subjects in the image in one sentence."
    MAX_NEW_TOKENS = 150  # 增加到150，确保能生成完整的句子描述
    MAX_TOKENS_FOR_SCORE = 40  # 计算分数时只使用前N个token
    
    # 遮挡方式：'mean', 'blur', 'black'
    # 改进：使用black fill更彻底地破坏语义
    MASK_METHOD = 'black'  # 使用纯黑色遮挡
    MASK_PADDING_RATIO = 0.0  # 0% padding，遮挡区域就是bbox范围，不超出
    
    # 设备
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== 日志设置 ====================
def setup_logging(output_dir: str):
    """设置日志"""
    log_file = os.path.join(output_dir, "subject_ranker.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


# ==================== Step 1: 工程骨架 ====================
def get_image_list(images_dir: str) -> List[str]:
    """获取所有图片路径"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(Path(images_dir).glob(f'*{ext}'))
        image_paths.extend(Path(images_dir).glob(f'*{ext.upper()}'))
    
    image_paths = sorted([str(p) for p in image_paths])
    logger.info(f"找到 {len(image_paths)} 张图片")
    
    if len(image_paths) > 0:
        logger.info(f"前5个路径:")
        for i, path in enumerate(image_paths[:5], 1):
            logger.info(f"  {i}. {path}")
    
    return image_paths


def setup_output_dirs(output_dir: str):
    """创建输出目录"""
    dirs = [
        output_dir,
        os.path.join(output_dir, "debug_boxes"),
        os.path.join(output_dir, "debug_masks"),
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
    # 注意：这里不使用logger，因为logger可能还未初始化


# ==================== Step 2: YOLO 检测 ====================
def detect_objects(yolo_model, image_path: str, target_class: int = None, k1: int = 10) -> List[Dict]:
    """
    使用YOLO检测目标，返回候选列表
    参考1_detect_mainobj.py的方法，支持所有类别
    返回: [{bbox: [x1,y1,x2,y2], conf: float, cls: int, cls_name: str}, ...]
    """
    results = yolo_model(image_path, conf=Config.YOLO_CONF_THRESH, iou=Config.YOLO_IOU_THRESH)
    
    candidates = []
    if len(results) > 0 and results[0].boxes is not None:
        boxes = results[0].boxes
        # 获取类别名称映射
        class_names = results[0].names
        
        for box in boxes:
            cls = int(box.cls.item())
            conf = float(box.conf.item())
            
            # 如果指定了target_class，只保留该类别；否则保留所有类别
            if target_class is None or cls == target_class:
                # YOLO返回的是xyxy格式
                bbox = box.xyxy[0].cpu().numpy().tolist()
                candidates.append({
                    'bbox': bbox,
                    'conf': conf,
                    'cls': cls,
                    'cls_name': class_names[cls]  # 添加类别名称
                })
    
    # 按置信度排序，取Top-K1
    candidates.sort(key=lambda x: x['conf'], reverse=True)
    candidates = candidates[:k1]
    
    return candidates


def visualize_boxes(image_path: str, candidates: List[Dict], output_path: str):
    """可视化检测框"""
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    for i, cand in enumerate(candidates):
        bbox = cand['bbox']
        conf = cand['conf']
        x1, y1, x2, y2 = bbox
        
        # 画框
        draw.rectangle([x1, y1, x2, y2], outline='red', width=3)
        
        # 标注置信度和类别名称
        cls_name = cand.get('cls_name', f'cls_{cand.get("cls", "?")}')
        label = f"{i}: {cls_name} {conf:.2f}"
        draw.text((x1, y1 - 20), label, fill='red')
    
    img.save(output_path)
    logger.debug(f"保存检测框可视化: {output_path}")


# ==================== Step 3: 候选压缩 ====================
def compress_candidates(candidates: List[Dict], k2: int, image_shape: Tuple[int, int], method: str = 'area') -> List[Dict]:
    """
    压缩候选：参考1_detect_mainobj.py的方法
    method: 'area'（面积最大法，更准确）或 'combined'（面积+中心性+置信度）
    保留orig_id以便debug时追踪
    """
    if len(candidates) <= k2:
        # 即使不压缩，也添加orig_id
        for i, cand in enumerate(candidates):
            if 'orig_id' not in cand:
                cand['orig_id'] = i
        return candidates
    
    img_h, img_w = image_shape
    
    scored_candidates = []
    for i, cand in enumerate(candidates):
        bbox = cand['bbox']
        x1, y1, x2, y2 = bbox
        
        # 计算面积（参考1_detect_mainobj.py）
        area = (x2 - x1) * (y2 - y1)
        
        if method == 'area':
            # 方法1：面积最大法（参考1_detect_mainobj.py）
            # 直接使用面积作为分数
            score = area
        else:
            # 方法2：综合分数法（面积+中心性+置信度）
            area_score = area / (img_w * img_h)  # 归一化
            
            # 计算中心性（距离图像中心的距离）
            img_center_x, img_center_y = img_w / 2, img_h / 2
            bbox_center_x = (x1 + x2) / 2
            bbox_center_y = (y1 + y2) / 2
            dist_from_center = np.sqrt(
                (bbox_center_x - img_center_x)**2 + (bbox_center_y - img_center_y)**2
            )
            max_dist = np.sqrt(img_center_x**2 + img_center_y**2)
            centrality_score = 1.0 - (dist_from_center / max_dist) if max_dist > 0 else 1.0
            
            # 综合分数：面积权重0.6，中心性权重0.4，加上原始置信度
            score = 0.6 * area_score + 0.4 * centrality_score + 0.2 * cand['conf']
        
        scored_candidates.append({
            **cand,
            'orig_id': i,  # 保留原始索引
            'heuristic_score': score
        })
    
    # 按分数排序
    scored_candidates.sort(key=lambda x: x['heuristic_score'], reverse=True)
    
    # 取Top-K2
    return scored_candidates[:k2]


# ==================== Step 4: 生成 Region Caption（改进版）====================
def generate_region_caption(model, processor, image: Image.Image, bbox: List[float], prompt: str) -> str:
    """
    对单个候选框生成region caption（crop出该区域）
    这样caption天然绑定该bbox，信号更强
    """
    # Crop出bbox区域
    x1, y1, x2, y2 = [int(coord) for coord in bbox]
    # 确保坐标在图像范围内
    w, h = image.size
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(w, x2), min(h, y2)
    
    # 如果bbox无效，返回空
    if x2 <= x1 or y2 <= y1:
        return ""
    
    # Crop区域
    cropped_image = image.crop((x1, y1, x2, y2))
    
    # 使用crop后的图像生成caption
    inputs = processor(images=cropped_image, text=prompt, return_tensors="pt").to(Config.DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=3
        )
    
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    # 清理输出
    if generated_text.lower().startswith(prompt.lower()):
        generated_text = generated_text[len(prompt):].strip()
        if generated_text and generated_text[0] in [':', '-', '.']:
            generated_text = generated_text[1:].strip()
    
    return generated_text


def generate_anchor_caption(model, processor, image: Image.Image, prompt: str, num_candidates: int = 1) -> str:
    """
    使用InstructBLIP生成anchor caption
    改进：根据候选数量选择不同的prompt策略，确保聚焦单一主体
    """
    # 根据候选数量选择prompt
    if num_candidates == 1:
        # 单个候选：使用简单但强调主体的prompt
        actual_prompt = Config.ANCHOR_PROMPT_SINGLE
    else:
        # 多个候选：使用强调"单一最显著主体"的prompt
        actual_prompt = Config.ANCHOR_PROMPT_MULTIPLE
    
    # 如果用户传入了自定义prompt，使用它
    if prompt != Config.ANCHOR_PROMPT_FALLBACK:
        actual_prompt = prompt
    
    inputs = processor(images=image, text=actual_prompt, return_tensors="pt").to(Config.DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=Config.MAX_NEW_TOKENS,
            do_sample=False,
            num_beams=3
        )
    
    generated_text = processor.batch_decode(outputs, skip_special_tokens=True)[0].strip()
    
    # 检查是否因为max_new_tokens限制而截断（如果最后一个token不是EOS，可能被截断）
    eos_token_id = processor.tokenizer.eos_token_id
    if eos_token_id is not None and outputs[0][-1].item() != eos_token_id:
        logger.debug(f"警告：生成的caption可能因max_new_tokens={Config.MAX_NEW_TOKENS}限制而截断")
    
    # 清理输出：只在输出以prompt开头时才截断（更安全）
    if generated_text.lower().startswith(actual_prompt.lower()):
        generated_text = generated_text[len(actual_prompt):].strip()
        # 移除可能的前导标点
        if generated_text and generated_text[0] in [':', '-', '.']:
            generated_text = generated_text[1:].strip()
    
    # 检查是否生成了群体描述（需要避免）
    group_keywords = ['group of', 'several', 'many', 'multiple', 'people', 'crowd', 'together']
    if any(keyword in generated_text.lower() for keyword in group_keywords):
        logger.warning(f"生成的caption可能包含群体描述，考虑使用更强的prompt: {generated_text[:100]}")
    
    logger.debug(f"Anchor caption ({num_candidates} candidates): {generated_text}")
    return generated_text


# ==================== Step 5: 生成遮挡图 ====================
def create_masked_image(image: Image.Image, bbox: List[float], method: str = 'mean', padding_ratio: float = 0.1) -> Image.Image:
    """
    创建遮挡图
    method: 'mean', 'blur', 'black'
    padding_ratio: bbox的padding比例（0.1表示扩展10%），提升遮挡效果
    
    重要：每次调用都会创建新的图像副本，不会修改原始图像
    """
    # 关键修复：创建图像副本，避免修改原始图像
    img_array = np.array(image).copy()  # 使用copy()确保不修改原图
    x1, y1, x2, y2 = [float(coord) for coord in bbox]
    
    # 计算bbox尺寸
    bbox_w = x2 - x1
    bbox_h = y2 - y1
    
    # 添加padding（扩展bbox）
    pad_w = bbox_w * padding_ratio
    pad_h = bbox_h * padding_ratio
    x1 = x1 - pad_w
    y1 = y1 - pad_h
    x2 = x2 + pad_w
    y2 = y2 + pad_h
    
    # 转换为整数并确保坐标在图像范围内
    h, w = img_array.shape[:2]
    x1, y1 = max(0, int(x1)), max(0, int(y1))
    x2, y2 = min(w, int(x2)), min(h, int(y2))
    
    # 检查bbox是否有效
    if x2 <= x1 or y2 <= y1:
        logger.warning(f"无效的bbox: ({x1}, {y1}, {x2}, {y2})，跳过遮挡")
        return image.copy()
    
    if method == 'mean':
        # 用整图均值填充
        mean_color = img_array.mean(axis=(0, 1)).astype(np.uint8)
        img_array[y1:y2, x1:x2] = mean_color
    elif method == 'black':
        # 用纯黑色填充（确保是 [0, 0, 0]）
        # 对于RGB图像，确保所有通道都是0
        if len(img_array.shape) == 3:
            img_array[y1:y2, x1:x2] = np.array([0, 0, 0], dtype=np.uint8)
        else:
            img_array[y1:y2, x1:x2] = 0
        logger.debug(f"遮挡区域: ({x1}, {y1}) -> ({x2}, {y2}), 大小: {x2-x1}x{y2-y1}")
    elif method == 'blur':
        # 模糊处理（简化版：用PIL的模糊滤镜）
        from PIL import ImageFilter
        blurred_img = image.filter(ImageFilter.GaussianBlur(radius=15))
        blurred_array = np.array(blurred_img)
        img_array[y1:y2, x1:x2] = blurred_array[y1:y2, x1:x2]
    else:
        raise ValueError(f"Unknown mask method: {method}")
    
    return Image.fromarray(img_array)


def create_masked_images(image: Image.Image, candidates: List[Dict], method: str = 'mean', 
                         padding_ratio: float = 0.1, save_debug: bool = False, 
                         output_dir: str = None, base_name: str = None) -> Dict[int, Image.Image]:
    """
    创建所有遮挡图
    save_debug: 是否保存到磁盘（默认False，节省空间和I/O）
    """
    masked_images = {}
    
    for i, cand in enumerate(candidates):
        # 关键：每次都从原始图像创建新的遮挡图，确保每个遮挡图都是独立的
        # 每个遮挡图只遮挡一个候选的bbox区域
        masked_img = create_masked_image(image, cand['bbox'], method, padding_ratio)
        masked_images[i] = masked_img
        
        # 调试：检查遮挡图是否真的不同
        masked_array = np.array(masked_img)
        masked_mean = masked_array.mean()
        bbox = cand['bbox']
        # 计算遮挡区域的像素数量，用于验证
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        masked_pixels = masked_array[y1:y2, x1:x2].mean()
        logger.info(f"创建遮挡图 {i}: bbox={bbox}, 整图均值={masked_mean:.2f}, 遮挡区域均值={masked_pixels:.2f}")
        
        # 可选：保存到磁盘用于调试
        if save_debug and output_dir and base_name:
            output_path = os.path.join(output_dir, f"{base_name}_mask_{i}.jpg")
            masked_img.save(output_path)
            logger.debug(f"保存遮挡图: {output_path}")
    
    return masked_images


# ==================== Step 6: 计算 log-likelihood drop ====================
def compute_log_likelihood(model, processor, image: Image.Image, prompt: str, caption: str, max_tokens: int = 40) -> float:
    """
    计算给定图像和caption的log-likelihood（teacher forcing）
    参考2_run_instructblip.py的实现方式，正确处理qformer_input_ids
    """
    # 关键：每次都重新处理图像，确保使用正确的图像（不是缓存的）
    # 先处理图像（确保使用当前图像）
    pixel_values = processor.image_processor(images=image, return_tensors="pt")["pixel_values"].to(Config.DEVICE)
    
    # 参考你的代码：尝试使用processor的标准方式，如果失败则手动构建
    try:
        # 标准方式：同时处理图像和文本
        inputs = processor(images=image, text=prompt, return_tensors="pt")
        # 移动到设备
        inputs = {k: v.to(Config.DEVICE) if isinstance(v, torch.Tensor) else v 
                 for k, v in inputs.items()}
        # 确保使用当前图像的 pixel_values（不是缓存的）
        inputs['pixel_values'] = pixel_values
        input_ids_len = inputs["input_ids"].shape[1]  # 记录输入长度
    except (TypeError, AttributeError) as e:
        # 如果processor失败，手动处理（参考你的代码）
        # 注意：pixel_values 已经在上面处理了
        
        # 对于Q-Former，检查是否有qformer_tokenizer
        if hasattr(processor, 'qformer_tokenizer'):
            qformer_tokenizer = processor.qformer_tokenizer
        else:
            qformer_tokenizer = processor.tokenizer
        
        # Q-Former输入（通常是一个简单的指令）
        qformer_prompt = prompt[:50] if len(prompt) > 50 else prompt
        qformer_inputs = qformer_tokenizer(
            qformer_prompt,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=32
        )
        qformer_input_ids = qformer_inputs["input_ids"].to(Config.DEVICE)
        qformer_attention_mask = qformer_inputs.get("attention_mask", None)
        if qformer_attention_mask is not None:
            qformer_attention_mask = qformer_attention_mask.to(Config.DEVICE)
        
        # 语言模型输入
        language_inputs = processor.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        input_ids = language_inputs["input_ids"].to(Config.DEVICE)
        input_ids_len = input_ids.shape[1]
        
        inputs = {
            "pixel_values": pixel_values,
            "input_ids": input_ids,
            "qformer_input_ids": qformer_input_ids,
        }
        if qformer_attention_mask is not None:
            inputs["qformer_attention_mask"] = qformer_attention_mask
    
    # 关键修复：必须用 processor(images=..., text=...) 来编码 full_text
    # 这样才能包含图像占位符 token（如 <image>），模型才会真正使用图像
    # 如果直接用 tokenizer，会丢失图像 token，导致模型变成纯文本模型
    
    full_text = f"{prompt} {caption}"
    
    # 方案A：用 processor 编码 full_text（会自动带上 image token）
    full_inputs = processor(images=image, text=full_text, return_tensors="pt")
    full_inputs = {k: v.to(Config.DEVICE) if isinstance(v, torch.Tensor) else v 
                   for k, v in full_inputs.items()}
    
    input_ids_full = full_inputs["input_ids"]
    attention_mask_full = full_inputs.get("attention_mask", torch.ones_like(input_ids_full))
    
    # 获取 prompt 的长度（也用 processor 编码，保持一致）
    prompt_inputs = processor(images=image, text=prompt, return_tensors="pt")
    prompt_inputs = {k: v.to(Config.DEVICE) if isinstance(v, torch.Tensor) else v 
                     for k, v in prompt_inputs.items()}
    prompt_len = prompt_inputs["input_ids"].shape[1]
    
    # 调试：检查是否有图像 token（前20个token）
    first_20_tokens = input_ids_full[0][:20].cpu().tolist()
    logger.debug(f"input_ids前20个token: {first_20_tokens}")
    
    # 构建 labels：与 input_ids_full 同长度
    labels = input_ids_full.clone()
    
    # 将 prompt 部分设为 -100（让 loss 忽略，只计算 caption 部分）
    labels[:, :prompt_len] = -100
    
    # 将 padding 位置设为 -100
    pad_token_id = processor.tokenizer.pad_token_id
    if pad_token_id is not None:
        labels[labels == pad_token_id] = -100
    
    # 使用 processor 编码的结果
    forward_inputs = {
        'pixel_values': full_inputs['pixel_values'],  # 使用当前图像的pixel_values
        'input_ids': input_ids_full,
        'attention_mask': attention_mask_full,
    }
    
    # 如果有 qformer_input_ids，也需要保留
    if 'qformer_input_ids' in full_inputs:
        forward_inputs['qformer_input_ids'] = full_inputs['qformer_input_ids']
    if 'qformer_attention_mask' in full_inputs:
        forward_inputs['qformer_attention_mask'] = full_inputs['qformer_attention_mask']
    
    with torch.no_grad():
        try:
            # 使用完整的 input_ids 和 labels
            forward_inputs['labels'] = labels
            
            outputs = model(**forward_inputs)
            
            if hasattr(outputs, 'loss') and outputs.loss is not None:
                loss = outputs.loss.item()
                num_tokens = (labels != -100).sum().item()
                
                if num_tokens > 0:
                    # loss 是平均 cross-entropy loss（per token）
                    # 总 log-likelihood = -loss * num_tokens
                    # 平均 log-likelihood per token = -loss
                    # 
                    # 我们使用归一化的值（平均 per token），因为：
                    # 1. 同一个 region_caption，token 数量相同
                    # 2. 归一化后数值更稳定，不受 token 数量影响
                    # 3. score = L_orig - L_mask = (-loss_orig) - (-loss_mask) = loss_mask - loss_orig
                    #    如果 loss_mask > loss_orig（遮挡后更难），score > 0（正数）✓
                    #    如果 loss_mask < loss_orig（遮挡后更容易），score < 0（负数）✓
                    normalized_log_likelihood = -loss
                    
                    # 调试：记录loss值，用于验证
                    logger.debug(f"计算log-likelihood: loss={loss:.4f}, num_tokens={num_tokens}, L={normalized_log_likelihood:.4f}")
                else:
                    logger.warning("No valid tokens in labels")
                    normalized_log_likelihood = -10.0
            else:
                raise ValueError("Forward did not return loss")
                
        except Exception as e:
            # 如果forward失败，使用generate方法（作为fallback）
            # 改为 logger.warning，这样能看到错误
            logger.warning(f"Forward方法失败 ({e})，使用generate方法")
            try:
                # Fallback: 使用 generate 方法
                # 注意：这里使用原始的 inputs（只有 prompt），不是 forward_inputs
                gen_outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(max_tokens, len(caption.split()) + 10),
                    return_dict_in_generate=True,
                    output_scores=True,
                    do_sample=False,
                    num_beams=1
                )
                
                if hasattr(gen_outputs, 'scores') and gen_outputs.scores:
                    # 使用完整序列中的 caption 部分计算 log-prob
                    # 从 input_ids_full 中提取 caption 部分
                    caption_ids = input_ids_full[0, prompt_len:]  # 跳过 prompt 部分
                    
                    total_log_prob = 0.0
                    valid_tokens = 0
                    
                    for i, target_id in enumerate(caption_ids):
                        if i >= len(gen_outputs.scores):
                            break
                        
                        # 跳过特殊token
                        if pad_token_id and target_id.item() == pad_token_id:
                            continue
                        eos_token_id = processor.tokenizer.eos_token_id
                        if eos_token_id and target_id.item() == eos_token_id:
                            break
                        
                        # 计算log-prob
                        step_logits = gen_outputs.scores[i][0]
                        log_probs = F.log_softmax(step_logits, dim=-1)
                        log_prob = log_probs[target_id.item()].item()
                        
                        total_log_prob += log_prob
                        valid_tokens += 1
                    
                    if valid_tokens > 0:
                        normalized_log_likelihood = total_log_prob / valid_tokens
                    else:
                        logger.warning("Generate fallback: No valid tokens found")
                        normalized_log_likelihood = -10.0
                else:
                    logger.warning("Generate fallback: No scores returned")
                    normalized_log_likelihood = -10.0
            except Exception as e2:
                logger.warning(f"所有方法都失败: {e2}")
                normalized_log_likelihood = -10.0
    
    return normalized_log_likelihood


def compute_scores(model, processor, original_image: Image.Image, prompt: str, 
                   candidates: List[Dict], masked_images: Dict[int, Image.Image]) -> List[Dict]:
    """
    计算每个候选的log-likelihood drop分数
    对每个候选框：遮挡后生成新caption，然后在原图上对比两个caption的log-likelihood
    这样更敏感，能直接反映遮挡对描述的影响
    """
    scores = []
    
    # Step 1: 在原图上生成caption
    logger.info(f"在原图上生成anchor caption...")
    caption_orig = generate_anchor_caption(model, processor, original_image, prompt, len(candidates))
    logger.info(f"原图caption (长度={len(caption_orig)}字符): {caption_orig}")
    
    # Step 2: 计算原图对原图caption的log-likelihood（基准）
    L_orig_orig = compute_log_likelihood(model, processor, original_image, prompt, caption_orig, Config.MAX_TOKENS_FOR_SCORE)
    logger.info(f"L_orig_orig = {L_orig_orig:.4f}")
    
    # Step 3: 对每个候选框，生成遮挡后的caption，然后在原图上对比
    for i, cand in enumerate(candidates):
        if i not in masked_images:
            logger.warning(f"遮挡图 {i} 不存在，跳过")
            continue
        
        bbox = cand['bbox']
        masked_img = masked_images[i]
        
        # 在遮挡图上生成新caption
        logger.info(f"候选 {i}: 在遮挡图上生成caption...")
        caption_mask = generate_anchor_caption(model, processor, masked_img, prompt, len(candidates))
        logger.info(f"候选 {i}: 遮挡图caption (长度={len(caption_mask)}字符) = {caption_mask}")
        
        # 在原图上计算两个caption的log-likelihood
        L_orig_mask = compute_log_likelihood(model, processor, original_image, prompt, caption_mask, Config.MAX_TOKENS_FOR_SCORE)
        
        # 重要性分数 = L_orig_orig - L_orig_mask
        # 如果原图更匹配原图caption（L_orig_orig更大），说明遮挡的object重要
        score = L_orig_orig - L_orig_mask
        
        orig_id = cand.get('orig_id', i)
        cls_name = cand.get('cls_name', f"cls_{cand.get('cls', '?')}")
        scores.append({
            'id': i,  # 在K2中的local id
            'orig_id': orig_id,  # 在K1中的原始id
            'score': float(score),
            'bbox': cand['bbox'],
            'cls_name': cls_name,
            'caption_orig': caption_orig,  # 原图caption
            'caption_mask': caption_mask,  # 遮挡图caption
            'L_orig_orig': float(L_orig_orig),  # 原图对原图caption的log-likelihood
            'L_orig_mask': float(L_orig_mask)  # 原图对遮挡图caption的log-likelihood
        })
        
        logger.info(f"候选 {i} ({cls_name}, orig_id={orig_id}): score={score:.4f} (L_orig_orig={L_orig_orig:.4f}, L_orig_mask={L_orig_mask:.4f})")
    
    return scores




# ==================== Step 7: 排序得到主次主体 ====================
def rank_subjects(scores: List[Dict]) -> List[int]:
    """按分数从大到小排序，返回id列表"""
    sorted_scores = sorted(scores, key=lambda x: x['score'], reverse=True)
    ranking = [item['id'] for item in sorted_scores]
    return ranking


def visualize_ranking(image_path: str, candidates: List[Dict], ranking: List[int], 
                     scores: List[Dict], output_path: str):
    """
    可视化排序结果：画出所有主体的框，标注排名和类别名称
    特别适用于区分多个相同类别的物体（如多个person）
    """
    img = Image.open(image_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    
    # 创建id到分数和类别名称的映射
    score_dict = {s['id']: s['score'] for s in scores}
    cls_name_dict = {s['id']: s.get('cls_name', 'unknown') for s in scores}
    
    # 定义颜色：Primary用红色，Secondary用不同深浅的蓝色
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    # 尝试加载字体
    try:
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            try:
                font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
            except:
                font = ImageFont.load_default()
    except:
        font = ImageFont.load_default()
    
    # 画所有候选框（按排名顺序）
    for rank, cand_id in enumerate(ranking):
        if cand_id >= len(candidates):
            continue
        
        cand = candidates[cand_id]
        bbox = cand['bbox']
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # 确保坐标在图像范围内
        x1 = max(0, min(x1, img.width))
        y1 = max(0, min(y1, img.height))
        x2 = max(0, min(x2, img.width))
        y2 = max(0, min(y2, img.height))
        
        score = score_dict.get(cand_id, 0.0)
        cls_name = cls_name_dict.get(cand_id, cand.get('cls_name', 'unknown'))
        
        # Primary用红色粗框，Secondary用其他颜色细框
        if rank == 0:
            color = 'red'
            width = 5
            rank_label = 'Rank 1 (Primary)'
        else:
            color = colors[min(rank, len(colors) - 1)]
            width = 3
            rank_label = f'Rank {rank + 1}'
        
        # 画框
        draw.rectangle([x1, y1, x2, y2], outline=color, width=width)
        
        # 构建标签：排名 + 类别名称 + 分数
        label = f"{rank_label}: {cls_name} (score={score:.3f})"
        
        # 计算标签位置（在框的上方，如果太靠上就放在框内）
        label_y = max(5, y1 - 25)
        
        # 获取文本边界框
        text_bbox = draw.textbbox((x1, label_y), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # 确保标签不超出图像边界
        label_x = x1
        if x1 + text_width > img.width:
            label_x = img.width - text_width - 5
        
        # 画白色背景（稍微扩大一点，增加可读性）
        padding = 3
        draw.rectangle(
            [text_bbox[0] - padding, text_bbox[1] - padding, 
             text_bbox[2] + padding, text_bbox[3] + padding],
            fill='white',
            outline=color,
            width=2
        )
        
        # 画文本
        draw.text((label_x, label_y), label, fill=color, font=font)
    
    img.save(output_path)
    logger.info(f"保存排序可视化: {output_path} (包含 {len(ranking)} 个主体)")


# ==================== Step 8: 保存结果 ====================
def save_result(image_path: str, candidates: List[Dict], 
               scores: List[Dict], ranking: List[int], output_file: str):
    """保存结果到jsonl文件"""
    # 从scores中提取caption信息（如果有）
    caption_orig = scores[0].get('caption_orig', '') if len(scores) > 0 else ''
    
    result = {
        "image": image_path,
        "prompt": Config.ANCHOR_PROMPT_SINGLE,
        "caption_orig": caption_orig,  # 原图生成的caption
        "candidates": [
            {
                "id": i,  # K2中的local id
                "orig_id": cand.get('orig_id', i),  # K1中的原始id
                "bbox": cand['bbox'],
                "conf": cand['conf'],
                "cls": cand['cls'],
                "cls_name": cand.get('cls_name', f"cls_{cand.get('cls', '?')}")  # 类别名称
            }
            for i, cand in enumerate(candidates)
        ],
        "scores": [
            {
                "id": s['id'],
                "orig_id": s.get('orig_id', s['id']),
                "score": s['score'],
                "cls_name": s.get('cls_name', ''),
                "caption_orig": s.get('caption_orig', ''),
                "caption_mask": s.get('caption_mask', ''),
                "L_orig_orig": s.get('L_orig_orig', None),
                "L_orig_mask": s.get('L_orig_mask', None)
            }
            for s in scores
        ],
        "ranking": ranking,
        "primary_id": ranking[0] if len(ranking) > 0 else None
    }
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    logger.info(f"结果已保存: {image_path}")


def save_ranking_summary(image_path: str, candidates: List[Dict], 
                        scores: List[Dict], ranking: List[int], ranking_file: str):
    """
    保存排名摘要到单独的JSON文件，专门用于查看每个object的排名
    简化格式：只包含图片编号和排名物体
    """
    # 提取图片编号（从路径中提取文件名，去掉扩展名）
    image_name = os.path.basename(image_path)
    image_id = os.path.splitext(image_name)[0]
    
    # 创建简化的排名列表：只包含排名和物体类别
    ranked_objects = []
    for rank, cand_id in enumerate(ranking):
        if cand_id >= len(candidates):
            continue
        
        cand = candidates[cand_id]
        cls_name = cand.get('cls_name', f"cls_{cand.get('cls', '?')}")
        
        ranked_objects.append({
            "rank": rank + 1,  # 1-based排名
            "object": cls_name  # 物体类别名称
        })
    
    # 构建简化的摘要
    summary = {
        "image_id": image_id,
        "ranking": ranked_objects  # 只包含排名和物体类别
    }
    
    # 读取现有文件（如果存在）
    all_summaries = []
    if os.path.exists(ranking_file):
        try:
            with open(ranking_file, 'r', encoding='utf-8') as f:
                all_summaries = json.load(f)
        except:
            all_summaries = []
    
    # 添加新摘要
    all_summaries.append(summary)
    
    # 保存
    with open(ranking_file, 'w', encoding='utf-8') as f:
        json.dump(all_summaries, f, ensure_ascii=False, indent=2)
    
    logger.info(f"排名摘要已保存: {image_id}")


def load_existing_results(output_file: str) -> set:
    """加载已有结果，用于断点续跑"""
    processed_images = set()
    
    if os.path.exists(output_file):
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    processed_images.add(data['image'])
                except:
                    continue
    
    logger.info(f"已处理 {len(processed_images)} 张图片（断点续跑）")
    return processed_images


# ==================== 主流程 ====================
def process_single_image(image_path: str, yolo_model, blip_model, blip_processor, 
                        output_dir: str, results_file: str, save_debug_masks: bool = False):
    """处理单张图片的完整流程"""
    try:
        base_name = Path(image_path).stem
        
        # 只打开一次图像（性能优化）
        original_image = Image.open(image_path).convert('RGB')
        
        # Step 2: YOLO检测（先取 K_FULL 个，得到全物体列表供评估；再取 K1 个做排序流程）
        logger.info(f"处理图片: {image_path}")
        candidates_all = detect_objects(yolo_model, image_path, Config.TARGET_CLASS, Config.K_FULL)
        if len(candidates_all) == 0:
            logger.warning(f"未检测到目标对象: {image_path}")
            return
        # 每张图「全物体」列表（去重类别名），供 eval_chair_benchmark / eval_chair 使用
        objects_in_image_eval = sorted(set(c['cls_name'] for c in candidates_all))
        logger.info(f"全物体列表({len(objects_in_image_eval)}类): {objects_in_image_eval[:10]}{'...' if len(objects_in_image_eval) > 10 else ''}")
        candidates = candidates_all[:Config.K1]
        if len(candidates) > 0:
            logger.info(f"用于排序的候选: {len(candidates)} 个，类别: {[c.get('cls_name', '?') for c in candidates[:5]]}")
        
        # 可视化检测框（可选：注释掉以节省空间，只保留排序结果可视化）
        # debug_box_path = os.path.join(output_dir, "debug_boxes", f"{base_name}.jpg")
        # visualize_boxes(image_path, candidates, debug_box_path)
        
        # Step 3: 候选压缩（使用面积最大法，参考1_detect_mainobj.py）
        candidates_top = compress_candidates(candidates, Config.K2, original_image.size[::-1], Config.COMPRESS_METHOD)  # PIL size是(w,h)，需要反转
        
        logger.info(f"候选数量: {len(candidates)} -> {len(candidates_top)}")
        
        # Step 4: 不需要提前生成anchor caption，会在compute_scores中生成
        
        # Step 5: 生成遮挡图（不保存到磁盘，除非指定）
        # 重要：每个候选都会生成一个独立的遮挡图，每个图只遮挡一个候选的bbox
        logger.info(f"开始生成 {len(candidates_top)} 个遮挡图（每个只遮挡一个候选）...")
        masked_images = create_masked_images(
            original_image,
            candidates_top,
            Config.MASK_METHOD,
            padding_ratio=Config.MASK_PADDING_RATIO,  # 使用配置的padding比例
            save_debug=save_debug_masks,
            output_dir=os.path.join(output_dir, "debug_masks") if save_debug_masks else None,
            base_name=base_name if save_debug_masks else None
        )
        logger.info(f"已生成 {len(masked_images)} 个遮挡图")
        
        # Step 6: 计算分数（动态caption方法：遮挡后生成新caption，然后在原图上对比）
        actual_prompt = Config.ANCHOR_PROMPT_SINGLE if len(candidates_top) == 1 else Config.ANCHOR_PROMPT_MULTIPLE
        logger.info(f"开始计算分数：遮挡后生成新caption，然后在原图上对比...")
        scores = compute_scores(blip_model, blip_processor, original_image, actual_prompt, 
                               candidates_top, masked_images)
        
        if len(scores) == 0:
            logger.warning(f"无法计算分数: {image_path}")
            return
        
        # 检查分数是否合理
        max_score = max(s['score'] for s in scores)
        min_score = min(s['score'] for s in scores)
        logger.info(f"分数范围: [{min_score:.4f}, {max_score:.4f}]")
        
        # 改进的检查：区分"全接近0"和"全为负数"的情况
        if max_score < 0.001:
            if max_score < -0.1:
                logger.warning(f"⚠️ 所有分数为负且较大 ({min_score:.4f} ~ {max_score:.4f})，说明遮挡后log-likelihood反而更高。")
                logger.warning(f"   可能原因：anchor caption是群体描述，或mask不够彻底。")
                logger.warning(f"   建议：检查anchor caption是否包含'group/multiple/several'等词")
            else:
                logger.warning(f"所有分数都接近0，可能有问题: {image_path}")
        elif max_score > 0.1:
            logger.info(f"✅ 分数区分度良好，最高分: {max_score:.4f}")
        
        # Step 7: 排序
        ranking = rank_subjects(scores)
        
        # 可视化排序结果
        ranking_viz_path = os.path.join(output_dir, "debug_boxes", f"{base_name}_ranking.jpg")
        visualize_ranking(image_path, candidates_top, ranking, scores, ranking_viz_path)
        
        # Step 8: 保存结果
        save_result(image_path, candidates_top, scores, ranking, results_file)
        
        # 单独写一份「全物体」列表供评估：eval 时用 --results-file outputs/results_full_objects.jsonl
        full_objects_file = os.path.join(output_dir, "results_full_objects.jsonl")
        with open(full_objects_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps({"image": image_path, "objects_in_image_eval": objects_in_image_eval}, ensure_ascii=False) + '\n')
        
        # Step 9: 保存排名摘要（单独的JSON文件，便于查看）
        ranking_file = os.path.join(output_dir, "ranking_summary.json")
        save_ranking_summary(image_path, candidates_top, scores, ranking, ranking_file)
        
        logger.info(f"完成: {image_path}")
        
    except Exception as e:
        logger.error(f"处理图片失败 {image_path}: {str(e)}", exc_info=True)


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='主体排序器')
    parser.add_argument('--images_dir', type=str, default=Config.IMAGES_DIR, help='图片目录')
    parser.add_argument('--yolo_model', type=str, default=Config.YOLO_MODEL_PATH, help='YOLO模型路径')
    parser.add_argument('--blip_model', type=str, default=Config.INSTRUCTBLIP_MODEL_PATH, help='InstructBLIP模型路径')
    parser.add_argument('--output_dir', type=str, default=Config.OUTPUT_DIR, help='输出目录')
    parser.add_argument('--resume', action='store_true', help='断点续跑')
    parser.add_argument('--save_debug_masks', action='store_true', help='保存遮挡图到磁盘（默认False，节省空间）')
    
    args = parser.parse_args()
    
    # 先创建输出目录（重要：在设置日志之前）
    os.makedirs(args.output_dir, exist_ok=True)
    setup_output_dirs(args.output_dir)
    
    # 设置日志（现在目录已存在）
    global logger
    logger = setup_logging(args.output_dir)
    
    logger.info("=" * 50)
    logger.info("主体排序器启动")
    logger.info("=" * 50)
    
    # Step 1: 工程骨架
    image_list = get_image_list(args.images_dir)
    
    if len(image_list) == 0:
        logger.error("未找到图片文件！")
        return
    
    results_file = os.path.join(args.output_dir, "results.jsonl")
    processed_images = set()
    
    if args.resume:
        processed_images = load_existing_results(results_file)
    
    # 加载模型
    logger.info("加载YOLO模型...")
    yolo_model = YOLO(args.yolo_model)
    yolo_model.to(Config.DEVICE)
    
    logger.info("加载InstructBLIP模型...")
    logger.info("  加载processor...")
    blip_processor = InstructBlipProcessor.from_pretrained(args.blip_model)
    logger.info("  加载model权重...")
    blip_model = InstructBlipForConditionalGeneration.from_pretrained(args.blip_model)
    logger.info(f"  移动模型到设备: {Config.DEVICE}...")
    blip_model.to(Config.DEVICE)
    logger.info("  设置模型为eval模式...")
    blip_model.eval()
    
    # 如果是CUDA，同步一下确保模型加载完成
    if Config.DEVICE == "cuda":
        import torch
        torch.cuda.synchronize()
        logger.info("  模型已加载到GPU并同步完成")
    
    logger.info(f"使用设备: {Config.DEVICE}")
    
    # 模型预热：进行一次dummy forward pass，触发可能的延迟初始化
    logger.info("预热模型（第一次forward pass可能需要编译）...")
    try:
        dummy_image = Image.new('RGB', (224, 224), color='white')
        dummy_inputs = blip_processor(images=dummy_image, text="test", return_tensors="pt").to(Config.DEVICE)
        with torch.no_grad():
            _ = blip_model.generate(**dummy_inputs, max_new_tokens=5, do_sample=False)
        if Config.DEVICE == "cuda":
            torch.cuda.synchronize()
        logger.info("模型预热完成")
    except Exception as e:
        logger.warning(f"模型预热时出现警告（可忽略）: {e}")
    
    logger.info("开始处理图片...")
    
    # 处理每张图片
    for image_path in tqdm(image_list, desc="处理图片"):
        # 跳过已处理的图片
        if args.resume and image_path in processed_images:
            logger.info(f"跳过已处理: {image_path}")
            continue
        
        process_single_image(
            image_path, 
            yolo_model, 
            blip_model, 
            blip_processor,
            args.output_dir,
            results_file,
            save_debug_masks=args.save_debug_masks
        )
    
    logger.info("=" * 50)
    logger.info("所有图片处理完成！")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
