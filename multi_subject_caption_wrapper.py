"""
多主体图像描述生成 Wrapper（LLaVA 版）
- 单一区域策略：bbox→初始 patch seed（step1），再逐圈加入空间邻接环上的 patch，直至邻接环为空（通常铺满整张 patch 网格，触边自然停止）（step3）。
- 无 bbox 时 step1 使用 attention/中心点 fallback（与旧版一致）。

使用方法:
    from multi_subject_caption_wrapper import MultiSubjectCaptionWrapper
    # LLaVA: tokenizer, model, image_processor 来自 load_pretrained_model
    wrapper = MultiSubjectCaptionWrapper(model=model, tokenizer=tokenizer, image_processor=image_processor, device="cuda")
    result = wrapper.generate(image, subjects=["person", "dog"], subject_bboxes={...})
"""

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from typing import List, Dict, Tuple, Set, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

# 按传入顺序最多参与描述的主体数（与 run_multi_subject_caption 统计档位一致）
MAX_EXPAND_SUBJECTS = 3

# 与五方法完全一致：必须用 conv_templates["vicuna_v1"] 对话模板生成 prompt（含 system），否则 LLaVA 易输出中文。
# 注意：这里的 vicuna_v1 是 LLaVA 内置的「对话格式名」，不是 Vicuna 权重路径；权重路径由 load_pretrained_model(model_path, model_base, …) 决定，
# model_path 为合并 checkpoint 时 model_base 应为 None（勿与「删 Vicuna 模板」混淆）。
# 参考 run_baseline: conv_templates["vicuna_v1"] + DEFAULT_IMAGE_TOKEN + "\n" + CAPTION_PROMPT
# 不在 prompt 里加 "Answer in English" 等，与 baseline 的 "Describe this image." 风格一致


class MultiSubjectCaptionWrapper:
    """
    多主体图像描述生成 Wrapper（LLaVA）
    固定流程：bbox→patch 初始区（step1）→ 逐圈邻接扩展至环为空（step3）→ 分段描述（step4）→ 纯文本整合（step5）。
    """

    def __init__(
        self,
        model,
        tokenizer,
        image_processor,
        device: str = "cuda",
        seed_k: int = 5,
        max_rounds: int = 2,
        expansion_threshold: float = 0.01,
        max_region_size_ratio: float = 5.0,
        use_spatial_neighbor: bool = True,
        generate_per_round: int = 2,
        max_subjects: Optional[int] = None,
        # 缓解描述被 max_new_tokens 截断：可按显存/速度调小
        max_new_tokens_gain: int = 64,
        max_new_tokens_memory: int = 160,
        max_new_tokens_subject: int = 256,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.device = device
        self.seed_k = seed_k
        self.max_rounds = max_rounds
        self.expansion_threshold = expansion_threshold
        self.max_region_size_ratio = max_region_size_ratio
        self.use_spatial_neighbor = use_spatial_neighbor
        self.generate_per_round = generate_per_round
        self.max_subjects = max_subjects
        self.max_new_tokens_gain = max_new_tokens_gain
        self.max_new_tokens_memory = max_new_tokens_memory
        self.max_new_tokens_subject = max_new_tokens_subject
        # LLaVA 使用 CLIP ViT：336/14=24 -> 24x24
        self.patch_size = 14

    def _llava_tokenizer_image_token(self, prompt: str) -> torch.Tensor:
        """
        与 llava.mm_utils.tokenizer_image_token 逻辑一致，但用 encode 拼 token id。
        新版 transformers 下 tokenizer(chunk) 可能触发 padding / batched tensor 相关报错。
        """
        from llava.constants import IMAGE_TOKEN_INDEX

        tokenizer = self.tokenizer
        image_token_index = IMAGE_TOKEN_INDEX
        prompt_chunks = []
        for chunk in prompt.split("<image>"):
            if len(chunk) > 0:
                prompt_chunks.append(
                    tokenizer.encode(chunk, add_special_tokens=False)
                )
            else:
                prompt_chunks.append([tokenizer.bos_token_id])

        def insert_separator(X, sep):
            return [ele for sublist in zip(X, [sep] * len(X)) for ele in sublist][:-1]

        input_ids = []
        offset = 0
        if (
            len(prompt_chunks) > 0
            and len(prompt_chunks[0]) > 0
            and prompt_chunks[0][0] == tokenizer.bos_token_id
        ):
            offset = 1
            input_ids.append(prompt_chunks[0][0])

        for x in insert_separator(prompt_chunks, [image_token_index] * (offset + 1)):
            input_ids.extend(x[offset:])

        return torch.tensor(input_ids, dtype=torch.long)

    def _preprocess_pixel_values_bchw(self, image: Image.Image) -> torch.Tensor:
        """
        返回 float32、形状 [1, C, H, W] 的 CPU tensor。

        优先 return_tensors=None，避免 transformers BatchFeature.convert_to_tensors('pt')
        在 NumPy 2.x 与旧版 torch/torchvision 组合下报错；且绝不依赖 torch.from_numpy
        （该环境下常出现 “Numpy is not available”）。
        """
        proc = self.image_processor

        def _to_bchw_torch(pv) -> torch.Tensor:
            if isinstance(pv, torch.Tensor):
                t = pv.float().contiguous()
            else:
                arr = np.asarray(pv, dtype=np.float32)
                if arr.ndim == 3:
                    arr = arr[np.newaxis, ...]
                elif arr.ndim != 4:
                    raise ValueError(f"unexpected pixel_values ndim={arr.ndim}")
                # 用 Python 嵌套列表转 tensor，绕过 torch.from_numpy 与损坏的 NumPy C API 桥接
                t = torch.tensor(arr.tolist(), dtype=torch.float32)
            if t.dim() == 3:
                t = t.unsqueeze(0)
            return t.contiguous()

        try:
            out = proc.preprocess(image, return_tensors=None)
            return _to_bchw_torch(out["pixel_values"])
        except Exception as e:
            logger.debug("_preprocess_pixel_values_bchw: return_tensors=None 失败，回退 pt: %s", e)
            out = proc.preprocess(image, return_tensors="pt")
            t = out["pixel_values"]
            if isinstance(t, torch.Tensor):
                if t.dim() == 3:
                    t = t.unsqueeze(0)
                return t.float().contiguous()
            return _to_bchw_torch(t)

    def _llava_build_prompt(self, prompt_text: str):
        """
        与五方法一致：用 vicuna_v1 对话模板生成完整 prompt（含 system），否则 LLaVA 易输出中文。
        返回 (full_prompt_str, stop_str)。
        """
        from llava.constants import DEFAULT_IMAGE_TOKEN
        from llava.conversation import conv_templates, SeparatorStyle
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + prompt_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
        stop_str = conv.sep2 if conv.sep_style == SeparatorStyle.TWO else conv.sep
        return prompt, stop_str

    def _llava_prepare_inputs(self, image: Image.Image, prompt_text: str):
        """
        构造 LLaVA 输入。返回 (inputs_dict, input_len, stop_str)。
        inputs_dict 含 input_ids, images, attention_mask；input_len 用于解码时只取新 token；stop_str 用于截断输出。
        """
        prompt, stop_str = self._llava_build_prompt(prompt_text)
        input_ids = self._llava_tokenizer_image_token(prompt).unsqueeze(0).to(self.device)
        images = self._preprocess_pixel_values_bchw(image).half().to(self.device)
        attention_mask = torch.ones_like(input_ids, dtype=torch.long, device=self.device)
        input_len = input_ids.shape[1]
        return {"input_ids": input_ids, "images": images, "attention_mask": attention_mask}, input_len, stop_str

    def _llava_generate(
        self,
        image: Image.Image,
        prompt_text: str,
        max_new_tokens: Optional[int] = None,
        num_beams: int = 3,
    ) -> str:
        """对单张图+文本 prompt 做 LLaVA 生成，返回解码后的回答（不含 prompt）。与 baseline 一致使用 vicuna_v1 模板与 stop_str。"""
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens_subject
        inputs, input_len, stop_str = self._llava_prepare_inputs(image, prompt_text)
        with torch.no_grad():
            # do_sample=False 时显式覆盖 temperature/top_p，避免 transformers 报 “only used in sample-based” 的警告
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                num_beams=num_beams,
                use_cache=True,
                temperature=1.0,
                top_p=1.0,
            )
        out = self.tokenizer.batch_decode(output_ids[:, input_len:], skip_special_tokens=True)
        text = (out[0] or "").strip()
        if stop_str and text.endswith(stop_str):
            text = text[:-len(stop_str)].strip()
        # 去掉可能重复的 prompt 前缀
        if prompt_text and text.lower().startswith(prompt_text.lower()):
            text = text[len(prompt_text):].strip()
        if text and text[0] in [':', '-', '.']:
            text = text[1:].strip()
        return text

    def _is_invalid_caption(self, text: str) -> bool:
        """检测韩文/中文占位符或无效输出（如 번역결과），避免进入后续步骤。"""
        if not text or len(text.strip()) < 3:
            return True
        t = text.strip()
        # 已知占位符或翻译结果标签
        if "번역결과" in t or "번역 결과" in t:
            return True
        # 几乎全是非英文且很短，视为无效
        ascii_count = sum(1 for c in t if ord(c) < 128)
        if len(t) < 30 and ascii_count < len(t) * 0.3:
            return True
        return False

    def _sanitize_caption(self, text: str, context: str = "") -> str:
        """若为无效/占位符输出，返回简短英文占位，避免记忆/整合阶段混入韩文或乱码。"""
        if not self._is_invalid_caption(text):
            return text.strip()
        return "This subject is visible in the image."

    def _get_image_patches(self, image: Image.Image) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        获取图像的 patch 表示（LLaVA 视觉塔：CLIP ViT）。
        返回: (patch_features, (H, W))
        """
        pixel_values = self._preprocess_pixel_values_bchw(image).to(self.device)
        if pixel_values.dim() == 3:
            pixel_values = pixel_values.unsqueeze(0)
        vision_tower = self.model.get_model().get_vision_tower()
        if vision_tower is None:
            raise RuntimeError("LLaVA model has no vision tower")
        with torch.no_grad():
            vision_outputs = vision_tower(pixel_values)
            # LLaVA 视觉塔可能直接返回 tensor（如 spin），或返回带 .last_hidden_state 的对象
            if isinstance(vision_outputs, torch.Tensor):
                patch_features = vision_outputs[0] if vision_outputs.dim() > 2 else vision_outputs
            else:
                patch_features = vision_outputs.last_hidden_state[0]
            num_tokens = patch_features.shape[0]
        side = int(num_tokens ** 0.5)
        if side * side == num_tokens:
            patch_grid = (side, side)
        else:
            patch_grid = (24, 24)
        return patch_features, patch_grid
    
    def _get_patch_attention(
        self,
        image: Image.Image,
        text_prompt: str,
        return_attention: bool = True
    ) -> torch.Tensor:
        """
        LLaVA 无 Q-Former，无法做 text→patch cross-attention。
        返回均匀 attention 作为 fallback（step1 无 bbox 时用中心 patch）。
        """
        _, patch_grid = self._get_image_patches(image)
        num_patches = patch_grid[0] * patch_grid[1]
        return torch.ones(num_patches, device=self.device) / num_patches
    
    def _get_spatial_neighbors(
        self, 
        patch_idx: int, 
        patch_grid: Tuple[int, int]
    ) -> List[int]:
        """
        获取patch的空间邻域（8-neighborhood）
        patch_idx: 线性索引
        patch_grid: (H, W) - patch网格尺寸
        返回: 邻居patch的索引列表
        """
        h, w = patch_grid
        row = patch_idx // w
        col = patch_idx % w
        
        neighbors = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < h and 0 <= nc < w:
                    neighbors.append(nr * w + nc)
        
        return neighbors
    
    def _get_ring_patches(
        self,
        region: Set[int],
        patch_grid: Tuple[int, int],
        all_patches: Set[int]
    ) -> Set[int]:
        """
        获取区域周围一圈的patch（Ring）
        region: 当前区域的patch索引集合
        patch_grid: patch网格尺寸
        all_patches: 所有patch的集合
        返回: Ring区域的patch索引集合
        """
        ring = set()
        for patch_idx in region:
            neighbors = self._get_spatial_neighbors(patch_idx, patch_grid)
            for neighbor_idx in neighbors:
                if neighbor_idx not in region and neighbor_idx in all_patches:
                    ring.add(neighbor_idx)
        
        return ring
    
    def _bbox_to_patches(
        self,
        bbox: List[float],
        image_size: Tuple[int, int],
        patch_grid: Tuple[int, int]
    ) -> Set[int]:
        """
        将bbox转换为patch索引集合
        
        输入:
            bbox: [x1, y1, x2, y2] 边界框坐标
            image_size: (width, height) 图像尺寸
            patch_grid: (H, W) patch网格尺寸
        
        返回:
            patch_indices: bbox覆盖的所有patch索引集合
        """
        x1, y1, x2, y2 = bbox
        img_w, img_h = image_size
        h_patches, w_patches = patch_grid
        
        # 归一化坐标到[0, 1]（如果bbox是归一化的）
        if x2 <= 1.0 and y2 <= 1.0:
            x1, y1, x2, y2 = x1 * img_w, y1 * img_h, x2 * img_w, y2 * img_h
        
        # 确保坐标在图像范围内
        x1, y1 = max(0, int(x1)), max(0, int(y1))
        x2, y2 = min(img_w, int(x2)), min(img_h, int(y2))
        
        # 计算每个patch的像素尺寸
        patch_w = img_w / w_patches
        patch_h = img_h / h_patches
        
        # 计算bbox覆盖的patch范围
        col_start = int(x1 / patch_w)
        col_end = int((x2 - 1) / patch_w) + 1  # -1确保不包含边界外的patch
        row_start = int(y1 / patch_h)
        row_end = int((y2 - 1) / patch_h) + 1
        
        # 确保在有效范围内
        col_start = max(0, min(col_start, w_patches - 1))
        col_end = max(0, min(col_end, w_patches))
        row_start = max(0, min(row_start, h_patches - 1))
        row_end = max(0, min(row_end, h_patches))
        
        # 收集所有patch索引
        patch_indices = set()
        for row in range(row_start, row_end):
            for col in range(col_start, col_end):
                patch_idx = row * w_patches + col
                patch_indices.add(patch_idx)
        
        return patch_indices

    def _static_subject_base_name(self, subject: str) -> str:
        return subject.split("_")[0] if "_" in subject and subject.split("_")[-1].isdigit() else subject

    def step1_subject_to_patch_mapping(
        self,
        image: Image.Image,
        subjects: List[str],
        patch_grid: Tuple[int, int],
        subject_bboxes: Optional[Dict[str, List[float]]] = None
    ) -> Dict[str, Set[int]]:
        """
        步骤1: 主体 → patch 映射
        
        输入:
            image: 输入图像
            subjects: 主体列表 [s1, s2, ..., sM]
            patch_grid: patch网格尺寸 (H, W)
            subject_bboxes: {subject: [x1, y1, x2, y2]} 可选的bbox信息
        
        返回:
            subject_seeds: {subject: set of patch indices}
        """
        subject_seeds = {}
        num_patches = patch_grid[0] * patch_grid[1]
        image_size = image.size  # (width, height)
        
        logger.info(f"步骤1: 为主体映射初始patches...")
        
        for subject in subjects:
            # 提取主体名称（去掉可能的序号后缀，如 "person_1" -> "person"）
            subject_base_name = subject.split('_')[0] if '_' in subject and subject.split('_')[-1].isdigit() else subject
            
            if subject_bboxes and subject in subject_bboxes:
                # 优先使用bbox：将bbox内的所有patch都包含进来
                bbox = subject_bboxes[subject]
                seed_patches = self._bbox_to_patches(bbox, image_size, patch_grid)
                logger.info(f"  {subject} ({subject_base_name}): 从bbox获取 {len(seed_patches)} 个patches (bbox={bbox})")
            else:
                # Fallback: 使用attention方法找seed（但可能触发递归错误）
                logger.warning(f"  {subject} ({subject_base_name}): 未提供bbox，尝试使用attention方法 (K={self.seed_k})...")
                logger.warning(f"    注意：attention方法可能在某些情况下触发递归错误，建议提供bbox")
                
                try:
                    # 使用基础名称生成prompt（避免 "person_1" 这样的名称）
                    subject_prompt = f"the {subject_base_name}"
                    patch_attention = self._get_patch_attention(image, subject_prompt)
                    
                    # 选择Top-K patch作为seed
                    topk_values, topk_indices = torch.topk(patch_attention, k=min(self.seed_k, num_patches))
                    seed_patches = set(topk_indices.cpu().tolist())
                    logger.info(f"  {subject} ({subject_base_name}): 选择了 {len(seed_patches)} 个seed patches")
                except RecursionError as e:
                    logger.error(f"  {subject} ({subject_base_name}): attention方法触发递归错误: {e}")
                    logger.error(f"    无法为该主体生成seed，跳过该主体")
                    # 使用一个默认的小区域（中心区域）
                    center_patch = num_patches // 2
                    seed_patches = {center_patch}
                    logger.warning(f"  {subject} ({subject_base_name}): 使用默认中心patch作为seed")
                except Exception as e:
                    logger.error(f"  {subject} ({subject_base_name}): attention方法失败: {e}")
                    center_patch = num_patches // 2
                    seed_patches = {center_patch}
                    logger.warning(f"  {subject} ({subject_base_name}): 使用默认中心patch作为seed")
            
            subject_seeds[subject] = seed_patches
        
        return subject_seeds
    
    def _compute_gain_score(
        self,
        image: Image.Image,
        region: Set[int],
        candidate_patch: int,
        subject: str,
        patch_grid: Tuple[int, int],
        subject_caption: Optional[str] = None
    ) -> float:
        """
        计算添加候选patch后的增益分数
        Δ_i(p) = L(I, R_i ∪ {p} → y_i) - L(I, R_i → y_i)
        
        使用实际计算log-likelihood的方法
        """
        if candidate_patch in region:
            return 0.0
        
        # 提取主体基础名称
        subject_base_name = subject.split('_')[0] if '_' in subject and subject.split('_')[-1].isdigit() else subject
        
        # 统一定义subject_prompt（无论subject_caption是否为None都需要）
        subject_prompt = f"Describe the {subject_base_name} in the image. Focus only on this subject."
        
        # 如果没有提供subject_caption，先生成一个
        if subject_caption is None:
            masked_image_region = self._mask_patches(image, region, patch_grid)
            subject_caption = self._llava_generate(
                masked_image_region, subject_prompt, max_new_tokens=self.max_new_tokens_gain, num_beams=2
            )
            # 清理
            if subject_caption.lower().startswith(subject_prompt.lower()):
                subject_caption = subject_caption[len(subject_prompt):].strip()
        
        # 计算两个masked图像的log-likelihood
        # 1. 当前region
        masked_image_region = self._mask_patches(image, region, patch_grid)
        L_region = self._compute_log_likelihood(masked_image_region, subject_prompt, subject_caption)
        
        # 2. region + candidate_patch
        region_with_candidate = region.copy()
        region_with_candidate.add(candidate_patch)
        masked_image_expanded = self._mask_patches(image, region_with_candidate, patch_grid)
        L_expanded = self._compute_log_likelihood(masked_image_expanded, subject_prompt, subject_caption)
        
        # 增益 = L_expanded - L_region
        gain = L_expanded - L_region
        
        return gain
    
    def _compute_log_likelihood(
        self,
        image: Image.Image,
        prompt: str,
        caption: str
    ) -> float:
        """
        计算给定图像和 caption 的 log-likelihood（LLaVA：teacher forcing，labels 只算 caption 部分）
        """
        from llava.constants import IGNORE_INDEX
        try:
            # prompt_len = vicuna_v1 模板下 prompt 的 token 长度
            _, prompt_len, _ = self._llava_prepare_inputs(image, prompt)
            # 完整序列与 baseline 一致：同一模板的 prompt + caption
            prompt_str, _ = self._llava_build_prompt(prompt)
            full_user_assistant = prompt_str + caption
            input_ids_full = self._llava_tokenizer_image_token(full_user_assistant).unsqueeze(0).to(self.device)
            image_tensor = self._preprocess_pixel_values_bchw(image).half().to(self.device)
            attention_mask = torch.ones_like(input_ids_full, device=self.device)
            labels = input_ids_full.clone()
            labels[:, :prompt_len] = IGNORE_INDEX
            with torch.no_grad():
                outputs = self.model(
                    input_ids=input_ids_full,
                    images=image_tensor,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_dict=True,
                )
                loss = outputs.loss.item()
                return -loss
        except Exception as e:
            logger.warning(f"计算log-likelihood失败: {e}")
            return 0.0
    
    def step3_rotating_expansion(
        self,
        image: Image.Image,
        subject_seeds: Dict[str, Set[int]],
        patch_grid: Tuple[int, int],
        subjects_order: Optional[List[str]] = None
    ) -> Tuple[Dict[str, Set[int]], Dict[str, str]]:
        """
        步骤3: 按环逐圈扩展，直到邻接环为空（触边后自然停止）。
        
        输入:
            image: 输入图像
            subject_seeds: 每个主体的初始 patch 区域（来自 bbox 或 attention seed）
            patch_grid: patch 网格尺寸
            subjects_order: 主体顺序（按重要性排名）
        
        返回:
            subject_regions: 扩展后的区域
        """
        subject_regions = {s: seed.copy() for s, seed in subject_seeds.items()}
        if subjects_order:
            subjects = [s for s in subjects_order if s in subject_seeds]
            for s in subject_seeds.keys():
                if s not in subjects:
                    subjects.append(s)
        else:
            subjects = list(subject_seeds.keys())
        num_patches = patch_grid[0] * patch_grid[1]
        
        logger.info(f"步骤3: 逐圈扩展（主体数={len(subjects)}，直到邻接环为空为止）")
        logger.info(f"  主体顺序（按重要性排名）: {subjects}")
        logger.info(f"  初始区域大小: {[f'{s}={len(subject_seeds[s])}patches' for s in subjects]}")
        
        subject_memories = {}
        
        for subject_idx, subject in enumerate(subjects):
            region = subject_regions[subject].copy()
            round_idx = 0
            while True:
                ring = self._get_ring_patches(region, patch_grid, set(range(num_patches)))
                if not ring:
                    break
                region.update(ring)
                round_idx += 1
            subject_regions[subject] = region
            logger.info(
                f"  {subject} (排名{subject_idx+1}): 扩展轮次={round_idx}，最终区域 {len(region)}/{num_patches} patches"
            )
            
            subject_base_name = subject.split('_')[0] if '_' in subject and subject.split('_')[-1].isdigit() else subject
            
            masked_image_region = self._mask_patches(image, region, patch_grid)
            subject_prompt = f"Describe the {subject_base_name} in the image. Focus only on this subject."
            temp_caption = self._llava_generate(
                masked_image_region, subject_prompt, max_new_tokens=self.max_new_tokens_memory, num_beams=2
            )
            subject_memories[subject] = self._sanitize_caption(temp_caption, subject)
            logger.info(f"  {subject}: 识别结果（记忆）: {(subject_memories[subject])[:80]}...")
        
        # 输出最终区域大小
        for subject, region in subject_regions.items():
            logger.info(f"  {subject}: 最终区域大小 = {len(region)} patches (初始={len(subject_seeds[subject])}patches)")
        
        return subject_regions, subject_memories
    
    def _mask_patches(
        self,
        image: Image.Image,
        visible_patches: Set[int],
        patch_grid: Tuple[int, int]
    ) -> Image.Image:
        """
        创建masked图像：只保留visible_patches可见，其余mask为黑色
        """
        # 确保输入图像有效
        if image is None or image.size[0] == 0 or image.size[1] == 0:
            logger.warning("输入图像无效，返回原图")
            return image
        
        img_array = np.array(image).copy()
        h, w = img_array.shape[:2]
        
        # 确保图像尺寸有效
        if h == 0 or w == 0:
            logger.warning("图像尺寸无效，返回原图")
            return image
        
        # 计算每个patch对应的像素区域
        patch_h = max(1, h // patch_grid[0])  # 确保至少为1
        patch_w = max(1, w // patch_grid[1])  # 确保至少为1
        
        # 创建mask（初始全黑）
        mask = np.zeros((h, w), dtype=bool)
        
        # 标记可见的patch区域
        for patch_idx in visible_patches:
            # 确保patch_idx在有效范围内
            max_patch_idx = patch_grid[0] * patch_grid[1] - 1
            if patch_idx < 0 or patch_idx > max_patch_idx:
                continue
            
            row = patch_idx // patch_grid[1]
            col = patch_idx % patch_grid[1]
            
            # 确保row和col在有效范围内
            if row >= patch_grid[0] or col >= patch_grid[1]:
                continue
            
            y1 = row * patch_h
            y2 = min((row + 1) * patch_h, h)
            x1 = col * patch_w
            x2 = min((col + 1) * patch_w, w)
            
            # 确保坐标有效
            if y2 > y1 and x2 > x1:
                mask[y1:y2, x1:x2] = True
        
        # 应用mask（修复维度问题）
        mask3 = mask[:, :, None]  # [H, W, 1]
        img_array = img_array * mask3  # [H, W, 3] 广播正确
        
        # 确保输出数组有效
        if img_array.size == 0:
            logger.warning("masked图像为空，返回原图")
            return image
        
        try:
            result_image = Image.fromarray(img_array.astype(np.uint8))
            return result_image
        except Exception as e:
            logger.warning(f"创建masked图像失败: {e}，返回原图")
            return image
    
    def step4_subject_captioning(
        self,
        image: Image.Image,
        subject_regions: Dict[str, Set[int]],
        patch_grid: Tuple[int, int],
        expansion_rounds: int,
        subject_memories: Optional[Dict[str, str]] = None,
        subjects_order: Optional[List[str]] = None
    ) -> Dict[str, str]:
        """
        步骤4: 主体分段描述（使用记忆机制）
        
        输入:
            image: 输入图像
            subject_regions: 每个主体的区域
            patch_grid: patch网格尺寸
            expansion_rounds: 保留参数兼容，当前不做区域扩展，此处未使用
            subject_memories: 之前主体的识别结果（作为记忆）
            subjects_order: 主体顺序（用于构建记忆上下文）
        
        返回:
            subject_captions: {subject: caption}
        """
        subject_captions = {}
        
        logger.info(f"步骤4: 生成主体分段描述（使用记忆机制）...")
        
        # 确定处理顺序（按排名顺序）
        if subjects_order:
            processing_order = [s for s in subjects_order if s in subject_regions]
            for s in subject_regions.keys():
                if s not in processing_order:
                    processing_order.append(s)
        else:
            processing_order = list(subject_regions.keys())
        
        # 按顺序处理每个主体，使用之前主体的记忆
        for subject_idx, subject in enumerate(processing_order):
            region = subject_regions[subject]
            
            # 提取主体名称（去掉可能的序号后缀）
            subject_base_name = subject.split('_')[0] if '_' in subject and subject.split('_')[-1].isdigit() else subject
            
            # 创建masked图像（只保留该主体的区域可见）
            masked_image = self._mask_patches(image, region, patch_grid)
            
            # 构建prompt，包含之前主体的记忆（如果有）
            # 不限制记忆长度，使用完整的记忆内容
            memory_context = ""
            if subject_memories and subject_idx > 0:
                # 收集之前所有主体的记忆（不截断，使用完整内容）
                previous_memories = []
                for prev_subject in processing_order[:subject_idx]:
                    if prev_subject in subject_memories:
                        prev_base_name = prev_subject.split('_')[0] if '_' in prev_subject and prev_subject.split('_')[-1].isdigit() else prev_subject
                        prev_memory = subject_memories[prev_subject]
                        # 使用完整的记忆内容，不截断
                        previous_memories.append(f"{prev_base_name}: {prev_memory}")
                
                if previous_memories:
                    memory_context = f"\n\nPreviously identified subjects:\n" + "\n".join(previous_memories) + "\n\n"
                    total_memory_length = sum(len(m) for m in previous_memories)
                    logger.info(f"  {subject}: 使用 {len(previous_memories)} 个之前主体的记忆 (总长度: {total_memory_length} 字符)")
            
            # 构造主体导向的prompt（与 instructblip 版本一致）
            if '_' in subject and subject.split('_')[-1].isdigit():
                instance_num = int(subject.split('_')[-1]) + 1  # 从1开始计数
                base_prompt = f"Describe the {subject_base_name} (instance {instance_num}) in detail. Include its appearance, position, pose, and any distinctive features. Focus only on this specific {subject_base_name}."
            else:
                base_prompt = f"Describe the {subject_base_name} in detail. Include its appearance, position, pose, and any distinctive features. Focus only on this {subject_base_name}."
            
            # 如果有记忆，添加到prompt中
            # 强调：只描述当前主体，不要与其它主体合并或省略；相似主体（如多个人、多只狗）也要单独描述
            if memory_context:
                subject_prompt = (
                    memory_context + base_prompt +
                    " Describe only THIS subject. Do not merge it with or omit it in favor of other subjects. "
                    "Even if other subjects are similar (e.g., another person or another dog), describe this one as a separate entity with its own position, appearance, and pose."
                )
            else:
                subject_prompt = base_prompt
            
            # 生成主体描述（LLaVA）
            generated_text = self._llava_generate(
                masked_image, subject_prompt, max_new_tokens=self.max_new_tokens_subject, num_beams=3
            )
            subject_captions[subject] = self._sanitize_caption(generated_text, subject)
            logger.info(f"  {subject} ({subject_base_name}): {(subject_captions[subject])[:100]}...")
            
            # 更新记忆（如果之前没有记忆，使用当前生成的描述）
            if subject not in subject_memories or not subject_memories[subject]:
                subject_memories[subject] = generated_text
        
        return subject_captions
    
    def step5_final_integration(
        self,
        subject_captions: Dict[str, str],
    ) -> str:
        """
        步骤5: 最后整合与去冗余：生成"最终描述"（纯文本，不再传入图像）
        
        输入:
            subject_captions: 每个主体的描述
        
        返回:
            final_caption: 整合后的最终描述
        """
        logger.info("步骤5: 纯文本整合去冗余，生成最终描述（不传入图像）...")
        
        # 过滤掉空的主体描述
        valid_subject_captions = {s: c for s, c in subject_captions.items() if c and c.strip()}
        
        if not valid_subject_captions:
            logger.warning("没有有效的主体描述，返回空字符串")
            return ""

        final_caption = self._smart_concatenate(valid_subject_captions)
        logger.info(f"  最终描述: {final_caption[:200]}...")
        return final_caption
    
    def _smart_concatenate(
        self,
        subject_captions: Dict[str, str],
    ) -> str:
        """
        改进的拼接方法：去冗余、合并相似句子
        """
        # 收集所有句子
        all_sentences = []
        
        # 从主体描述中提取句子
        for subject, caption in subject_captions.items():
            # 按句号、感叹号、问号分割
            sentences = [s.strip() for s in caption.replace('!', '.').replace('?', '.').split('.') if s.strip()]
            all_sentences.extend(sentences)
        
        # 去冗余：移除重复或高度相似的句子
        unique_sentences = []
        seen_lower = set()
        
        for sent in all_sentences:
            sent_lower = sent.lower().strip()
            # 检查是否与已有句子相似（简单的子串匹配）
            is_duplicate = False
            for seen in seen_lower:
                # 如果新句子是已有句子的子串，或已有句子是新句子的子串，认为是重复
                if len(sent_lower) > 20 and len(seen) > 20:  # 只对长句子做相似度检查
                    if sent_lower in seen or seen in sent_lower:
                        is_duplicate = True
                        break
                    # 简单的单词重叠检查（如果超过80%的单词相同，认为是重复）
                    sent_words = set(sent_lower.split())
                    seen_words = set(seen.split())
                    if len(sent_words) > 0 and len(seen_words) > 0:
                        overlap = len(sent_words & seen_words) / max(len(sent_words), len(seen_words))
                        if overlap > 0.8:
                            is_duplicate = True
                            break
            
            if not is_duplicate and sent_lower:
                unique_sentences.append(sent)
                seen_lower.add(sent_lower)
        
        # 合并成最终描述
        final_caption = ". ".join(unique_sentences)
        if final_caption and not final_caption.endswith('.'):
            final_caption += "."
        
        return final_caption
    
    def generate(
        self,
        image: Image.Image,
        subjects: List[str],
        subject_bboxes: Optional[Dict[str, List[float]]] = None
    ) -> Dict:
        """
        完整的多主体描述生成流程。
        - step1：bbox→初始 patch（无 bbox 时用 attention/中心 fallback）
        - step3：逐圈扩展邻接 patch，邻接环为空则停（通常铺满全图 patch 网格）
        按顺序最多保留前 MAX_EXPAND_SUBJECTS 个主体；不生成背景。

        输入:
            image: 输入图像
            subjects: 主体列表（按顺序最多取前 MAX_EXPAND_SUBJECTS 个参与描述）

        返回:
            result: {
                'subject_regions': {subject: set of patch indices},
                'subject_captions': {subject: caption},
                'final_caption': str
            }
        """
        subjects = list(subjects or [])
        if len(subjects) > MAX_EXPAND_SUBJECTS:
            logger.info(
                f"主体数 {len(subjects)} 超过上限，按顺序仅保留前 {MAX_EXPAND_SUBJECTS} 个: "
                f"{subjects[:MAX_EXPAND_SUBJECTS]}"
            )
            subjects = subjects[:MAX_EXPAND_SUBJECTS]

        if subject_bboxes:
            subject_bboxes = {k: v for k, v in (subject_bboxes or {}).items() if k in subjects}
        else:
            logger.warning(
                "未提供 subject_bboxes：step1 将使用 attention/中心点 fallback，建议 ranking 管线传入 bbox"
            )

        logger.info("=" * 50)
        logger.info(
            "开始多主体描述生成（主体数: %d）：step1 bbox→patch seed；step3 逐圈邻接扩展至环为空",
            len(subjects),
        )
        logger.info("=" * 50)

        # 获取patch信息
        _, patch_grid = self._get_image_patches(image)
        logger.info(f"图像patch网格: {patch_grid}")
        
        # 步骤1: 主体 → patch 映射（传入bbox信息）
        subject_seeds = self.step1_subject_to_patch_mapping(image, subjects, patch_grid, subject_bboxes)
        
        # 步骤3: 逐圈扩展至邻接环为空（触边停止）；按顺序生成记忆
        subject_regions, subject_memories = self.step3_rotating_expansion(image, subject_seeds, patch_grid, subjects_order=subjects)
        
        # 步骤4: 主体分段描述（使用记忆机制）
        subject_captions = self.step4_subject_captioning(
            image, subject_regions, patch_grid, self.max_rounds, 
            subject_memories=subject_memories, subjects_order=subjects
        )
        
        # 步骤5: 最终整合（纯文本，不再传图）
        final_caption = self.step5_final_integration(subject_captions)
        
        result = {
            'subject_regions': subject_regions,
            'subject_captions': subject_captions,
            'final_caption': final_caption
        }
        
        logger.info("=" * 50)
        logger.info("多主体描述生成完成")
        logger.info("=" * 50)
        
        return result
