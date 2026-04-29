"""
从subject_ranker的输出生成多主体描述
衔接之前的outputs，读取主体信息并生成详细描述
"""

import os
import sys
import json
import argparse
import logging
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from tqdm import tqdm
import torch
from PIL import Image

# LLaVA 使用 对比代码/spin（llava_llama 注册，兼容 transformers 4.37）；模型地址由 config/命令行 配置
# 工程根 ofgd/：内含 对比代码/spin 与 multi_subject_caption_wrapper.py（与 llava/ 同级）。
# 脚本可在 ofgd/run_multi_subject_caption.py 或 ofgd/llava/run_multi_subject_caption.py。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if os.path.basename(os.path.normpath(_SCRIPT_DIR)) == "llava":
    _REPO_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
else:
    _REPO_ROOT = _SCRIPT_DIR
_spin_root = os.path.join(_REPO_ROOT, "对比代码", "spin")
if not os.path.isdir(_spin_root):
    raise RuntimeError(
        f"未找到 LLaVA 路径: {_spin_root}，请将 对比代码/spin 置于工程根目录（与 llava 同级）下。"
    )
sys.path.insert(0, _spin_root)
sys.path.insert(0, _REPO_ROOT)  # 从工程根加载 multi_subject_caption_wrapper.py

from multi_subject_caption_wrapper import MultiSubjectCaptionWrapper

try:
    from multi_subject_caption_wrapper import MAX_EXPAND_SUBJECTS as _EXPAND_SUBJECT_CAP
except ImportError:
    _EXPAND_SUBJECT_CAP = 5


def apply_vision_tower_local_path(local_path):
    """让 LLaVA 的 CLIP 视觉塔从本地路径加载，不再访问 HuggingFace。"""
    if not local_path or not os.path.isdir(local_path):
        return
    from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
    _hf_ids = ("openai/clip-vit-large-patch14-336", "openai/clip-vit-large-patch14")

    def _replace(name_or_path):
        return local_path if name_or_path in _hf_ids else name_or_path

    _ov = CLIPVisionModel.from_pretrained
    _op = CLIPImageProcessor.from_pretrained
    _oc = CLIPVisionConfig.from_pretrained

    def _pv(p, *a, **k):
        return _ov(_replace(p), *a, **k)

    def _pp(p, *a, **k):
        return _op(_replace(p), *a, **k)

    def _pc(p, *a, **k):
        return _oc(_replace(p), *a, **k)

    CLIPVisionModel.from_pretrained = _pv
    CLIPImageProcessor.from_pretrained = _pp
    CLIPVisionConfig.from_pretrained = _pc


def setup_logging(output_dir: str):
    """设置日志"""
    log_file = os.path.join(output_dir, "multi_subject_caption.log")
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)


def load_results_from_jsonl(results_file: str) -> List[Dict]:
    """从results.jsonl加载结果"""
    results = []
    if not os.path.exists(results_file):
        logger.warning(f"结果文件不存在: {results_file}")
        return results
    
    with open(results_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    result = json.loads(line)
                    results.append(result)
                except json.JSONDecodeError as e:
                    logger.warning(f"解析JSON失败: {e}")
    
    return results


def load_ranking_summary(ranking_file: str) -> Dict[str, List[str]]:
    """从ranking_summary.json加载排名信息，返回 {image_id: [subject1, subject2, ...]}"""
    ranking_dict = {}
    
    if not os.path.exists(ranking_file):
        logger.warning(f"排名文件不存在: {ranking_file}")
        return ranking_dict
    
    with open(ranking_file, 'r', encoding='utf-8') as f:
        summaries = json.load(f)
    
    for summary in summaries:
        image_id = summary.get('image_id', '')
        ranking = summary.get('ranking', [])
        # 提取所有主体（去重，但保留顺序）
        subjects = []
        seen = set()
        for item in ranking:
            obj_name = item.get('object', '')
            if obj_name and obj_name not in seen:
                subjects.append(obj_name)
                seen.add(obj_name)
        ranking_dict[image_id] = subjects
    
    return ranking_dict


def get_subjects_from_result(result: Dict) -> Tuple[List[str], Dict[str, List[float]]]:
    """
    从result中提取主体列表和bbox信息（按排名顺序）
    支持多个同名主体实例（如多个person、多个motorcycle）
    
    返回:
        subjects: 主体列表（每个实例都有唯一标识，如 "person_0", "person_1"）
        subject_bboxes: {subject_id: [x1, y1, x2, y2]} bbox信息
    """
    candidates = result.get('candidates', [])
    ranking = result.get('ranking', [])
    
    # 按排名顺序提取主体和bbox（保留所有实例，即使名字相同）
    subjects = []
    subject_bboxes = {}
    
    # 统计每个类别的出现次数，用于生成唯一标识
    cls_name_count = {}
    
    for rank_id in ranking:
        if rank_id < len(candidates):
            cand = candidates[rank_id]
            cls_name = cand.get('cls_name', '')
            bbox = cand.get('bbox', [])
            cand_id = cand.get('id', rank_id)  # 使用candidate的id作为唯一标识的一部分
            
            if cls_name:
                # 为每个实例生成唯一标识
                if cls_name not in cls_name_count:
                    cls_name_count[cls_name] = 0
                else:
                    cls_name_count[cls_name] += 1
                
                # 如果有多个同名实例，添加序号；如果只有一个，保持原名
                if cls_name_count[cls_name] == 0:
                    subject_id = cls_name  # 第一个实例保持原名
                else:
                    subject_id = f"{cls_name}_{cls_name_count[cls_name]}"  # 后续实例添加序号
                
                subjects.append(subject_id)
                
                # 为每个实例保存bbox
                if bbox and len(bbox) == 4:
                    subject_bboxes[subject_id] = bbox
                else:
                    logger.warning(f"主体 {subject_id} (cand_id={cand_id}) 没有有效的bbox")
    
    # 检查哪些主体没有bbox
    missing_bbox = [s for s in subjects if s not in subject_bboxes]
    if missing_bbox:
        logger.warning(f"以下主体没有bbox信息，将使用attention方法: {missing_bbox}")
    
    logger.info(f"提取到 {len(subjects)} 个主体实例: {subjects}")
    if subject_bboxes:
        logger.info(f"  其中 {len(subject_bboxes)} 个有bbox信息")
    
    return subjects, subject_bboxes


def save_multi_subject_result(
    image_path: str,
    result: Dict,
    output_file: str,
    original_ranking: List = None,
    original_primary_id: int = None
):
    """保存多主体描述结果"""
    # 将result中的set转换为list，以便JSON序列化
    serializable_result = {}
    for key, value in result.items():
        if key == 'subject_regions':
            # subject_regions是 {subject: set of patch indices}
            serializable_result[key] = {
                subject: sorted(list(patch_set))  # 转换为排序的list
                for subject, patch_set in value.items()
            }
        else:
            serializable_result[key] = value
    
    output = {
        "image": image_path,
        "original_result": {
            "ranking": original_ranking if original_ranking is not None else [],
            "primary_id": original_primary_id,
        },
        "multi_subject_caption": serializable_result
    }
    
    with open(output_file, 'a', encoding='utf-8') as f:
        f.write(json.dumps(output, ensure_ascii=False) + '\n')


def load_config(config_file: str) -> dict:
    """从JSON配置文件加载参数"""
    if not os.path.exists(config_file):
        return {}
    
    with open(config_file, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # 将resume从bool转换为action='store_true'兼容的格式
    if 'resume' in config and isinstance(config['resume'], bool):
        # 保持bool值，稍后在解析时处理
        pass
    
    return config


def main():
    parser = argparse.ArgumentParser(description='从subject_ranker输出生成多主体描述')
    parser.add_argument('--config', type=str, default='config_multi_subject.json',
                       help='配置文件路径（JSON格式）')
    parser.add_argument('--results_file', type=str, default=None,
                       help='subject_ranker的results.jsonl文件路径')
    parser.add_argument('--ranking_file', type=str, default=None,
                       help='subject_ranker的ranking_summary.json文件路径（可选，如果提供会优先使用）')
    parser.add_argument('--images_dir', type=str, default=None,
                       help='图像目录（用于查找原始图像）')
    parser.add_argument('--model_path', type=str, default=None,
                       help='LLaVA 模型路径（如 llava-v1.5-13b 目录）')
    parser.add_argument('--model_base', type=str, default=None,
                       help='LLaVA 的 Vicuna 基座路径，若 MODEL_PATH 为合并权重则留空')
    parser.add_argument('--vision_tower_path', type=str, default=None,
                       help='本地 CLIP 视觉塔路径，可避免从 HuggingFace 拉取')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='输出目录')
    parser.add_argument('--output_file', type=str, default=None,
                       help='输出文件名')
    parser.add_argument('--resume', action='store_true', default=None,
                       help='断点续跑（跳过已处理的图像）')
    parser.add_argument('--seed_k', type=int, default=None,
                       help='初始seed的patch数量')
    parser.add_argument('--max_rounds', type=int, default=None,
                       help='兼容参数（传给 wrapper，用于 step4 等）')
    parser.add_argument('--expansion_threshold', type=float, default=None,
                       help='扩展增益阈值')
    parser.add_argument('--max_region_size_ratio', type=float, default=None,
                       help='区域最大尺寸（相对于seed的倍数）')
    parser.add_argument('--max_subjects', type=int, default=None,
                       help='扩展的主体数量（默认 3）')

    # 先解析配置文件参数
    args, remaining = parser.parse_known_args()
    
    # 加载配置文件
    config = {}
    if args.config and os.path.exists(args.config):
        config = load_config(args.config)
        print(f"从配置文件加载参数: {args.config}")
    
    # 设置默认值（优先级：命令行参数 > 配置文件 > 硬编码默认值）
    defaults = {
        'results_file': 'outputs/results.jsonl',
        'ranking_file': 'outputs/ranking_summary.json',
        'images_dir': 'images',
        'output_dir': 'outputs',
        'output_file': 'multi_subject_captions.jsonl',
        'model_path': None,
        'model_base': None,
        'vision_tower_path': None,
        'resume': False,
        'seed_k': 5,
        'max_rounds': 2,  # 兼容；wrapper 不做区域扩展
        'expansion_threshold': 0.01,
        'max_region_size_ratio': 5.0,
        'max_subjects': None,
    }
    
    # 合并参数（命令行 > 配置文件 > 默认值）
    final_args = argparse.Namespace()
    
    # 先处理普通参数（排除需要特殊处理的参数）
    special_keys = {'model_path', 'model_base', 'vision_tower_path', 'resume'}
    for key, default_value in defaults.items():
        if key in special_keys:
            continue  # 跳过需要特殊处理的参数
        
        # 命令行参数（如果提供了且不是None）
        cmd_value = getattr(args, key, None)
        if cmd_value is not None:
            final_args.__setattr__(key, cmd_value)
        # 配置文件中的值
        elif key in config:
            final_args.__setattr__(key, config[key])
        # 默认值
        else:
            final_args.__setattr__(key, default_value)
    
    # 特殊处理：LLaVA 模型路径（本方法单独配置）
    for key in ('model_path', 'model_base', 'vision_tower_path'):
        final_args.__setattr__(key, getattr(args, key, None) or config.get(key) or defaults.get(key))

    # 特殊处理：resume（action='store_true'的特殊处理）
    # action='store_true'时，如果提供了--resume，args.resume是True；否则是None
    if args.resume is True:
        # 命令行明确指定了--resume
        final_args.resume = True
    elif 'resume' in config:
        # 从配置文件读取
        final_args.resume = config['resume']
    else:
        # 使用默认值
        final_args.resume = defaults['resume']
    
    # 保留config参数，用于日志记录
    final_args.config = args.config
    
    args = final_args
    
    # 创建输出目录（在设置日志之前）
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 设置日志
    global logger
    logger = setup_logging(args.output_dir)

    try:
        import numpy as np
        if int(np.__version__.split(".")[0]) >= 2:
            logger.warning(
                "检测到 NumPy>=2（%s）。LLaVA+spin 依赖链与 NumPy 2 常不兼容，"
                "强烈建议在 ofgd-llava 环境执行: pip install \"numpy<2.0\" 后重启脚本。",
                np.__version__,
            )
    except Exception:
        pass
    
    # 记录使用的配置
    logger.info(f"使用配置文件: {args.config if args.config and os.path.exists(args.config) else '无'}")
    
    logger.info("=" * 50)
    logger.info(
        "多主体描述生成器启动（LLaVA）：ranking 中实例按顺序最多保留 %d 个主体；"
        "固定 patch 掩码 + 逐圈邻接扩展至环为空；不生成背景"
        % _EXPAND_SUBJECT_CAP
    )
    logger.info("=" * 50)
    
    # 加载 LLaVA 模型（本方法单独配置：config 或 --model_path / --model_base / --vision_tower_path）
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not args.model_path:
        raise RuntimeError("必须提供 model_path（命令行 --model_path 或配置文件 model_path）")
    from llava.model.builder import load_pretrained_model
    from llava.mm_utils import get_model_name_from_path
    from llava.utils import disable_torch_init
    if args.vision_tower_path:
        vp = args.vision_tower_path
        if not os.path.isabs(vp):
            vp = os.path.join(_REPO_ROOT, vp)
        apply_vision_tower_local_path(vp)
    disable_torch_init()
    model_path = os.path.expanduser(args.model_path)
    model_base = os.path.expanduser(args.model_base) if args.model_base else None
    model_name = get_model_name_from_path(model_path)
    logger.info(f"加载 LLaVA: {model_path}")
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path, model_base, model_name, device=device
    )
    model.eval()
    # 消除 transformers 警告：模型配置里 do_sample=False 却带了 temperature/top_p，在加载后统一改成非采样默认值
    if getattr(model, "generation_config", None) is not None:
        model.generation_config.temperature = 1.0
        model.generation_config.top_p = 1.0
    logger.info(f"模型已加载到设备: {device}")
    
    # 创建 wrapper：最多 _EXPAND_SUBJECT_CAP 个主体；逐圈 patch 扩展至邻接环为空；不生成背景
    wrapper = MultiSubjectCaptionWrapper(
        model=model,
        tokenizer=tokenizer,
        image_processor=image_processor,
        device=device,
        seed_k=args.seed_k,
        max_rounds=args.max_rounds,
        expansion_threshold=args.expansion_threshold,
        max_region_size_ratio=args.max_region_size_ratio,
        use_spatial_neighbor=True,
        generate_per_round=2,
        max_subjects=None,
    )
    
    # 加载之前的输出
    logger.info(f"加载结果文件: {args.results_file}")
    results = load_results_from_jsonl(args.results_file)
    logger.info(f"找到 {len(results)} 条结果")
    
    # 如果提供了ranking_file，优先使用它（更简洁）
    ranking_dict = None
    if args.ranking_file and os.path.exists(args.ranking_file):
        logger.info(f"加载排名文件: {args.ranking_file}")
        ranking_dict = load_ranking_summary(args.ranking_file)
        logger.info(f"找到 {len(ranking_dict)} 个图像的排名信息")
    
    # 加载已处理的结果（用于断点续跑）
    output_file = os.path.join(args.output_dir, args.output_file)
    processed_images = set()
    if args.resume and os.path.exists(output_file):
        logger.info("检查已处理的结果...")
        with open(output_file, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        result = json.loads(line)
                        img_path = result.get('image', '')
                        if img_path:
                            processed_images.add(img_path)
                    except:
                        pass
        logger.info(f"已处理 {len(processed_images)} 张图像")
    
    # 处理每张图像
    logger.info("开始处理图像...")
    success_count = 0
    skip_count = 0
    error_count = 0
    final_captions_list = []  # 每张图的最终输出，用于单独保存一份 JSON 方便查看
    expand_time_stats = {k: {"count": 0, "total_seconds": 0.0} for k in range(1, _EXPAND_SUBJECT_CAP + 1)}
    
    for result in tqdm(results, desc="处理图像"):
        image_path = result.get('image', '')
        if not image_path:
            logger.warning("跳过：缺少图像路径")
            continue
        
        # 检查是否已处理
        if image_path in processed_images:
            logger.debug(f"跳过已处理: {image_path}")
            skip_count += 1
            continue
        
        # 查找图像文件
        # 如果image_path是绝对路径，直接使用；否则在images_dir中查找
        if os.path.isabs(image_path):
            img_file = image_path
        else:
            # 尝试从images_dir查找
            image_name = os.path.basename(image_path)
            img_file = os.path.join(args.images_dir, image_name)
        
        if not os.path.exists(img_file):
            logger.warning(f"图像文件不存在: {img_file}")
            error_count += 1
            continue
        
        try:
            # 加载图像
            image = Image.open(img_file).convert('RGB')
            
            # 获取主体列表和bbox信息
            image_id = os.path.splitext(os.path.basename(image_path))[0]
            
            # 从result中提取主体和bbox（results.jsonl包含完整信息）
            subjects, subject_bboxes = get_subjects_from_result(result)
            
            # 如果从result中没找到，尝试从ranking_summary获取主体列表（但没有bbox）
            if not subjects and ranking_dict is not None:
                subjects = ranking_dict.get(image_id, [])
                subject_bboxes = {}  # ranking_summary没有bbox信息
                logger.debug(f"从ranking_summary获取主体列表: {subjects}")
            
            if not subjects:
                logger.warning(f"未找到主体: {image_path}")
                error_count += 1
                continue
            
            logger.info(f"处理: {image_path} (主体: {subjects})")
            if subject_bboxes:
                logger.info(f"  提供bbox信息的主体: {list(subject_bboxes.keys())}")
            
            # 生成多主体描述（传入bbox信息）
            expand_subject_count = max(1, min(len(subjects), _EXPAND_SUBJECT_CAP))
            t0 = time.perf_counter()
            multi_result = wrapper.generate(image, subjects, subject_bboxes)
            elapsed = time.perf_counter() - t0
            expand_time_stats[expand_subject_count]["count"] += 1
            expand_time_stats[expand_subject_count]["total_seconds"] += elapsed
            logger.info(f"  参与主体数={expand_subject_count}，本图耗时={elapsed:.2f}s")
            
            # 从原始result中获取ranking和primary_id（results.jsonl中有完整信息）
            original_ranking = result.get('ranking', [])
            original_primary_id = result.get('primary_id', None)
            
            # 保存结果
            save_multi_subject_result(image_path, multi_result, output_file, original_ranking, original_primary_id)
            
            # 累积每张图的最终输出（仅 image + final_caption）
            final_captions_list.append({
                "image": image_path,
                "final_caption": multi_result.get("final_caption", ""),
            })
            
            success_count += 1
            logger.info(f"完成: {image_path}")
            
        except Exception as e:
            logger.error(f"处理失败 {image_path}: {str(e)}", exc_info=True)
            error_count += 1
    
    # 保存「每张图最终输出」的单独 JSON，方便查看
    final_captions_file = os.path.join(args.output_dir, "multi_subject_final_captions.json")
    with open(final_captions_file, "w", encoding="utf-8") as f:
        json.dump(final_captions_list, f, ensure_ascii=False, indent=2)
    logger.info(f"每张图最终描述已保存: {final_captions_file}")

    time_stats_payload = {}
    logger.info("按主体数耗时统计（仅成功样本）:")
    for n in range(1, _EXPAND_SUBJECT_CAP + 1):
        c = expand_time_stats[n]["count"]
        total_s = expand_time_stats[n]["total_seconds"]
        avg_s = (total_s / c) if c else 0.0
        time_stats_payload[str(n)] = {
            "count": c,
            "total_seconds": round(total_s, 4),
            "avg_seconds": round(avg_s, 4),
        }
        logger.info(
            f"  {n}个主体: 数量={c}, 总耗时={total_s:.2f}s, 平均耗时={avg_s:.2f}s"
        )
    time_stats_file = os.path.join(args.output_dir, "expansion_time_stats.json")
    with open(time_stats_file, "w", encoding="utf-8") as f:
        json.dump(time_stats_payload, f, ensure_ascii=False, indent=2)
    logger.info(f"耗时统计已保存: {time_stats_file}")
    
    # 总结
    logger.info("=" * 50)
    logger.info("处理完成")
    logger.info(f"成功: {success_count}")
    logger.info(f"跳过: {skip_count}")
    logger.info(f"失败: {error_count}")
    logger.info(f"结果保存到: {output_file}")
    logger.info("=" * 50)


if __name__ == "__main__":
    main()
