import os
import re
import json
import argparse
from collections import defaultdict

import torch
import numpy as np
import torchvision.transforms as T
from PIL import Image
from tqdm import tqdm
from datasets import load_dataset
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer

# =========================================================
# KidVis task/capability mapping
# =========================================================
TASK_CONFIG = {
    "Question_1": {
        "task_name": "Body Part Counting",
        "capabilities": ["VD", "VM"],
    },
    "Question_2": {
        "task_name": "Clock Reading",
        "capabilities": ["VS", "VD"],
    },
    "Question_3": {
        "task_name": "Complex Scene Counting",
        "capabilities": ["VC", "VD"],
    },
    "Question_4": {
        "task_name": "Hidden Figures",
        "capabilities": ["VT", "VCl"],
    },
    "Question_5": {
        "task_name": "Schulte Grid",
        "capabilities": ["VC", "VT"],
    },
    "Question_6": {
        "task_name": "Spatial Orientation",
        "capabilities": ["VS", "VD"],
    },
    "Question_7": {
        "task_name": "Visual Completion",
        "capabilities": ["VM", "VCl"],
    },
    "Question_8": {
        "task_name": "Visual Reasoning",
        "capabilities": ["VD", "VM"],
    },
    "Question_9": {
        "task_name": "Jigsaw Assembly",
        "capabilities": ["VCl", "VS"],
    },
    "Question_10": {
        "task_name": "Path Tracing",
        "capabilities": ["VT", "VC"],
    },
}

# =========================================================
# InternVL3.5 Image Processing Utilities
# =========================================================
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        split_img = resized_img.crop(box)
        processed_images.append(split_img)
    assert len(processed_images) == blocks
    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images

def process_image(image: Image.Image, input_size=448, max_num=12):
    """Modified load_image to take a PIL Image directly from datasets"""
    if image.mode != 'RGB':
        image = image.convert('RGB')
    transform = build_transform(input_size=input_size)
    images = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(img) for img in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values

# =========================================================
# Evaluation Functions
# =========================================================
def check_local_dataset(data_dir: str):
    if not os.path.isdir(data_dir):
        raise FileNotFoundError(f"Dataset directory not found: {data_dir}")

    metadata_path = os.path.join(data_dir, "metadata.csv")
    images_dir = os.path.join(data_dir, "images")

    if not os.path.isfile(metadata_path):
        raise FileNotFoundError(f"metadata.csv not found: {metadata_path}")

    if not os.path.isdir(images_dir):
        raise FileNotFoundError(f"images directory not found: {images_dir}")


def load_kidvis(data_dir: str, split: str = "train"):
    check_local_dataset(data_dir)
    return load_dataset("imagefolder", data_dir=data_dir, split=split)


def build_prompt(item, lang: str = "en") -> str:
    # InternVL uses <image>\n for single image prompting
    if lang == "zh":
        question = item["question_zh"]
    elif lang == "en":
        question = item["question_en"]
    else:
        raise ValueError("lang must be 'zh' or 'en'")
    
    return f"<image>\n{question}"


def extract_choice(text: str):
    """
    Extract a single option letter from model output.
    Priority: standalone A/B/C/D.
    """
    if text is None:
        return None

    text = text.strip().upper()

    matches = re.findall(r"\b([A-D])\b", text)
    if matches:
        return matches[0]

    if len(text) == 1 and text in {"A", "B", "C", "D"}:
        return text

    return None


def run_single_inference(model, tokenizer, image, prompt, max_new_tokens=1024):
    pixel_values = process_image(image, max_num=12).to(model.dtype).to(model.device)
    
    generation_config = dict(max_new_tokens=max_new_tokens, do_sample=False)
    
    # Use model.chat as in InternVL documentation
    response = model.chat(tokenizer, pixel_values, prompt, generation_config)
    
    return response


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").lower()


def compute_capability_scores(subset_results: dict):
    capability_buckets = defaultdict(lambda: {"correct": 0, "total": 0, "tasks": []})

    for subset, result in subset_results.items():
        if subset not in TASK_CONFIG:
            continue

        task_name = TASK_CONFIG[subset]["task_name"]
        capabilities = TASK_CONFIG[subset]["capabilities"]

        for cap in capabilities:
            capability_buckets[cap]["correct"] += result["correct"]
            capability_buckets[cap]["total"] += result["total"]
            capability_buckets[cap]["tasks"].append(task_name)

    capability_scores = {}
    for cap, stats in capability_buckets.items():
        acc = stats["correct"] / stats["total"] if stats["total"] > 0 else 0.0
        capability_scores[cap] = {
            "accuracy": acc,
            "score_100": acc * 100.0,
            "correct": stats["correct"],
            "total": stats["total"],
            "tasks": sorted(set(stats["tasks"])),
        }

    return capability_scores


def evaluate(args):
    print(f"Loading local dataset from: {args.data_dir} ({args.split})")
    dataset = load_kidvis(data_dir=args.data_dir, split=args.split)

    if args.num_samples is not None:
        dataset = dataset.select(range(min(args.num_samples, len(dataset))))
        print(f"Using first {len(dataset)} samples for evaluation")
    else:
        print(f"Loaded {len(dataset)} samples")

    print(f"Loading model: {args.model_name}")
    

    model = AutoModel.from_pretrained(
        args.model_name,
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=False,
        use_flash_attn=args.use_flash_attn,
        trust_remote_code=True,
    ).eval().cuda()
    
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name, 
        trust_remote_code=True, 
        use_fast=False
    )

    total = 0
    correct = 0
    subset_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []

    for item in tqdm(dataset, desc="Evaluating"):
        prompt = build_prompt(item, lang=args.lang)

        raw_output = run_single_inference(
            model=model,
            tokenizer=tokenizer,
            image=item["image"],
            prompt=prompt,
            max_new_tokens=args.max_new_tokens,
        )

        pred = extract_choice(raw_output)
        gold = str(item["answer"]).strip().upper()
        is_correct = int(pred == gold)

        subset = item["subset"]

        total += 1
        correct += is_correct
        subset_stats[subset]["correct"] += is_correct
        subset_stats[subset]["total"] += 1

        task_name = TASK_CONFIG.get(subset, {}).get("task_name", subset)
        capabilities = TASK_CONFIG.get(subset, {}).get("capabilities", [])

        predictions.append(
            {
                "subset": subset,
                "task_name": task_name,
                "capabilities": capabilities,
                "question_id": item["question_id"],
                "question_zh": item["question_zh"],
                "question_en": item["question_en"],
                "gold": gold,
                "pred": pred,
                "correct": is_correct,
                "raw_output": raw_output,
            }
        )

    overall_acc = correct / total if total > 0 else 0.0

    subset_results = {}
    for subset in sorted(subset_stats.keys()):
        s_correct = subset_stats[subset]["correct"]
        s_total = subset_stats[subset]["total"]
        s_acc = s_correct / s_total if s_total > 0 else 0.0

        subset_results[subset] = {
            "task_name": TASK_CONFIG.get(subset, {}).get("task_name", subset),
            "capabilities": TASK_CONFIG.get(subset, {}).get("capabilities", []),
            "accuracy": s_acc,
            "score_100": s_acc * 100.0,
            "correct": s_correct,
            "total": s_total,
        }

    capability_results = compute_capability_scores(subset_results)

    summary = {
        "model_name": args.model_name,
        "split": args.split,
        "language": args.lang,
        "data_dir": args.data_dir,
        "num_samples": total,
        "overall_accuracy": overall_acc,
        "overall_score_100": overall_acc * 100.0,
        "subset_results": subset_results,
        "capability_results": capability_results,
    }

    print("\n===== Overall Result =====")
    print(f"Accuracy: {overall_acc:.4f} ({correct}/{total})")
    print(f"Score(0-100): {overall_acc * 100.0:.2f}")

    print("\n===== Subset Results =====")
    for subset, result in subset_results.items():
        print(
            f"{subset} | {result['task_name']}: "
            f"{result['accuracy']:.4f} "
            f"({result['correct']}/{result['total']}) | "
            f"Score={result['score_100']:.2f}"
        )

    print("\n===== Capability Results =====")
    for cap in ["VC", "VT", "VD", "VM", "VS", "VCl"]:
        if cap not in capability_results:
            continue
        result = capability_results[cap]
        print(
            f"{cap}: {result['accuracy']:.4f} "
            f"({result['correct']}/{result['total']}) | "
            f"Score={result['score_100']:.2f}"
        )

    os.makedirs(args.output_dir, exist_ok=True)
    model_tag = sanitize_model_name(args.model_name)

    pred_path = os.path.join(
        args.output_dir,
        f"predictions_{model_tag}_{args.lang}.jsonl",
    )
    with open(pred_path, "w", encoding="utf-8") as f:
        for row in predictions:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary_path = os.path.join(
        args.output_dir,
        f"summary_{model_tag}_{args.lang}.json",
    )
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("\nSaved files:")
    print(f"- {pred_path}")
    print(f"- {summary_path}")


def parse_args():
    parser = argparse.ArgumentParser()
    # Updated default model to InternVL3.5
    parser.add_argument("--model_name", type=str, default="OpenGVLab/InternVL3_5-8B")
    parser.add_argument("--data_dir", type=str, default="./KidVis")
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--lang", type=str, default="en", choices=["zh", "en"])
    parser.add_argument("--max_new_tokens", type=int, default=16)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num_samples", type=int, default=None)
    parser.add_argument("--use_flash_attn", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    evaluate(args)