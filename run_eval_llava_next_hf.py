import os
import re
import json
import argparse
from collections import defaultdict

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration


# =========================================================
# KidVis task/capability mapping
# Assumption:
#   Question_1 ... Question_10 follow the same order as Task 1 ... Task 10
# in the paper.
# If your subset numbering differs, only edit TASK_CONFIG below.
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
    # Keep the original benchmark prompt fields to stay close to the Qwen script.
    if lang == "zh":
        return item["question_zh"]
    if lang == "en":
        return item["question_en"]
    raise ValueError("lang must be 'zh' or 'en'")


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


def run_single_inference(model, processor, image, prompt, max_new_tokens=1024):
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image"},
            ],
        }
    ]

    prompt_text = processor.apply_chat_template(
        conversation,
        add_generation_prompt=True,
    )

    inputs = processor(
        images=image,
        text=prompt_text,
        return_tensors="pt",
    )
    inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

    with torch.inference_mode():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    input_length = inputs["input_ids"].shape[1]
    generated_ids_trimmed = generated_ids[:, input_length:]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )[0]

    return output_text


def get_model_kwargs(use_flash_attn: bool):
    model_kwargs = {
        "torch_dtype": torch.float16,
        "device_map": "auto",
    }

    if use_flash_attn:
        model_kwargs["attn_implementation"] = "flash_attention_2"

    return model_kwargs


def sanitize_model_name(model_name: str) -> str:
    return model_name.replace("/", "_").replace("-", "_").lower()


def compute_capability_scores(subset_results: dict):
    """
    Capability Score is computed from the tasks probing the same capability.
    In the paper, this is a weighted average over related tasks.
    Since each KidVis task has 50 questions, using the per-task totals here
    is equivalent to the intended weighted average.
    """
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
    model_kwargs = get_model_kwargs(args.use_flash_attn)

    model = LlavaNextForConditionalGeneration.from_pretrained(
        args.model_name,
        **model_kwargs,
    )
    processor = LlavaNextProcessor.from_pretrained(args.model_name)

    total = 0
    correct = 0
    subset_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    predictions = []

    for item in tqdm(dataset, desc="Evaluating"):
        prompt = build_prompt(item, lang=args.lang)

        raw_output = run_single_inference(
            model=model,
            processor=processor,
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
    parser.add_argument("--model_name", type=str, default="llava-hf/llama3-llava-next-8b-hf")
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
