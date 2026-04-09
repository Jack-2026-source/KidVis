from datasets import load_dataset


def get_kidvis_dataset(split: str = "train"):
    """Load the KidVis dataset from Hugging Face."""
    return load_dataset("Jack-2026/KidVis", split=split)


if __name__ == "__main__":
    ds = get_kidvis_dataset()
    sample = ds[0]

    print(f"Loaded {len(ds)} samples.")
    print(f"Subset: {sample['subset']}")
    print(f"Question ID: {sample['question_id']}")
    print(f"Answer: {sample['answer']}")
    print(f"Chinese prompt: {sample['question_zh']}")
    print(f"English prompt: {sample['question_en']}")
    print(f"Image size: {sample['image'].size}")