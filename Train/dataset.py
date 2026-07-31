# Dataset loader
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class ImageCaptionDataset(Dataset):

    def __init__(
        self,
        image_dir,
        caption_dir,
        image_size=512
    ):

        self.image_dir = image_dir
        self.caption_dir = caption_dir

        self.image_files = sorted(
            [
                file
                for file in os.listdir(image_dir)
                if file.endswith((".png", ".jpg", ".jpeg"))
            ]
        )

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.5, 0.5, 0.5],
                    [0.5, 0.5, 0.5]
                ),
            ]
        )

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        image_name = self.image_files[index]
        image_path = os.path.join(
            self.image_dir,
            image_name
        )
        
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        caption_name = os.path.splitext(image_name)[0] + ".txt"
        caption_path = os.path.join(
            self.caption_dir,
            caption_name
        )

        with open(caption_path, "r", encoding="utf-8") as file:
            caption = file.read().strip()

        return {
            "image": image,
            "caption": caption,
        }


class HFImageCaptionDataset(Dataset):
    """
    Dataset wrapper for HuggingFace Datasets (e.g., AIGCDuckBoss/fluxlora_cool-posters).
    """

    def __init__(self, hf_dataset, image_size=512, default_caption="a cool poster"):
        self.dataset = hf_dataset
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.default_caption = default_caption

        column_names = getattr(self.dataset, "column_names", [])
        if not column_names and hasattr(self.dataset, "features"):
            column_names = list(self.dataset.features.keys())

        # Auto-detect image and text columns
        self.image_column = next((col for col in ["image", "img"] if col in column_names), None)
        if self.image_column is None:
            for col in column_names:
                if "image" in col.lower() or "img" in col.lower():
                    self.image_column = col
                    break
        if self.image_column is None and len(column_names) > 0:
            self.image_column = column_names[0]

        self.text_column = next(
            (col for col in ["text", "caption", "prompt", "description"] if col in column_names),
            None,
        )

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        item = self.dataset[index]
        image_data = item.get(self.image_column) if self.image_column else None

        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        elif hasattr(image_data, "convert"):
            image = image_data.convert("RGB")
        elif isinstance(image_data, Image.Image):
            image = image_data.convert("RGB")
        else:
            raise ValueError(f"Unable to process image column '{self.image_column}' at index {index}")

        image = self.transform(image)

        caption = ""
        if self.text_column and item.get(self.text_column):
            caption = str(item.get(self.text_column)).strip()

        if not caption:
            caption = self.default_caption

        return {
            "image": image,
            "caption": caption,
        }


class HFModelRepoDataset(Dataset):
    """
    Dataset wrapper for HuggingFace Model Repositories containing images (e.g. AIGCDuckBoss/fluxlora_cool-posters).
    """

    def __init__(self, image_dir, image_size=512, default_caption="a cool poster"):
        self.image_files = sorted(
            [
                os.path.join(image_dir, f)
                for f in os.listdir(image_dir)
                if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))
            ]
        )
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        self.default_caption = default_caption

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, index):
        img_path = self.image_files[index]
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        return {
            "image": image,
            "caption": self.default_caption,
        }


def load_hf_dataset(repo_id, image_size=512, default_caption="a cool poster"):
    """
    Loads dataset from HuggingFace, supporting both HF Dataset repos and HF Model repos.
    """
    try:
        from datasets import load_dataset
        raw_dataset = load_dataset(repo_id, split="train")
        return HFImageCaptionDataset(raw_dataset, image_size=image_size, default_caption=default_caption)
    except Exception as e:
        print(f"Loading '{repo_id}' via HuggingFace Hub snapshot...")
        from huggingface_hub import snapshot_download
        repo_dir = snapshot_download(repo_id=repo_id)
        image_dir = os.path.join(repo_dir, "images")
        if not os.path.exists(image_dir) or len(os.listdir(image_dir)) == 0:
            image_dir = repo_dir

        return HFModelRepoDataset(image_dir, image_size=image_size, default_caption=default_caption)
