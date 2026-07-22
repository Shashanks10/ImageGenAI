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