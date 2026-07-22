# Dataset loader
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class CustomDataset(Dataset):
    def __init__(self, image_folder, caption_folder, transform=None):
        self.image_folder = image_folder
        self.caption_folder = caption_folder
        self.transform = transform
        
        self.image_files = [f for f in os.listdir(image_folder) if f.endswith(".jpg")]
        self.image_files.sort()
        
        # Optional: create mapping between image and caption files
        self.caption_map = {f.replace(".jpg", ".txt"): f for f in self.image_files}
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Get image
        image_file = self.image_files[idx]
        image_path = os.path.join(self.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        
        # Get caption
        caption_file = self.caption_map[image_file]
        caption_path = os.path.join(self.caption_folder, caption_file)
        with open(caption_path, "r") as f:
            caption = f.read().strip()
        
        # Apply transform
        if self.transform:
            image = self.transform(image)
        
        return {"image": image, "caption": caption}