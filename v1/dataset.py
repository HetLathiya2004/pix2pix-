import numpy as np
import config
import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.utils import save_image
import glob

class MapDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.list_files = glob.glob(f'{self.root_dir}/*.png') 

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = img_file
        image = Image.open(img_path)
        image = image.convert('RGB')
        width, height = image.size
        target_image = image.crop((0, 0, width // 2, height))       
        input_image = image.crop((width // 2, 0, width, height))
        
        target_image = config.both_transform(target_image)
        input_image = config.both_transform(input_image)
        input_image = config.transform_only_input(input_image)
        target_image = config.transform_only_mask(target_image)
        return input_image, target_image


if __name__ == "__main__":
    dataset = MapDataset("data/train/")
    loader = DataLoader(dataset, batch_size=5)
    for x, y in loader:
        print(x.shape)
        save_image(x, "x.png")
        save_image(y, "y.png")
        import sys

        sys.exit()