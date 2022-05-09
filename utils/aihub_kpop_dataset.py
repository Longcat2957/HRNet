import os
import torch
import pandas as pd

from pathlib import Path
from torch.utils.data import Dataset
from torchvision.io import read_image   # Reads a JPEG or PNG image into a 3 dimensional RGB or grayscale Tensor. Optionally converts the image to the desired format. The values of the output tensor are uint8 in [0, 255].
from torchvision.transforms.functional import crop, resize

# Directory
current_path = Path(os.path.dirname(os.path.abspath(__file__))) # ../HRNET/utils 의 절대 경로
parent_path = current_path.parent                               # ../HRNet/ 의 절대경로
data_path = parent_path / 'data/openai'                         # ../HRNet/data/openai의 절대 경로
annotation_dir = data_path / 'annotation'
image_dir = data_path / 'image'
annotation_file_path = annotation_dir / 'annotation.csv'

def kpopimgT(img, bbox:tuple):
    l, t, w, h = bbox
    img = crop(img, top=t, left=l, height=h, width=w)
    img = resize(img, 256)  # resize to 256 x 256
    return img
    

def kpoplabelT(joints, bbox:tuple):
    l, t, w, h = bbox
    empty_list = []
    for i in range(29):
        x, y, v = joints[i], joints[i+1], joints[i+2]
        x = (x - l) / w
        y = (y - t) / h
        empty_list.append((x, y))
    
    return empty_list

class KpopImageDataset(Dataset):
    def __init__(self, annotations_file, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.transform = transform
        self.target_transform = target_transform
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        line = self.img_labels[idx]
        img_path = line[2]
        image = read_image(img_path)
        label = line[5]
        
        # for transform
        bbox = line[3]
        # l, t, w, h = bbox
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
            
        return image, label
    
class KpopImageDatasetwT(Dataset):
    # KpopImageDataset with Transform
    def __init__(self, annotations_file=annotation_file_path):
        self.img_labels = pd.read_csv(annotations_file)
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        line = self.img_labels[idx]
        img_path = line[2]
        image = read_image(img_path)
        label = line[5]
        
        # for transform
        bbox = line[3]
        # l, t, w, h = bbox

        image = kpopimgT(image, bbox)
        label = kpoplabelT(label, bbox)
            
        return image, label