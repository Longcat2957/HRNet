import os
import torch
import pandas as pd
import ast
import numpy as np
import math
import numpy as np

from PIL import Image
from scipy import ndimage
from torch import Tensor, nn
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image   # Reads a JPEG or PNG image into a 3 dimensional RGB or grayscale Tensor. Optionally converts the image to the desired format. The values of the output tensor are uint8 in [0, 255].
from torchvision import transforms
from torchvision.transforms.functional import crop, resize

# Directory
current_path = Path(os.path.dirname(os.path.abspath(__file__))) # ../HRNET/utils 의 절대 경로
parent_path = current_path.parent                               # ../HRNet/ 의 절대경로
data_path = parent_path / 'data/openai'                         # ../HRNet/data/openai의 절대 경로
annotation_dir = data_path / 'annotation'
image_dir = data_path / 'image'
annotation_file_path = annotation_dir / 'annotation.csv'



def kpoplabelT(joints, bbox:list, sigma:int=1):
    l, t, w, h = bbox
    empty_list = np.zeros((29, 64, 64))
    for i in range(29):
        x, y, v = joints[3*i], joints[3*i+1], joints[3*i+2]
        rx, ry = (x-l)/w, (y-t)/h

        zero = np.zeros((64, 64))
        if v > 0:
            for a in range(64):
                for b in range(64):
                    zero[a][b] = (1/(2 * math.pi) * sigma) * math.exp(-((64 * rx - b)**2+(64 * ry - a)**2)/(2 * sigma**2))
        empty_list[i, :] = zero
    # 가우시안 분포를 리턴합니다.
    # visibility = 0, 1, 2
    return torch.Tensor(empty_list)


class KpopImageDatasetwT(Dataset):
    # KpopImageDataset with Transform
    def __init__(self, annotations_file=annotation_file_path):
        self.img_labels = pd.read_csv(annotations_file)
        self.num_joints = 29
        self.label_transform = torch.nn.UpsamplingNearest2d(size=(64,64))
        self.img_T = transforms.Compose([transforms.Resize((256, 256)),
                                         transforms.ToTensor(),
                                         transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        line = self.img_labels.loc[idx]
        img_path = line[2]
        image = Image.open(img_path)        #image -> PIL Image Format
        label = ast.literal_eval(line[5])
        
        # for transform
        bbox = ast.literal_eval(line[3])
        l, t, w, h = bbox                   # bbox 범위만큼 원본 이미지를 크롭
        image = image.crop((l, t, l+w, t+h))
        image = self.img_T(image)
        label = kpoplabelT(label, bbox)

        
        return image, label

# testdataset = KpopImageDatasetwT()
# print(testdataset.__len__())
# print(testdataset.__getitem__(0))