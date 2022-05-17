import numpy as np
from torch import nn, Tensor
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms
import seaborn as sns

class poseimage(object):
    def __init__(self, img:Tensor, bbox:Tensor=None, joints:Tensor=None, heatmap:bool=True):
        self.img = img
        self.bbox = bbox
        self.joints = joints
        self.heatmap_transform = True
        if self.heatmap_transform:
            self.tensor_to_img_T = transforms.ToPILImage()

    def show_image_only(self):
        rgb_img = self.tensor_to_img_T(self.img)
        plt.imshow(rgb_img)
        plt.show()
        return
    
    def show_joints_heatmap(self):

        return