import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image

import glob
import os
import random
import numpy as np



class Loader(data.Dataset):
    def __init__(self, path, transform):
        super(Loader, self).__init__()
        self.path = path

        folder_A = glob.glob(f'{path}/*')
        folder_B = glob.glob(f'{path}/*')

        self.transform = transform

        self.path_A = []
        self.path_B = []

        """
        inner class image 셔플
        """
        for i in range(len(folder_A)):
            A = glob.glob(f'{folder_A[i]}/*.png')
            B = glob.glob(f'{folder_B[i]}/*.png')
            B = self.shuffle_image(A, B)

            self.path_A = self.path_A + A
            self.path_B = self.path_B + B

        self.image_path_A = [[path, 0] for path in self.path_A]
        self.image_path_B = [[path, 0] for path in self.path_B]

    def shuffle_image(self, A, B):
        random.shuffle(B)
        for i in range(len(A)):
            if A[i] == B[i]:
                return self.shuffle_image(A, B)
        return B

    def __getitem__(self, index):

        # augmentation transform 이 비어 있을 경우 augmentation을 적용한 data까지 가져오고
        # 아닌경우 그냥 원본 이미지만 가져온다.
        item_A = self.transform(Image.open(self.image_path_A[index][0]))
        item_B = self.transform(Image.open(self.image_path_B[index][0]))

        return [item_A, item_B, self.image_path_A[index][0]]

    def __len__(self):
        return len(self.image_path_A)
