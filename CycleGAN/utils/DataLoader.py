import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import PIL.Image as Image

import glob
import os
import random
import numpy as np


class Loader(data.DataLoader):
    def __init__(self, dataset_dir, styles, transforms, aug_transform=None):
        super(Loader, self).__init__(self)
        self.dataset_dir = dataset_dir
        self.styles = styles

        folder_A = glob.glob(f'{os.path.join(dataset_dir, styles[0])}/*')
        folder_B = glob.glob(f'{os.path.join(dataset_dir, styles[0])}/*')

        self.transform = transforms

        self.aug_transform = aug_transform

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

        """ 
        데이터에 augmentation class 추가하는 부분 
        0 -> augmentation 안함
        1 -> augmentation 진행함
        """
        original_A = [[path, 0] for path in self.path_A]
        original_B = [[path, 0] for path in self.path_B]

        if self.aug_transform is not None:
            aug_path_A = [[path, 1] for path in self.path_A]
            aug_path_B = [[path, 1] for path in self.path_B]
            self.image_path_A = original_A + aug_path_A
            self.image_path_B = original_B + aug_path_B

        else:
            self.image_path_A = original_A
            self.image_path_B = original_B

    def shuffle_image(self, A, B):
        random.shuffle(B)
        for i in range(len(A)):
            if A[i] == B[i]:
                return self.shuffle_image(A, B)
        return B

    def __getitem__(self, index):

        # augmentation transform 이 비어 있을 경우 augmentation을 적용한 data까지 가져오고
        # 아닌경우 그냥 원본 이미지만 가져온다.

        if self.aug_transform is not None:
            # DATASET A
            if self.image_path_A[index][1] == 0:
                item_A = self.aug_transform(Image.open(self.image_path_A[index][0]))

            else:
                item_A = self.transform(Image.open(self.image_path_A[index][0]))

            # DATASET B
            if self.image_path_B[index][1] == 0:
                item_B = self.aug_transform(Image.open(self.image_path_B[index][0]))

            else:
                item_B = self.transform(Image.open(self.image_path_B[index][0]))

        else:
            item_A = self.transform(Image.open(self.image_path_A[index][0]))
            item_B = self.transform(Image.open(self.image_path_B[index][0]))

        return [item_A, item_B, self.image_path_A[index][0]]

    def __len__(self):
        return len(self.image_path_A)