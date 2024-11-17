import argparse
import os
import torch
import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from PIL import ImageFile
from tensorboardX import SummaryWriter
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path
import models.transformer as transformer
import models.StyTR as StyTR
from sampler import InfiniteSamplerWrapper
from torchvision.utils import save_image
import random
import glob
from fvcore.nn import FlopCountAnalysis, flop_count_table


def train_transform():
    transform_list = [
        transforms.Resize(size=(512, 512)),
        transforms.RandomCrop(224),
        transforms.ToTensor()
    ]
    return transforms.Compose(transform_list)

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
        item_A = self.transform(Image.open(self.image_path_A[index][0]).convert('RGB'))
        item_B = self.transform(Image.open(self.image_path_B[index][0]).convert('RGB'))

        return [item_A, item_B, self.image_path_A[index][0]]

    def __len__(self):
        return len(self.image_path_A)

def adjust_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = 2e-4 / (1.0 + args.lr_decay * (iteration_count - 1e4))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def warmup_learning_rate(optimizer, iteration_count):
    """Imitating the original implementation"""
    lr = args.lr * 0.1 * (1.0 + 3e-4 * iteration_count)
    # print(lr)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


parser = argparse.ArgumentParser()
# Basic options
parser.add_argument('--content_dir', default='./datasets/train2014', type=str,
                    help='Directory path to a batch of content images')
parser.add_argument('--style_dir', default='./datasets/Images', type=str,  #wikiart dataset crawled from https://www.wikiart.org/
                    help='Directory path to a batch of style images')
parser.add_argument('--vgg', type=str, default='./vgg_normalised.pth')  #run the train.py, please download the pretrained vgg checkpoint
parser.add_argument('--custom_path', type=str, default='/home/kimjungsoo/Lab/dataset/ND/original/A')

# training options
parser.add_argument('--save_dir', default='/home/kimjungsoo/Lab/Compare/backup/StyTr2/experiments',
                    help='Directory to save the model')
parser.add_argument('--log_dir', default='/home/kimjungsoo/Lab/Compare/backup/StyTr2/logs',
                    help='Directory to save the log')
parser.add_argument('--lr', type=float, default=5e-4)
parser.add_argument('--lr_decay', type=float, default=1e-5)
parser.add_argument('--max_iter', type=int, default=160000)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--style_weight', type=float, default=10.0)
parser.add_argument('--content_weight', type=float, default=7.0)
parser.add_argument('--n_threads', type=int, default=1)
parser.add_argument('--save_model_interval', type=int, default=10000)
parser.add_argument('--position_embedding', default='sine', type=str, choices=('sine', 'learned'),
                        help="Type of positional embedding to use on top of the image features")
parser.add_argument('--hidden_dim', default=512, type=int,
                        help="Size of the embeddings (dimension of the transformer)")
args = parser.parse_args()

USE_CUDA = torch.cuda.is_available()
device = torch.device("cuda:0" if USE_CUDA else "cpu")

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

if not os.path.exists(args.log_dir):
    os.mkdir(args.log_dir)
writer = SummaryWriter(log_dir=args.log_dir)

vgg = StyTR.vgg
vgg = nn.Sequential(*list(vgg.children())[:44])

decoder = StyTR.decoder
embedding = StyTR.PatchEmbed()

Trans = transformer.Transformer()
with torch.no_grad():
    network = StyTR.StyTrans(vgg,decoder,embedding, Trans,args)

network = network.cuda()
input_img = torch.ones(1, 3, 224, 224).cuda()
flops = FlopCountAnalysis(network, (input_img, input_img))
print(f'flops: {flops.total() / (2**30)}')  # kb단위로 모델전체 FLOPs 출력해줌
print(flop_count_table(flops))  # 테이블 형태로 각 연산하는 모듈마다 출력해주고, 전체도 출력해줌