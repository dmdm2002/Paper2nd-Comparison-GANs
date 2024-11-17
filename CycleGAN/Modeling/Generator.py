import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from CycleGAN.Modeling.Layers import ResBlock
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, flop_count_table
from thop import profile, clever_format


class Gen(nn.Module):
    def __init__(self, blocks=6):
        super(Gen, self).__init__()

        # init = nn.R
        encoder_1 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7),
            nn.InstanceNorm2d(64)
        ]
        encoder_2 = [
            self.__conv_block(64, 128)
        ]
        encoder_3 = [
            self.__conv_block(128, 256)
        ]

        self.downsample_graph = nn.Conv2d(256, 256, kernel_size=3, stride=10, padding=1)

        res_blocks = [
            ResBlock(256) for _ in range(blocks)
        ]

        self.upsample_graph = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=10, padding=1,
                                                 output_padding=5)
        decoder_1 = [
            self.__conv_block(256, 128, upsample=True),
        ]

        decoder_2 = [
            self.__conv_block(128, 64, upsample=True)
        ]

        decoder_3 = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7),
            nn.Tanh(),
        ]

        self.relu = nn.LeakyReLU(0.2)
        self.encoder_1 = nn.Sequential(*encoder_1)
        self.encoder_2 = nn.Sequential(*encoder_2)
        self.encoder_3 = nn.Sequential(*encoder_3)

        self.res_blocks = nn.Sequential(*res_blocks)

        self.decoder_1 = nn.Sequential(*decoder_1)
        self.decoder_2 = nn.Sequential(*decoder_2)
        self.decoder_3 = nn.Sequential(*decoder_3)

    def __conv_block(self, in_features, out_features, upsample=False):
        if upsample:
            conv = nn.ConvTranspose2d(in_features, out_features, 3, 2, 1, output_padding=1)

        else:
            conv = nn.Conv2d(in_features, out_features, 3, 2, 1)

        return nn.Sequential(
            conv,
            nn.InstanceNorm2d(256),
            nn.ReLU()
        )

    def gen_adj(self, A):
        D = torch.pow(A.sum(1).float(), -0.5)
        D = torch.diag(D)
        adj = torch.matmul(torch.matmul(A, D).t(), D)

        return adj

    def forward(self, input):
        x = self.encoder_1(input)
        x = self.encoder_2(x)
        x = self.encoder_3(x)

        x = self.res_blocks(x)

        x = self.decoder_1(x)
        x = self.decoder_2(x)
        x = self.decoder_3(x)

        return x

if __name__ == '__main__':
    model = Gen().cuda()
    input_img = torch.ones(1, 3, 224, 224).cuda()
    flops = FlopCountAnalysis(model, input_img)
    print(f'flops: {flops.total() / (2**30)}') # kb단위로 모델전체 FLOPs 출력해줌
    print(flop_count_table(flops))  # 테이블 형태로 각 연산하는 모듈마다 출력해주고, 전체도 출력해줌

    input_img = torch.ones(1, 3, 224, 224).cuda()
    macs, params = profile(model, inputs=(input_img, ))

    macs, params = clever_format([macs, params], "%.3f")

    print(macs)
    print(params)
