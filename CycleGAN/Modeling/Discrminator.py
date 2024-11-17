import torch
import torch.nn as nn
import torch.optim as optim
from torchsummary import summary
from fvcore.nn import FlopCountAnalysis, flop_count_table


class disc(nn.Module):
    def __init__(self):
        super(disc, self).__init__()

        self.model = nn.Sequential(
            self.__conv_layer(3, 64, norm=False),
            self.__conv_layer(64, 128),
            self.__conv_layer(128, 256),
            self.__conv_layer(256, 512, stride=1),
            nn.Conv2d(512, 1, 4, 1, 1),
        )

    def __conv_layer(self, in_features, out_features, stride=2, norm=True):
        layer = [nn.Conv2d(in_features, out_features, 4, stride, 1)]

        if norm:
            layer.append(nn.InstanceNorm2d(out_features))

        layer.append(nn.LeakyReLU(0.2))
        layer = nn.Sequential(*layer)

        return layer

    def forward(self, x):
        return self.model(x)

if __name__ == '__main__':
    model = disc().cuda()
    input_img = torch.ones(1, 3, 224, 224).cuda()
    flops = FlopCountAnalysis(model, input_img)
    print(f'flops: {flops.total() / (2**30)}') # kb단위로 모델전체 FLOPs 출력해줌
    print(flop_count_table(flops))  # 테이블 형태로 각 연산하는 모듈마다 출력해주고, 전체도 출력해줌