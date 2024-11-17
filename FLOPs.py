# import torch
#
# from fvcore.nn import FlopCountAnalysis, flop_count_table
# from flopco import FlopCo
# from thop import profile
# 
# input_img = torch.ones(1, 3, 224, 224).cuda()
#
# flops = FlopCountAnalysis(model, input_img)
#
# print(flops.total()) # kb단위로 모델전체 FLOPs 출력해줌
# print(flop_count_table(flops)) # 테이블 형태로 각 연산하는 모듈마다 출력해주고, 전체도 출력해줌
#
# stats = FlopCo(model, img_size = (1, 3, 224, 224), device = device)
# print(f"MAC: {stats.total_macs / 10 ** 9:.03} G")
#
# input = torch.randn(1, 3, 224, 224).cuda()
# macs, params = profile(model, inputs=(input, ))
# print(f'macs: {macs/ 10 ** 9:.03}, params: {params}')
# # print(stats.total_macs, stats.relative_flops)