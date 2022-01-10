import torch.nn as nn
import torch.nn.functional as F
import torchvision


def _round_to_multiple_of(val, divisor, round_up_bias=0.9):
    """ Asymmetric(비대칭) rounding to make `val` divisible by `divisor`. With default
    bias, will round up, unless the number is no more than 10% greater than the
    smaller divisible value, i.e. (83, 8) -> 80, but (84, 8) -> 88. """
    # 여기 무슨 내용인지... 왜하지
    assert 0.0 < round_up_bias < 1.0
    new_val = max(divisor, int(val + divisor / 2) // divisor * divisor)  # // : 나눈 후 정수만
    return new_val if new_val >= round_up_bias * val else new_val + divisor


def _get_depths(alpha): #back bone shape에 사용
    """ Scales tensor depths as in reference MobileNet code, prefers rouding up
    rather than down. """  # MobileNet 코드에서와 같이 텐서 깊이 스케일링 , 내림보단 반올림 하는 것 좋다.
    depths = [32, 16, 24, 40, 80, 96, 192, 320]
    return [_round_to_multiple_of(depth * alpha, 8) for depth in depths]


class MnasMulti(nn.Module):

    def __init__(self, alpha=1.0):
        super(MnasMulti, self).__init__()
        depths = _get_depths(alpha) #tensor의 depth scaling
        if alpha == 1.0:
            MNASNet = torchvision.models.mnasnet1_0(pretrained=True, progress=True)
        else:
            MNASNet = torchvision.models.MNASNet(alpha=alpha)
            """
                논문에서  image backbone is a variant of MnasNet [41] and is initialized with the weights pretrained from ImageNet
                MNASNet : 모바일에서 사용 가능한 구조를 자동으로 찾는 것 
                - 모델의 latency(실제 휴대폰에 모델을 실행시키는 방법)를 주 목표로 포함 시켜 accuray와 latency
                사이에 좋은 균형 이루는 최적의 모델 찾도록 함 
                - Novel factorized hierarchical search space
                네트워크 전체에서 레이어의 다양성 장려
                이전엔 적은 종류 cell 반복적으로 쌓아 검색 과정 단조롭 -> 다양성 낮음
                따라서 flexibility와 search space size 간 균형 맞추며 layer 구조적으로 다룰 수 있는 search space 제안 
                
                점진적으로 filter 사이즈 늘려나감 
                
                Transfer Learning
                pretrained = true : 미리 학습된 weight들 이어서 가져옴 
                
            """
        self.conv0 = nn.Sequential( # 입력값 하나에 데이터 순차적으로 처리 
            MNASNet.layers._modules['0'],
            MNASNet.layers._modules['1'],
            MNASNet.layers._modules['2'],
            MNASNet.layers._modules['3'],
            MNASNet.layers._modules['4'],
            MNASNet.layers._modules['5'],
            MNASNet.layers._modules['6'],
            MNASNet.layers._modules['7'],
            MNASNet.layers._modules['8'],
        )

        self.conv1 = MNASNet.layers._modules['9'] #modueles -> OrderDic : 입력된 순서 기억하는 딕셔너리
        self.conv2 = MNASNet.layers._modules['10']

        #depths = [32, 16, 24, 40, 80, 96, 192, 320]
                        #in_chan(n,c,h,w), out_chan, kernel_size
        self.out1 = nn.Conv2d(depths[4], depths[4], 1, bias=False)   # 80 X 80
        self.out_channels = [depths[4]]  # 80  , [32, 16, 24, 40, 80, 96, 192, 320]

        final_chs = depths[4]
        self.inner1 = nn.Conv2d(depths[3], final_chs, 1, bias=True)   # 40 X 80
        self.inner2 = nn.Conv2d(depths[2], final_chs, 1, bias=True)  #

        self.out2 = nn.Conv2d(final_chs, depths[3], 3, padding=1, bias=False)
        self.out3 = nn.Conv2d(final_chs, depths[2], 3, padding=1, bias=False)
        self.out_channels.append(depths[3])
        self.out_channels.append(depths[2])

    def forward(self, x):
        #MnasNet module 0~9
        conv0 = self.conv0(x)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)

        intra_feat = conv2
        outputs = []
        out = self.out1(intra_feat)
        outputs.append(out)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner1(conv1)
        out = self.out2(intra_feat)
        outputs.append(out)

        intra_feat = F.interpolate(intra_feat, scale_factor=2, mode="nearest") + self.inner2(conv0)
        out = self.out3(intra_feat)
        outputs.append(out)

        return outputs[::-1] #out 순서 뒤집기 out3,out2,out1
