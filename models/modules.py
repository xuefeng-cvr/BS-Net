import torch
import torch.nn.functional as F
import torch.nn as nn

class _UpProjection(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_UpProjection, self).__init__()

        self.conv1 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm2d(num_output_features)
        self.relu = nn.ReLU(inplace=True)
        self.conv1_2 = nn.Conv2d(num_output_features, num_output_features,
                                 kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1_2 = nn.BatchNorm2d(num_output_features)

        self.conv2 = nn.Conv2d(num_input_features, num_output_features,
                               kernel_size=5, stride=1, padding=2, bias=False)
        self.bn2 = nn.BatchNorm2d(num_output_features)

    def forward(self, x, size):
        x = F.upsample(x, size=size, mode='bilinear')
        x_conv1 = self.relu(self.bn1(self.conv1(x)))
        bran1 = self.bn1_2(self.conv1_2(x_conv1))
        bran2 = self.bn2(self.conv2(x))
        out = self.relu(bran1 + bran2)
        return out

class E_resnet(nn.Module):
    def __init__(self, original_model, num_features=2048):
        super(E_resnet, self).__init__()
        self.conv1 = original_model.conv1
        self.bn1 = original_model.bn1
        self.relu = original_model.relu
        self.maxpool = original_model.maxpool

        self.layer1 = original_model.layer1
        self.layer2 = original_model.layer2
        self.layer3 = original_model.layer3
        self.layer4 = original_model.layer4

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block3 = self.layer3(x_block2)
        x_block4 = self.layer4(x_block3)

        return x_block1, x_block2, x_block3, x_block4

class multi_dilated_layer(nn.Module):
    def __init__(self, input_channels,dilation_rate=[6, 12, 18]):
        super(multi_dilated_layer, self).__init__()
        self.rates = dilation_rate
        self.layer1 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//4, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//4, input_channels//4, 1),
            nn.ReLU(inplace=True)
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//4, 3, padding=6, dilation=self.rates[0]),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//4, input_channels//4, 1),
            nn.ReLU(inplace=True)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//4, 3, padding=12, dilation=self.rates[1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//4, input_channels//4, 1),
            nn.ReLU(inplace=True)
        )
        self.layer4 = nn.Sequential(
            nn.Conv2d(input_channels, input_channels//4, 3, padding=18, dilation=self.rates[2]),
            nn.ReLU(inplace=True),
            nn.Conv2d(input_channels//4, input_channels//4, 1),
            nn.ReLU(inplace=True)
        )
        self.concat_process = nn.Sequential(
            nn.Conv2d(input_channels, 1024, 1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):

        x1 = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x)
        x4 = self.layer4(x)

        x4_cat = torch.cat((x1, x2, x3, x4), 1)
        return x4_cat

class DCE(nn.Module): #DepthCorrelation Encoder
    def __init__(self, features, out_features, sizes=(1, 2, 3, 6)):
        super(DCE,self).__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.ups = nn.ModuleList([_UpProjection(out_features//2,out_features//2) for i in range(4)])
        self.bottleneck = nn.Conv2d(features//4*len(sizes), out_features//2, kernel_size=3,padding=1,bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.multi_layers = multi_dilated_layer(features)
        self.fusion = nn.Sequential(
            nn.Conv2d(in_channels=features//4*5, out_channels=features, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(features),
            nn.ReLU(inplace=True)
        )

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features//4, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        # pdb.set_trace()
        h, w = feats.size(2), feats.size(3)
        x4_cat = self.multi_layers(feats)  # 1024
        # pdb.set_trace()
        priors = [up(stage(feats), [h, w]) for (stage,up) in zip(self.stages,self.ups)]
        bottle = self.bottleneck(torch.cat(priors, 1))
        psp = self.relu(bottle)  # 1024
        fusion_feat = torch.cat((psp,x4_cat), 1)
        return self.fusion(fusion_feat)

class Decoder(nn.Module):
    def __init__(self, num_features=2048):
        super(Decoder, self).__init__()
        self.conv = nn.Conv2d(num_features, num_features //2, kernel_size=1, stride=1, bias=False)
        num_features = num_features // 2
        self.bn = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)

        self.up1 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up2 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up3 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

        self.up4 = _UpProjection(
            num_input_features=num_features, num_output_features=num_features // 2)
        num_features = num_features // 2

    def forward(self, x_block1, x_block2, x_block3, x_block4, x_dce):
        x_d1 = self.relu(self.bn(self.conv(x_dce)))
        x_d1 = self.up1(x_d1, [x_block3.size(2), x_block3.size(3)])
        x_d2 = self.up2(x_d1, [x_block2.size(2), x_block2.size(3)])
        x_d3 = self.up3(x_d2, [x_block1.size(2), x_block1.size(3)])
        x_d4 = self.up4(x_d3, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        return x_d4

class SRM(nn.Module):#Stripe Refinement
    def __init__(self,num_feature):
        super(SRM,self).__init__()
        self.ssp = SSP(64+num_feature//32)
        self.R = RP(num_feature//32)

    def forward(self,x_decoder,x_bubf):
        out = self.R(self.ssp(torch.cat((x_decoder, x_bubf), 1)))
        return out

class RP(nn.Module): #Residual prediction
    def __init__(self, block_channel=184):
        super(RP, self).__init__()
        num_features = 64 + block_channel
        self.conv0 = nn.Conv2d(num_features, num_features,kernel_size=5, stride=1, padding=2, bias=False)

        self.bn0 = nn.BatchNorm2d(num_features)

        self.conv1 = nn.Conv2d(num_features, num_features,kernel_size=5, stride=1, padding=2, bias=False)

        self.bn1 = nn.BatchNorm2d(num_features)

        self.conv2 = nn.Conv2d(
            num_features, 1, kernel_size=5, stride=1, padding=2, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # import pdb
        # pdb.set_trace()
        x0 = self.conv0(x)

        x0 = self.bn0(x0)
        x0 = self.relu(x0)
        x1 = self.conv1(x0)
        x1 = self.bn1(x1)
        x1 = self.relu(x1)
        x1 = x + x1
        x2 = self.conv2(x1)

        return x2

class SSP(nn.Module):#Strip Spatial Perception
    def __init__(self,inchannels,midchannels=21, k=11, w=3):
        super(SSP,self).__init__()
        self.conv1 = nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=(k, w), stride=1,
                               padding=(5, 1))
        self.conv2 = nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=(w, k), stride=1,
                               padding=(1, 5))
        self.conv5 = nn.Conv2d(in_channels=inchannels, out_channels=inchannels, kernel_size=3, stride=1,padding=1,bias=False)
        self.bn = nn.BatchNorm2d(num_features=inchannels)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        b1 = self.conv1(x)
        b2 = self.conv2(x)
        x = b1 + b2
        x = self.relu(self.bn(self.conv5(x)))

        return x

class lRB(nn.Module):   #large Eefinement Block
    def __init__(self, in_channels, out_channels):
        super(lRB, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU(inplace=True)
        self.bn = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1(x)
        res = self.conv2(x)
        res = self.bn(res)
        res = self.relu(res)
        res = self.conv3(res)
        return self.relu(x + res)

class BUBF(nn.Module):  #Bottom-Up Boundary Fusion
    def __init__(self, channels, out_channel):
        super(BUBF, self).__init__()
        self.lrb_1 = lRB(channels//8, out_channel)
        self.lrb_2 = lRB(channels//4, out_channel)
        self.lrb_3 = lRB(channels//2, out_channel)
        self.lrb_4 = lRB(channels, out_channel)
        self.lrb_5 = lRB(out_channel, out_channel)
        self.lrb_6 = lRB(out_channel, out_channel)
        self.lrb_7 = lRB(out_channel, out_channel)

        self.up1 = _UpProjection(out_channel, out_channel)
        self.up2 = _UpProjection(out_channel, out_channel)
        self.up3 = _UpProjection(out_channel, out_channel)
        self.up4 = _UpProjection(out_channel, out_channel)
    def forward(self, x_block1, x_block2, x_block3, x_block4):
        x1 = self.lrb_1(x_block1)
        x1 = self.up4(x1, [x_block1.size(2) * 2, x_block1.size(3) * 2])

        x2 = self.lrb_2(x_block2)
        x2 = self.up1(x2, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        x2 = x1 + x2
        x2 = self.lrb_5(x2)

        x3 = self.lrb_3(x_block3)
        x3 = self.up2(x3, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        x3 = x2 + x3
        x3 = self.lrb_6(x3)

        x4 = self.lrb_4(x_block4)
        x4 = self.up3(x4, [x_block1.size(2) * 2, x_block1.size(3) * 2])
        x4 = x3 + x4
        x4 = self.lrb_7(x4)
        return x4

