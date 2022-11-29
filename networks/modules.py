import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal
from networks.submodules import build_corr, ResBlock

class MatchLayer(nn.Module):

    def __init__(self, maxdisp=192, input_channel=3, output_channel=32):
        super(MatchLayer, self).__init__()
        
        self.input_channel = input_channel
        self.output_channel = output_channel
        self.maxdisp = maxdisp

        self.corr_width = maxdisp

        # shrink and extract features
        self.block1 = ResBlock(input_channel, output_channel, stride=2)
        self.block2 = ResBlock(output_channel, output_channel, stride=1)

        self.corr_act = nn.LeakyReLU(0.1, inplace=True)
        self.corr_block = ResBlock(output_channel+maxdisp, output_channel, stride=1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, left_fea, right_fea):

        # split left image and right image
        block1_l = self.block1(left_fea)
        block2_l = self.block2(block1_l) 

        block1_r = self.block1(right_fea)
        block2_r = self.block2(block1_r)

        out_corr = build_corr(block2_l, block2_r, max_disp=self.maxdisp)
        out_corr = self.corr_act(out_corr)
        concat_corr = self.corr_block(torch.cat((block2_l, out_corr), 1))
        return block2_l, block2_r, concat_corr

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class FeatureLayer(nn.Module):

    def __init__(self, input_channel=3, output_channel=32):
        super(FeatureLayer, self).__init__()
        
        self.input_channel = input_channel
        self.output_channel = output_channel

        # shrink and extract features
        self.block1 = ResBlock(input_channel, output_channel, stride=2)
        self.block2 = ResBlock(output_channel, output_channel, stride=1)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, features):

        # extract high-level features
        block1 = self.block1(features)
        block2 = self.block2(block1) 

        return block2

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


class UpSamplingLayer(nn.Module):

    def __init__(self, input_channel, output_channel=16):
        super(UpSamplingLayer, self).__init__()
        
        self.input_channel = input_channel
        self.output_channel = output_channel

        self.block1 = nn.ConvTranspose2d(input_channel, output_channel, kernel_size=3, stride=1, padding=1, bias=False)
        self.block2 = nn.ConvTranspose2d(output_channel, output_channel, kernel_size=4, stride=2, padding=1, bias=False)
        self.act_fn = nn.LeakyReLU(0.1, inplace=True)

        self.disp_up = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.disp_regr = nn.Conv2d(output_channel, 1, kernel_size=3, stride=1, padding=1, bias=False)
        
        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, bottom_fea, bottom_disp=None):

        if bottom_disp != None:
            upsampled_disp = self.disp_up(bottom_disp)
            concat_fea = torch.cat((bottom_fea, upsampled_disp), 1)
        else:
            concat_fea = bottom_fea

        block1 = self.block1(concat_fea)
        disp = self.disp_regr(block1)

        block2 = self.block2(block1)
        block2 = self.act_fn(block2)

        return block2, disp

    def weight_parameters(self):
        return [param for name, param in self.named_parameters() if 'weight' in name]

    def bias_parameters(self):
        return [param for name, param in self.named_parameters() if 'bias' in name]


