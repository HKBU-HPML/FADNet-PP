import torch
import torch.nn as nn
from networks.modules import MatchLayer, UpSamplingLayer, FeatureLayer
from networks.submodules import warp_right_to_left, channel_length


class UCRefineNet(nn.Module):

    def __init__(self, scale=6, init_channel=16):

        super(UCRefineNet, self).__init__()

        self.scale = scale
        self.init_channel = init_channel

        # concat img0(3), img1(3), img1->img0(3), flow(1), diff-img(1)
        self.first_conv = nn.Conv2d(11, init_channel, 5, 1, 2)

        self.down_layers = []
        self.up_layers = []
        for i in range(scale):
            input_channel = init_channel*(2**i)
            output_channel = init_channel*(2**(i+1))
            if i == (self.scale - 1):
                up_input_channel = output_channel
            else:
                up_input_channel = output_channel*2 + 1
            up_output_channel = init_channel*(2**i)
            self.down_layers.append(FeatureLayer(input_channel=input_channel, output_channel=output_channel))
            self.up_layers.append(UpSamplingLayer(input_channel=up_input_channel, output_channel=up_output_channel))

        self.down_layers = nn.ModuleList(self.down_layers)
        self.up_layers = nn.ModuleList(self.up_layers)

        self.last_up_layer = nn.ConvTranspose2d(init_channel+3+1, init_channel, kernel_size=3, stride=1, padding=1)
        self.last_disp_up = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.last_disp_regr = nn.Conv2d(init_channel, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs, disps):

        down_feas = []
        up_feas = []
        final_disps = []

        down_feas.append(self.first_conv(inputs))
        for i in range(self.scale):
            down_fea = self.down_layers[i](down_feas[i])
            down_feas.append(down_fea)

        for i in range(self.scale, 0, -1):
            if i == (self.scale):
                up_fea, res_disp = self.up_layers[i-1](down_feas[i], None)
            else:
                input_fea = torch.cat((down_feas[i], up_feas[self.scale-i-1]), 1)
                up_fea, res_disp = self.up_layers[i-1](input_fea, final_disps[self.scale-i-1])

            final_disp = disps[self.scale-i] + res_disp
            final_disp = self.relu(final_disp)

            up_feas.append(up_fea)
            final_disps.append(final_disp)

        left_img = inputs[:, :3, :, :]
        last_up_disp = self.last_disp_up(final_disps[-1])
        last_input_fea = torch.cat((left_img, up_feas[-1], last_up_disp), 1)
        last_res_disp = self.last_disp_regr(self.last_up_layer(last_input_fea))

        final_disp = disps[-1] + last_res_disp
        final_disp = self.relu(final_disp)
        final_disps.append(final_disp)

        return final_disps


class UCNet(nn.Module):

    def __init__(self, maxdisp=192, scale=6, init_channel=16):

        super(UCNet, self).__init__()

        self.disp_range = [maxdisp // 3 * 2 // (2**i) for i in range(scale)]
        self.scale = scale
        self.init_channel = init_channel

        self.first_conv = nn.Conv2d(3, init_channel, 5, 1, 2)

        self.down_layers = []
        self.up_layers = []
        for i in range(scale):
            input_channel = init_channel*(2**i)
            output_channel = init_channel*(2**(i+1))
            corr_channel = self.disp_range[i]
            if i == (self.scale - 1):
                up_input_channel = output_channel*2
            else:
                up_input_channel = output_channel*3 + 1
            up_output_channel = init_channel*(2**(i))
            self.down_layers.append(MatchLayer(maxdisp=corr_channel, input_channel=input_channel, output_channel=output_channel))
            self.up_layers.append(UpSamplingLayer(input_channel=up_input_channel, output_channel=up_output_channel))

        self.down_layers = nn.ModuleList(self.down_layers)
        self.up_layers = nn.ModuleList(self.up_layers)

        self.last_up_layer = nn.ConvTranspose2d(init_channel+3+1, init_channel, kernel_size=3, stride=1, padding=1)
        self.last_disp_up = nn.ConvTranspose2d(1, 1, 4, 2, 1, bias=False)
        self.last_disp_regr = nn.Conv2d(init_channel, 1, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs):

        imgs = torch.chunk(inputs, 2, dim = 1)
        left_img = imgs[0]
        right_img = imgs[1]

        left_feas = []
        right_feas = []
        corr_feas = [0]   # a dummy element
        up_feas = []
        disps = []

        left_feas.append(self.first_conv(left_img))
        right_feas.append(self.first_conv(right_img))
        for i in range(self.scale):
            if i == 0:
                left_fea, right_fea, corr_fea = self.down_layers[i](left_feas[i], right_feas[i])
            else:
                left_fea, right_fea, corr_fea = self.down_layers[i](corr_feas[-1], right_feas[i])
            left_feas.append(left_fea)
            right_feas.append(right_fea)
            corr_feas.append(corr_fea)

        for i in range(self.scale, 0, -1):
            if i == (self.scale):
                input_feas = torch.cat((corr_feas[i], left_feas[i]), 1)
                up_fea, disp = self.up_layers[i-1](input_feas, None)
            else:
                input_feas = torch.cat((corr_feas[i], left_feas[i], up_feas[self.scale-i-1]), 1)
                up_fea, disp = self.up_layers[i-1](input_feas, disps[self.scale-i-1])

            disp = self.relu(disp)
            up_feas.append(up_fea)
            disps.append(disp)

        last_up_disp = self.last_disp_up(disps[-1])
        last_input_fea = torch.cat((left_img, up_feas[-1], last_up_disp), 1)
        last_disp = self.last_disp_regr(self.last_up_layer(last_input_fea))
        last_disp = self.relu(last_disp)
        disps.append(last_disp)

        return disps


class UCResNet(nn.Module):

    def __init__(self, maxdisp=192, scale=6, init_channel=16):

        super(UCResNet, self).__init__()

        self.disp_range = [maxdisp // (2**(i+1)) for i in range(scale)]
        self.scale = scale
        self.init_channel = init_channel

        self.basic_net = UCNet(maxdisp, scale, init_channel)
        self.refine_net = UCRefineNet(scale, init_channel)

    def forward(self, inputs):

        imgs = torch.chunk(inputs, 2, dim = 1)
        left_img = imgs[0]
        right_img = imgs[1]

        disps = self.basic_net(inputs)

        # warp img1 to img0; magnitude of diff between img0 and warped_img1,
        resampled_img1 = warp_right_to_left(inputs[:, 3:, :, :], -disps[-1])
        diff_img0 = inputs[:, :3, :, :] - resampled_img1
        norm_diff_img0 = channel_length(diff_img0)

        # concat img0, img1, img1->img0, flow, diff-img
        inputs_refine = torch.cat((inputs, resampled_img1, disps[-1], norm_diff_img0), dim = 1)

        refine_disps = self.refine_net(inputs_refine, disps)

        return disps, refine_disps
