# Copyright 2023 Chen Zhou
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#!/usr/bin/python3
# coding=utf-8
import torch.nn as nn
import torch
import torch.nn.functional as F
class ConvGRU(nn.Module):
    def __init__(self,inc=64,outc=64):
        super(ConvGRU, self).__init__()
        """GRU卷积流程
        args:
            x: input，新的循环的输入
            h_t_1: 上一次循环的输出
        shape：
            x: [batchsize, channels, width, lenth]
        """
        self.conv_x_z = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.conv_h_z = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.conv_x_r = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.conv_h_r = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.conv_u = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
        self.conv_out = nn.Conv2d(inc, outc, kernel_size=3, padding=1)  # (hidden_size, out_size)

        #verson 2:
        self.conv_zr = nn.Conv2d(inc*2, inc*2, kernel_size=3, padding=1)
        self.conv_h1 = nn.Conv2d(inc , inc , kernel_size=3, padding=1)
        self.conv_h2 = nn.Conv2d(inc, inc, kernel_size=3, padding=1)
    def forward(self, x, h_t_1):
        '''
        version2:

        combined = torch.cat((x, h_t_1), dim=1)  # concatenate along channel axis

        combined_conv = F.sigmoid(self.conv_zr(combined))

        z, r = torch.split(combined_conv, self.hidden_dim, dim=1)

        h_ = torch.tanh(self.conv_h1(input) + r * self.conv_h2(h_prev))

        h_cur = (1 - z) * h_ + z * h_prev
        :return h_cur
        '''
        print(x.shape)
        z_t = F.sigmoid(self.conv_x_z(x) + self.conv_h_z(h_t_1))
        print(z_t.shape)
        r_t = F.sigmoid((self.conv_x_r(x) + self.conv_h_r(h_t_1)))
        h_hat_t = torch.tanh(self.conv(x) + self.conv_u(torch.mul(r_t, h_t_1)))
        print(h_hat_t.shape)
        h_t = torch.mul((1 - z_t), h_t_1) + torch.mul(z_t, h_hat_t)
        print(h_t.shape)
        h_cur = self.conv_out(h_t)
        print(h_cur.shape)
        return h_cur




    # def initialize(self):
    #     weight_init(self)
if __name__ == '__main__':
    x = torch.randn(1, 64, 160, 160)
    h_t_1 = torch.randn(1, 64, 160, 160)
    model = ConvGRU(64,64)
    model.cuda()
    x,h_t_1 = x.cuda(),h_t_1.cuda()
    y_3, h_3 = model(x, h_t_1)

