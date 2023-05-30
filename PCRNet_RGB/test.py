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
#!/usr/bin/python3
#coding=utf-8

import os
import sys
sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import cv2
import numpy as np
import matplotlib.pyplot as plt
plt.ion()
import time
import torch
import dataset
from torch.utils.data import DataLoader
from net import LDF
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "1"
class Test(object):
    def __init__(self, Dataset, Network, Path):
        ## dataset
        self.save_dir ='./out/1222_ssim_stage0/'
        self.epoch = 60
        self.cfg    = Dataset.Config(datapath=Path, snapshot=f'{self.save_dir}model-{self.epoch}', mode='test')
        #self.cfg    = Dataset.Config(datapath=Path, snapshot='./out/orifinal0408-3/model-50', mode='test')
        self.data   = Dataset.Data(self.cfg)
        self.loader = DataLoader(self.data, batch_size=1, shuffle=False, num_workers=8)
        ## network
        self.net    = Network(self.cfg)
        self.net.train(False)
        self.net.cuda()

    def save(self):
        time_s = time.time()
        img_num = len(self.loader)
        with torch.no_grad():
            for image, (H, W), name in self.loader:
                image, shape  = image.cuda().float(), (H, W)
                out = self.net(image, shape)
                #out  = out1
                pred = torch.sigmoid(out[0,0]).cpu().numpy()*255
                head = f'{self.save_dir}' + f'{self.epoch}/' + self.cfg.datapath.split('/')[-1]
                #head = f'{self.save_dir}ECSSD320'
                if not os.path.exists(head):
                    os.makedirs(head)
                cv2.imwrite(head + '/' + name[0] + '.png', np.round(pred))
                # pred = torch.sigmoid(outb2[0,0]).cpu().numpy()*255
                # head = f'{self.save_dir}'+'/body'
                # #head = self.cfg.datapath+'/oriModel_body3'
                # if not os.path.exists(head):
                #     os.makedirs(head)
                # cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))
                #
                # pred = torch.sigmoid(outd2[0,0]).cpu().numpy()*255
                # head = f'{self.save_dir}' + '/detail'
                # #head = self.cfg.datapath+'/oriModel_detail3'
                # if not os.path.exists(head):
                #     os.makedirs(head)
                # cv2.imwrite(head+'/'+name[0]+'.png', np.round(pred))
        time_e = time.time()
        print('Speed: %f FPS' % (img_num / (time_e - time_s)))

if __name__=='__main__':
    for set in ['PASCAL-S','ECSSD']:#'PASCAL-S','ECSSD','SOD','HKU','DUTS-TE','DUT-OMRON'
        path  = '../data/testset/'+set
        #path = '../data/DUTS-TR'
        t = Test(dataset, LDF, path)
        t.save_dir()
