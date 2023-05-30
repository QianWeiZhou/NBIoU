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
# coding=utf-8

import sys
import datetime

sys.path.insert(0, '../')
sys.dont_write_bytecode = True

import dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import pytorch_ssim
# from torch.utils.tensorboard import SummaryWriter
from apex import amp
from net import LDF
import os
import cv2
import numpy as np
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

SSIM = pytorch_ssim.SSIMLoss()
MSE = nn.MSELoss(reduce=True, size_average=True)
def iou_loss(pred, mask):
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    iou = 1 - (inter + 1) / (union - inter + 1)
    return iou.mean()
def iou_loss_bd(pred, bd,mask):
    pred = torch.sigmoid(pred)

    bd2  = torch.max(pred,bd)
    pred2 = torch.min(pred,bd)
    inter = (pred2 * mask).sum(dim=(2, 3))
    union = (pred2 + bd2).sum(dim=(2, 3))
    iou = 1 - (inter+1) / (union - inter+1)
    return iou.mean()


def rdPrint(anyThing, path='./'):
    orig_stdout = sys.stdout
    f = open(path, 'a+')
    sys.stdout = f

    print(anyThing)

    sys.stdout = orig_stdout
    f.close()


def train(Dataset, Network):
    ## dataset
    save_dir = './out/1222_IOU_stage0'
    img_dir = save_dir + '/img'
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    if not os.path.isdir(img_dir):
        os.mkdir(img_dir)
    cfg = Dataset.Config(datapath='../data/DUTS-TR2', savepath=save_dir, mode='train', batch=16, lr=0.03, momen=0.9,
                         decay=5e-4, epoch=60)
    data = Dataset.Data(cfg)
    loader = DataLoader(data, collate_fn=data.collate, batch_size=cfg.batch, shuffle=True, pin_memory=True,
                        num_workers=16)
    ## network
    net = Network(cfg)
    net.train(True)
    net.cuda()
    ## parameter
    base, head = [], []
    for name, param in net.named_parameters():
        if 'bkbone' in name:
            base.append(param)
        else:
            head.append(param)
    optimizer = torch.optim.SGD([{'params': base}, {'params': head}], lr=cfg.lr, momentum=cfg.momen,
                                weight_decay=cfg.decay, nesterov=True)
    net, optimizer = amp.initialize(net, optimizer, opt_level='O2')
    if not os.path.isdir(cfg.savepath):
        os.mkdir(cfg.savepath)
    os.system('mkdir -p {}/script'.format(os.path.join(cfg.savepath)))  # 要存放当前代码的地方
    os.system('cp -rfp ./*.py {}/script'.format(os.path.join(cfg.savepath)))  # 复制过去
    # checkpoint = torch.load('./out/1221_GRUv2_stage1_in_decoder/model-57')
    # epoch1 = 57
    # # # # # #net, optimizer = amp.initialize(net, optimizer, opt_level=opt_level)
    # net.load_state_dict(checkpoint)
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #amp.load_state_dict(checkpoint['amp'])
    # optimizer.param_groups[0]['lr'] = optimizer.param_groups[0]['lr'] / 10
    # optimizer.param_groups[1]['lr'] = optimizer.param_groups[1]['lr'] / 10
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


    model.train()
    for epoch in range(cfg.epoch):
        optimizer.param_groups[0]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr * 0.005
        optimizer.param_groups[1]['lr'] = (1 - abs((epoch + 1) / (cfg.epoch + 1) * 2 - 1)) * cfg.lr
        loss_epoch = 0
        global_step = 0
        for step, (image, mask, body, detail) in enumerate(loader):
            # curtime1 = datetime.datetime.now()
            image, mask, body, detail = image.cuda(), mask.cuda(), body.cuda(), detail.cuda()
            # print(image.shape)
            # quit()
            d1, b1, out1= net(image)

            lossb1 = F.binary_cross_entropy_with_logits(b1, body)+iou_loss_bd(b1, body,mask)
            lossd1 = F.binary_cross_entropy_with_logits(d1, detail)+iou_loss_bd(d1, detail,mask)
            loss1 = F.binary_cross_entropy_with_logits(out1, mask) + iou_loss(out1, mask)
            # lossb1 = SSIM(torch.sigmoid(b1),body)
            # lossd1 =SSIM(torch.sigmoid(d1),detail)
            # loss1 =SSIM(torch.sigmoid(out1),mask)

            loss_stage0 = lossb1 + lossd1 + loss1

            # lossb2 = F.binary_cross_entropy_with_logits(b2, body)+ 1-ssim_loss(body,torch.sigmoid(b2))
            # lossd2 = F.binary_cross_entropy_with_logits(d2, detail)+ 1-ssim_loss(detail,torch.sigmoid(d2))
            # loss2 = F.binary_cross_entropy_with_logits(out2, mask) + 1-ssim_loss(mask,torch.sigmoid(out2))+iou_loss(out2, mask)
            # # lossb2 = 1- ssim_loss(body,torch.sigmoid(b2))
            # # lossd2 = 1-ssim_loss(detail,torch.sigmoid(d2))
            # # loss2 = 1-ssim_loss(mask,torch.sigmoid(out2))
            # loss_stage1 = lossb2 + lossd2 + loss2
            #print( loss2)
            # lossb3 = F.binary_cross_entropy_with_logits(b3, body)
            # lossd3 = F.binary_cross_entropy_with_logits(d3, detail)
            # loss3 = F.binary_cross_entropy_with_logits(out3, mask) + iou_loss(out3, mask)
            # loss_stage2 = lossb3 + lossd3 + loss3
            # #
            # lossb4 = F.binary_cross_entropy_with_logits(b4, body)
            # lossd4 = F.binary_cross_entropy_with_logits(d4, detail)
            # loss4 = F.binary_cross_entropy_with_logits(out4, mask) + iou_loss(out4, mask)
            # loss_stage3 = lossb4 + lossd4 + loss4

            #loss = (loss_stage0 + loss_stage1+loss_stage2+loss_stage3)/4
            loss = loss_stage0
            #loss = (lossb1 + lossd1 + loss1)
            if step %100 ==0 :
                com4 = torch.sigmoid(out1[0, 0]).detach().cpu().numpy() * 255
                cv2.imwrite(img_dir + '/' + f'{epoch}.png', np.round(com4))
                com4 = mask[0, 0].detach().cpu().numpy() * 255
                cv2.imwrite(img_dir + '/' + f'mask{epoch}.png', np.round(com4))
            optimizer.zero_grad()
            with amp.scale_loss(loss, optimizer) as scale_loss:
                scale_loss.backward()
            optimizer.step()
            loss_epoch = loss_epoch + loss.item()
            

            global_step += 1
            # sw.add_scalar('lr'   , optimizer.param_groups[0]['lr'], global_step=global_step)
            # sw.add_scalars('loss', {'lossb1':lossb1.item(), 'lossd1':lossd1.item(), 'loss1':loss1.item(), 'lossb2':lossb2.item(), 'lossd2':lossd2.item(), 'loss2':loss2.item()}, global_step=global_step)
            if step % 10 == 0:
                print(
                    '%s | step:%d/%d | epoch:%d/%d |  lrbd=%.6f | lossb1=%.6f | lossd1=%.6f | loss1=%.6f |'# | lossb2=%.6f | lossd2=%.6f | loss2=%.6f'
                    % (datetime.datetime.now(), global_step, len(loader), epoch, cfg.epoch,
                       optimizer.param_groups[0]['lr'],
                       lossb1.item(), lossd1.item(), loss1.item()))#, lossb2.item(), lossd2.item(), loss2.item()))
            # if global_step == 1: break

        model.eval()
    if __name__=='__main__':
        for set in ['PASCAL-S','ECSSD']:#'PASCAL-S','ECSSD','SOD','HKU','DUTS-TE','DUT-OMRON'
        path  = '../data/testset/'+set
        #path = '../data/DUTS-TR'
        t = Test(dataset, LDF, path)
        t.save()
        ## log


        rdPrint('%i\t%.7f' % (epoch, loss_epoch / global_step),
                '{}/loss_epoch.log'.format(cfg.savepath))

        if epoch%10==0 or epoch > cfg.epoch * 3 / 4:
            torch.save(net.state_dict(), cfg.savepath + '/model-' + str(epoch + 1))


if __name__ == '__main__':
    train(dataset, LDF)
