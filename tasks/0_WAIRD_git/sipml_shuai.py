import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from copy import deepcopy
import copy
import numpy as np
from collections import OrderedDict
from torch.optim.lr_scheduler import StepLR
import timm
from timm import create_model as creat
model = creat('vit_base_patch16_224', pretrained=True, num_classes=1000)

class resnet_18(nn.Module):
    'resnet18'

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(resnet_18, self).__init__()
        self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))

    def update(self, x, device):
        bs = x.shape[0]
        emb = self.encoder(x.to(device))
        emb = emb.view((bs, -1))
        return emb


class DS_AE_CFRperfectADP(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(DS_AE_CFRperfectADP, self).__init__()
        self.hparams = hparams
        self.num_domain = 500# 6 选定有多少个domain
        self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))#输入通道是2
        # self.encoder = nn.Sequential(torchvision.models.resnet18(pretrained=False))  # 输入通道是3
        # self.encoder = nn.Sequential(torchvision.models.alexnet())
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )#4096
        self.encoder_withouchannel = nn.Sequential(
            torch.nn.Linear(4, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 1000)
        )#输出4096
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000 + 1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder = StepLR(self.optimizer_encoder, step_size=85, gamma=0.3)
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder_withouchannel = StepLR(self.optimizer_encoder_withouchannel, step_size=85, gamma=0.3)
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_decoder = StepLR(self.optimizer_decoder, step_size=85, gamma=0.3)
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_proj = StepLR(self.optimizer_proj, step_size=85, gamma=0.3)
        self.optimizer_clf_list = []
        self.scheduler_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            self.scheduler_clf_list.append(StepLR(self.optimizer_clf_list[i], step_size=85, gamma=0.3))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    # def update(self, z, x, y, a, device):  # x:channel; y:(x,y);
    #
    #     bs = x.shape[0]
    #     emb = self.encoder(x.to(device))
    #     emb = emb.view((bs, -1))
    #     emb_2 = self.encoder_withouchannel(a.to(device))
    #     emb = torch.concat([emb,emb_2], dim = 1)
    #     # print(emb.shape)
    #     x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
    #     emb = emb.view((bs, -1))
    #     # print("emb",emb.shape)
    #     y_hat = torch.zeros(bs, 2)
    #
    #     ############################
    #     #  y_hat 的逻辑改一下
    #     # y_hat = self.clf(emb)
    #     for i in range(bs):
    #         y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
    #         # y_hat_temp = self.clf_list[0](emb[i])
    #         if i == 0:
    #             y_hat = y_hat_temp.reshape(1,2)
    #         else:
    #             y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
    #     ############################1,2
    #     loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
    #     self.optimizer_encoder.zero_grad()
    #     self.optimizer_encoder_withouchannel.zero_grad()
    #     self.optimizer_decoder.zero_grad()
    #     self.optimizer_proj.zero_grad()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].zero_grad()
    #     loss.backward()
    #     self.optimizer_encoder.step()
    #     self.optimizer_encoder_withouchannel.step()
    #     self.optimizer_decoder.step()
    #     self.optimizer_proj.step()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].step()
    #     result = {"loss": ((y - y_hat) ** 2).mean()}
    #     return result

    def update(self, z, x, y, a, epoch, device):  # x:channel; y:(x,y);
        # decay = 0.5
        # if epoch == 85:# 每迭代20次，更新一次学习率
        #     for params in self.optimizer_encoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_encoder_withouchannel.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_decoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_proj.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for i in range(self.num_domain):
        #         for params in self.optimizer_clf_list[i].param_groups:
        #             # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #             params['lr'] *= decay

        bs = x.shape[0]
        emb = self.encoder(x.to(device))
        emb = emb.view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb,emb_2], dim = 1)
        # print(emb.shape)
        x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
        self.optimizer_encoder.zero_grad()
        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb, emb_2], dim=1)
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat


class DS_AE_bothmulti(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(DS_AE_bothmulti, self).__init__()
        self.hparams = hparams
        self.num_domain = 35# 6 选定有多少个domain
        # self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))#输入通道是2
        self.encoder = []
        # self.encoder = nn.Sequential(torchvision.models.resnet18(pretrained=False))  # 输入通道是3
        self.encoder_withouchannel = nn.Sequential(
            torch.nn.Linear(4, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 1000)
        )
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(4096 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            # self.encoder.append(torchvision.models.resnet18(pretrained=False))
            self.encoder.append(nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ))
            self.clf_list.append(torch.nn.Linear(4096 + 1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.encoder[i] = self.encoder[i].to(device)
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = []
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            self.optimizer_encoder.append(torch.optim.Adam(
                self.encoder[i].parameters()
                , lr=self.hparams["lr"]
            ))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    def update(self, z, x, y, a, device):  # x:channel; y:(x,y);

        bs = x.shape[0]
        # emb = self.encoder(x.to(device))
        for i in range(bs):
            emb_temp = self.encoder[z[i]](x[i].unsqueeze(0).to(device))#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                emb = emb_temp.reshape(1,-1)
            else:
                emb = torch.concat([emb, emb_temp.reshape(1,-1)],dim = 0)

        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb,emb_2], dim = 1)
        # print(emb.shape)
        x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?

        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_encoder[i].zero_grad()
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_encoder[i].step()
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        # emb = self.encoder(x.to(device)).view((bs, -1))
        for i in range(bs):
            emb_temp = self.encoder[z[i]](x[i].unsqueeze(0).to(device))#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                emb = emb_temp.reshape(1,-1)
            else:
                emb = torch.concat([emb, emb_temp.reshape(1,-1)],dim = 0)
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb, emb_2], dim=1)
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat


class DS_AE_noprior(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(DS_AE_noprior, self).__init__()
        self.hparams = hparams
        self.num_domain = 14# 6 选定有多少个domain
        self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))
        self.encoder_withouchannel = nn.Sequential(
            torch.nn.Linear(7, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 1000)
        )
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    def update(self, z, x, y, a, device):  # x:channel; y:(x,y);

        bs = x.shape[0]
        emb = self.encoder(x.to(device))
        # emb_2 = self.encoder_withouchannel(a.to(device))
        # emb = torch.concat([emb,emb_2], dim = 1)
        # print(emb.shape)
        x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() + ((x - x_hat) ** 2).mean()  # ?
        self.optimizer_encoder.zero_grad()
        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1))
        # emb_2 = self.encoder_withouchannel(a.to(device))
        # emb = torch.concat([emb, emb_2], dim=1)
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat


class CNN_prior(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(CNN_prior, self).__init__()
        self.hparams = hparams
        self.num_domain = 50# 6 选定有多少个domain
        # self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))
        # self.encoder = nn.Sequential(torchvision.models.resnet18(pretrained=False))
        self.encoder = nn.Sequential(torchvision.models.alexnet())
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        self.encoder_withouchannel = nn.Sequential(
            torch.nn.Linear(4, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 1000)
        )
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000 + 1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    def update(self, z, x, y, a, epoch, device):  # x:channel; y:(x,y);

        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb,emb_2], dim = 1)
        # print(emb.shape)
        # x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            # y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
        self.optimizer_encoder.zero_grad()
        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb, emb_2], dim=1)
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            # y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat


class DS_AE_CL(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(DS_AE_CL, self).__init__()
        self.hparams = hparams
        self.num_domain = 26
        self.encoder_1 = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))
        self.encoder_2 = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
                                       torchvision.models.resnet18(pretrained=False))
        self.proj = torch.nn.Linear(1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000, 2))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder_1 = torch.optim.Adam(
            self.encoder_1.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_encoder_2 = torch.optim.Adam(
            self.encoder_2.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))
    def contrastive_loss(self, emb_i, emb_j):
        device = "cuda"
        self.batch_size = emb_i.shape[0]
        batch_size = self.batch_size
        self.register_buffer("temperature", torch.tensor(self.hparams['temperature']).to(device))
        self.register_buffer("negatives_mask",
                             (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
        z_i = F.normalize(emb_i, dim=1)  # (batch_size, dim)  --->  (batch_size, dim) L2
        z_j = F.normalize(emb_j, dim=1)  # (batch_size, dim)  --->  (batch_size, dim) L2

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*batch_size, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # batch_size
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # batch_size
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*batch_size

        nominator = torch.exp(positives / self.temperature)  # 2*batch_size
        denominator = self.negatives_mask * torch.exp(
            similarity_matrix / self.temperature)  # 2*batch_size, 2*batch_size
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*batch_size
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    def update(self, z, x, y, device):  # x:channel; y:(x,y);

        bs = x.shape[0]
        emb = self.encoder_1(x.to(device))
        emb_2 = self.encoder_2(x.to(device))
        # print(emb.shape)
        x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() + ((x - x_hat) ** 2).mean() #+ self.contrastive_loss(emb, emb_2)# ?
        self.optimizer_encoder_1.zero_grad()
        self.optimizer_encoder_2.zero_grad()
        self.optimizer_proj.zero_grad()
        self.optimizer_decoder.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder_1.step()
        self.optimizer_encoder_2.step()
        self.optimizer_proj.step()
        self.optimizer_decoder.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, device):
        bs = x.shape[0]
        emb = self.encoder_1(x.to(device)).view((bs, -1))
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat

class AE_CL(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(AE_CL, self).__init__()
        self.hparams = hparams
        self.num_domain = 26
        self.encoder_1 = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))
        self.encoder_2 = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
                                       torchvision.models.resnet18(pretrained=False))
        self.proj = torch.nn.Linear(1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000, 2))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder_1 = torch.optim.Adam(
            self.encoder_1.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_encoder_2 = torch.optim.Adam(
            self.encoder_2.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))
    def contrastive_loss(self, emb_i, emb_j):
        device = "cuda"
        self.batch_size = emb_i.shape[0]
        batch_size = self.batch_size
        self.register_buffer("temperature", torch.tensor(self.hparams['temperature']).to(device))
        self.register_buffer("negatives_mask",
                             (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
        z_i = F.normalize(emb_i, dim=1)  # (batch_size, dim)  --->  (batch_size, dim) L2
        z_j = F.normalize(emb_j, dim=1)  # (batch_size, dim)  --->  (batch_size, dim) L2

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*batch_size, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # batch_size
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # batch_size
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*batch_size

        nominator = torch.exp(positives / self.temperature)  # 2*batch_size
        denominator = self.negatives_mask * torch.exp(
            similarity_matrix / self.temperature)  # 2*batch_size, 2*batch_size
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*batch_size
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss
    def update(self, z, x, y, device):  # x:channel; y:(x,y);

        bs = x.shape[0]
        emb = self.encoder_1(x.to(device))
        emb_2 = self.encoder_2(x.to(device))
        # print(emb.shape)
        x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)

        y_hat = self.clf_list[0](emb)

        loss = ((y - y_hat) ** 2).mean() + ((x - x_hat) ** 2).mean() + self.contrastive_loss(emb, emb_2)#
        self.optimizer_encoder_1.zero_grad()
        self.optimizer_encoder_2.zero_grad()
        self.optimizer_proj.zero_grad()
        self.optimizer_decoder.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder_1.step()
        self.optimizer_encoder_2.step()
        self.optimizer_proj.step()
        self.optimizer_decoder.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, device):
        bs = x.shape[0]
        emb = self.encoder_1(x.to(device)).view((bs, -1))
        y_hat = self.clf_list[0](emb)


        return y_hat
class CNN(nn.Module):
        '''
        Domain-Specific Auto-Encoder
        '''

        def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
            super(CNN, self).__init__()
            self.hparams = hparams
            self.num_domain = 500
            self.encoder_1 = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                                           torchvision.models.resnet18(pretrained = False)) #首通道为1的ADP
            # self.encoder_1 = nn.Sequential(torchvision.models.resnet18(pretrained=False)) #首通道为3
            # self.encoder_1 = nn.Sequential(torchvision.models.alexnet())#1000
            # self.encoder_1 = nn.Sequential(
            #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(),
            #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(),
            #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            #     nn.BatchNorm2d(64),
            #     nn.ReLU(),
            # )#4096
            # self.encoder_1 = nn.Sequential(
            #     nn.Linear(3, 256),
            #     nn.ReLU(),
            #     nn.Linear(256, 1024),
            #     nn.ReLU(),
            #     nn.Linear(1024, 4096),
            #     nn.ReLU(),
            #     nn.Linear(4096, 1000),
            #     nn.ReLU(),
            # )
            self.encoder_2 = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
                                           torchvision.models.resnet18(pretrained=False))#首通道为2的ADP
            self.proj = torch.nn.Linear(4096, 7 * 7 * 64)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
                nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
                nn.ReLU(),  # 6*2-2+3+1=14
                nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),  # 28
                nn.BatchNorm2d(16),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),  # 56
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),  # 112
                nn.BatchNorm2d(8),
                nn.ReLU(),
                nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                                   padding=1, output_padding=1)  # 224
            )
            self.clf_list = []
            for i in range(self.num_domain):
                self.clf_list.append(torch.nn.Linear(1000, 2))
                self.clf_list[i] = self.clf_list[i].to(device)
            m = 0.95
            self.optimizer_encoder_1 = torch.optim.Adam(
                self.encoder_1.parameters()
                , lr=self.hparams["lr"]
            )
            self.optimizer_encoder_2 = torch.optim.Adam(
                self.encoder_2.parameters()
                , lr=self.hparams["lr"]
            )
            self.optimizer_decoder = torch.optim.Adam(
                self.decoder.parameters()
                , lr=self.hparams["lr"]
            )
            self.optimizer_proj = torch.optim.Adam(
                self.proj.parameters()
                , lr=self.hparams["lr"]
            )
            self.optimizer_clf_list = []
            for i in range(self.num_domain):
                self.optimizer_clf_list.append(torch.optim.Adam(
                    self.clf_list[i].parameters()
                    , lr=self.hparams["lr"]
                ))
                # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

        # def __info__(self):
        #     print("\t Number of samples: {}".format(len(self.image_path)))
        def contrastive_loss(self, emb_i, emb_j):
            device = "cuda"
            self.batch_size = emb_i.shape[0]
            batch_size = self.batch_size
            self.register_buffer("temperature", torch.tensor(self.hparams['temperature']).to(device))
            self.register_buffer("negatives_mask",
                                 (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
            z_i = F.normalize(emb_i, dim=1)  # (batch_size, dim)  --->  (batch_size, dim) L2
            z_j = F.normalize(emb_j, dim=1)  # (batch_size, dim)  --->  (batch_size, dim) L2

            representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*batch_size, dim)
            similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
            sim_ij = torch.diag(similarity_matrix, self.batch_size)  # batch_size
            sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # batch_size
            positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*batch_size

            nominator = torch.exp(positives / self.temperature)  # 2*batch_size
            denominator = self.negatives_mask * torch.exp(
                similarity_matrix / self.temperature)  # 2*batch_size, 2*batch_size
            loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*batch_size
            loss = torch.sum(loss_partial) / (2 * self.batch_size)
            return loss

        def update(self, z, x, y, device):  # x:channel; y:(x,y);

            bs = x.shape[0]
            # emb = self.encoder_1(x.to(device)) #ADP
            emb = self.encoder_2(x.to(device)) #CFR
            # emb_2 = self.encoder_2(x.to(device))
            # print(emb.shape)
            # x_hat = self.decoder(self.proj(emb).reshape(bs, 64, 7, 7))
            emb = emb.view((bs, -1))
            # print("emb",emb.shape)

            y_hat = self.clf_list[0](emb)

            loss = ((y - y_hat) ** 2).mean()
            self.optimizer_encoder_1.zero_grad()
            self.optimizer_encoder_2.zero_grad()
            self.optimizer_proj.zero_grad()
            self.optimizer_decoder.zero_grad()
            for i in range(self.num_domain):
                self.optimizer_clf_list[i].zero_grad()
            loss.backward()
            self.optimizer_encoder_1.step()
            self.optimizer_encoder_2.step()
            self.optimizer_proj.step()
            self.optimizer_decoder.step()
            for i in range(self.num_domain):
                self.optimizer_clf_list[i].step()
            result = {"loss": ((y - y_hat) ** 2).mean()}
            return result

        def predict(self, z, x, device):
            bs = x.shape[0]
            # emb = self.encoder_1(x.to(device)).view((bs, -1)) #ADP
            emb = self.encoder_2(x.to(device)).view((bs, -1)) #CFR
            y_hat = self.clf_list[0](emb)

            return y_hat


class DS_CNN(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(DS_CNN, self).__init__()
        self.hparams = hparams
        self.num_domain = 26
        self.encoder_1 = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
                                       torchvision.models.resnet18(pretrained=False))
        self.encoder_2 = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
                                       torchvision.models.resnet18(pretrained=False))
        self.proj = torch.nn.Linear(1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000, 2))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder_1 = torch.optim.Adam(
            self.encoder_1.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_encoder_2 = torch.optim.Adam(
            self.encoder_2.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))
    def contrastive_loss(self, emb_i, emb_j):
        device = "cuda"
        self.batch_size = emb_i.shape[0]
        batch_size = self.batch_size
        self.register_buffer("temperature", torch.tensor(self.hparams['temperature']).to(device))
        self.register_buffer("negatives_mask",
                             (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())
        z_i = F.normalize(emb_i, dim=1)  # (batch_size, dim)  --->  (batch_size, dim) L2
        z_j = F.normalize(emb_j, dim=1)  # (batch_size, dim)  --->  (batch_size, dim) L2

        representations = torch.cat([z_i, z_j], dim=0)  # repre: (2*batch_size, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)  # batch_size
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)  # batch_size
        positives = torch.cat([sim_ij, sim_ji], dim=0)  # 2*batch_size

        nominator = torch.exp(positives / self.temperature)  # 2*batch_size
        denominator = self.negatives_mask * torch.exp(
            similarity_matrix / self.temperature)  # 2*batch_size, 2*batch_size
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))  # 2*batch_size
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss

    def update(self, z, x, y, device):  # x:channel; y:(x,y);

        bs = x.shape[0]
        emb = self.encoder_1(x.to(device))
        emb_2 = self.encoder_2(x.to(device))
        # print(emb.shape)
        x_hat = self.decoder(self.proj(emb).reshape(bs, 64, 7, 7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)

        y_hat = self.clf_list[0](emb)

        loss = ((y - y_hat) ** 2).mean()
        self.optimizer_encoder_1.zero_grad()
        self.optimizer_encoder_2.zero_grad()
        self.optimizer_proj.zero_grad()
        self.optimizer_decoder.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder_1.step()
        self.optimizer_encoder_2.step()
        self.optimizer_proj.step()
        self.optimizer_decoder.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, device):
        bs = x.shape[0]
        emb = self.encoder_1(x.to(device)).view((bs, -1))
        y_hat = self.clf_list[0](emb)

        return y_hat

class DS_AE_multi(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(DS_AE_multi, self).__init__()
        self.hparams = hparams
        self.num_domain = 8# 6 选定有多少个domain
        self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))
        self.encoder_withouchannel = nn.Sequential(
            torch.nn.Linear(7, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 1000)
        )
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000, 2))
            # self.clf_list.append(nn.Sequential(
            #     torch.nn.Linear(1000, 512),
            #     nn.ReLU(),
            #     torch.nn.Linear(512, 64),
            #     nn.ReLU(),
            #     torch.nn.Linear(64, 2)
            # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    def update(self, z, x, y, a, device):  # x:channel; y:(x,y);

        bs = x.shape[0]
        emb = self.encoder(x.to(device))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = emb * emb_2
        # emb = torch.concat([emb,emb_2], dim = 1)
        # print(emb.shape)
        x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() + ((x - x_hat) ** 2).mean()  # ?
        self.optimizer_encoder.zero_grad()
        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = emb * emb_2
        # emb = torch.concat([emb, emb_2], dim=1)
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat


class classsify_DS_AE_bothmulti(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device, class_z, class_num):  # (channel, z, UE_x, UE_y)
        super(classsify_DS_AE_bothmulti, self).__init__()
        self.hparams = hparams
        self.num_domain = 35# 6 选定有多少个domain
        # self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))#输入通道是2
        self.encoder = []
        # self.encoder = nn.Sequential(torchvision.models.resnet18(pretrained=False))  # 输入通道是3
        self.encoder_withouchannel = nn.Sequential(
            torch.nn.Linear(4, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 1000)
        )
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(4096 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            # self.encoder.append(torchvision.models.resnet18(pretrained=False))
            self.encoder.append(nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        ))
            self.clf_list.append(torch.nn.Linear(4096 + 1000, class_num[class_z.index(i)]))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.encoder[i] = self.encoder[i].to(device)
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = []
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            self.optimizer_encoder.append(torch.optim.Adam(
                self.encoder[i].parameters()
                , lr=self.hparams["lr"]
            ))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    def update(self, z, x, y, a, device):  # x:channel; y:(x,y);

        bs = x.shape[0]
        # emb = self.encoder(x.to(device))
        for i in range(bs):
            emb_temp = self.encoder[z[i]](x[i].unsqueeze(0).to(device))#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                emb = emb_temp.reshape(1,-1)
            else:
                emb = torch.concat([emb, emb_temp.reshape(1,-1)],dim = 0)

        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb,emb_2], dim = 1)
        # print(emb.shape)
        # #解码并且放在loss当中的时候才用
        # x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?

        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_encoder[i].zero_grad()
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_encoder[i].step()
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        # emb = self.encoder(x.to(device)).view((bs, -1))
        for i in range(bs):
            emb_temp = self.encoder[z[i]](x[i].unsqueeze(0).to(device))#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                emb = emb_temp.reshape(1,-1)
            else:
                emb = torch.concat([emb, emb_temp.reshape(1,-1)],dim = 0)
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb, emb_2], dim=1)
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat


'''
CFR+ADP
'''
class DS_AE_CFRADP(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(DS_AE_CFRADP, self).__init__()
        self.hparams = hparams
        self.num_domain = 500# 6 选定有多少个domain
        # self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))#输入通道是2
        self.encoder = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
                                     timm.create_model('resnet18', pretrained=False, num_classes=100))  # 输入通道是2
        # self.encoder = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
        #                              timm.create_model('densenet121', pretrained=False, num_classes=100))  # 输入通道是2
        # self.encoder = nn.Sequential(torchvision.models.resnet18(pretrained=False))  # 输入通道是3
        # self.encoder = nn.Sequential(torchvision.models.alexnet())
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )#4096
        self.encoder_withouchannel = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                                           torchvision.models.resnet18(pretrained = False) # 输入通道是1
        )#输出4096
        # self.encoder_withouchannel = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
        #                              timm.create_model('densenet121', pretrained=False)
        # )#输出4096
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(100 + 1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder = StepLR(self.optimizer_encoder, step_size=85, gamma=0.3)
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder_withouchannel = StepLR(self.optimizer_encoder_withouchannel, step_size=85, gamma=0.3)
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_decoder = StepLR(self.optimizer_decoder, step_size=85, gamma=0.3)
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_proj = StepLR(self.optimizer_proj, step_size=85, gamma=0.3)
        self.optimizer_clf_list = []
        self.scheduler_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            self.scheduler_clf_list.append(StepLR(self.optimizer_clf_list[i], step_size=85, gamma=0.3))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    # def update(self, z, x, y, a, device):  # x:channel; y:(x,y);
    #
    #     bs = x.shape[0]
    #     emb = self.encoder(x.to(device))
    #     emb = emb.view((bs, -1))
    #     emb_2 = self.encoder_withouchannel(a.to(device))
    #     emb = torch.concat([emb,emb_2], dim = 1)
    #     # print(emb.shape)
    #     x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
    #     emb = emb.view((bs, -1))
    #     # print("emb",emb.shape)
    #     y_hat = torch.zeros(bs, 2)
    #
    #     ############################
    #     #  y_hat 的逻辑改一下
    #     # y_hat = self.clf(emb)
    #     for i in range(bs):
    #         y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
    #         # y_hat_temp = self.clf_list[0](emb[i])
    #         if i == 0:
    #             y_hat = y_hat_temp.reshape(1,2)
    #         else:
    #             y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
    #     ############################1,2
    #     loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
    #     self.optimizer_encoder.zero_grad()
    #     self.optimizer_encoder_withouchannel.zero_grad()
    #     self.optimizer_decoder.zero_grad()
    #     self.optimizer_proj.zero_grad()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].zero_grad()
    #     loss.backward()
    #     self.optimizer_encoder.step()
    #     self.optimizer_encoder_withouchannel.step()
    #     self.optimizer_decoder.step()
    #     self.optimizer_proj.step()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].step()
    #     result = {"loss": ((y - y_hat) ** 2).mean()}
    #     return result

    def update(self, z, x, y, a, epoch, device):  # x:channel; y:(x,y);
        # decay = 0.5
        # if epoch == 85:# 每迭代20次，更新一次学习率
        #     for params in self.optimizer_encoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_encoder_withouchannel.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_decoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_proj.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for i in range(self.num_domain):
        #         for params in self.optimizer_clf_list[i].param_groups:
        #             # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #             params['lr'] *= decay

        bs = x.shape[0]
        emb = self.encoder(x.to(device))
        emb = emb.view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb,emb_2], dim = 1)
        # print(emb.shape)
        # x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
        self.optimizer_encoder.zero_grad()
        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb, emb_2], dim=1)
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat

class CNN_prior_CFRADP(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(CNN_prior_CFRADP, self).__init__()
        self.hparams = hparams
        self.num_domain = 500# 6 选定有多少个domain
        # self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))
        # self.encoder = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1), torchvision.models.resnet18(pretrained=False))
        self.encoder = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
                                     timm.create_model('resnet18', pretrained=False, num_classes=100))  # 输入通道是2
        # self.encoder = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
        #                              timm.create_model('densenet121', pretrained=False, num_classes=100))  # 输入通道是2
        # self.encoder = nn.Sequential(torchvision.models.alexnet())
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )
        self.encoder_withouchannel = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                                           torchvision.models.resnet18(pretrained = False)) # 输入通道是1

        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(100 + 1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.optimizer_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    def update(self, z, x, y, a, epoch, device):  # x:channel; y:(x,y);

        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb,emb_2], dim = 1)
        # print(emb.shape)
        # x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            # y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
        self.optimizer_encoder.zero_grad()
        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb, emb_2], dim=1)
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            # y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat


class DS_AE_manmade(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(DS_AE_manmade, self).__init__()
        self.hparams = hparams
        self.num_domain = 500# 6 选定有多少个domain
        self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))#输入通道是2
        # self.encoder = nn.Sequential(torchvision.models.resnet18(pretrained=False))  # 输入通道是3
        # self.encoder = nn.Sequential(torchvision.models.alexnet())
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )#4096
        self.encoder_withouchannel = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                                           torchvision.models.resnet18(pretrained = False) # 输入通道是1
        )#输出4096
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder = StepLR(self.optimizer_encoder, step_size=85, gamma=0.3)
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder_withouchannel = StepLR(self.optimizer_encoder_withouchannel, step_size=85, gamma=0.3)
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_decoder = StepLR(self.optimizer_decoder, step_size=85, gamma=0.3)
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_proj = StepLR(self.optimizer_proj, step_size=85, gamma=0.3)
        self.optimizer_clf_list = []
        self.scheduler_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            self.scheduler_clf_list.append(StepLR(self.optimizer_clf_list[i], step_size=85, gamma=0.3))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    # def update(self, z, x, y, a, device):  # x:channel; y:(x,y);
    #
    #     bs = x.shape[0]
    #     emb = self.encoder(x.to(device))
    #     emb = emb.view((bs, -1))
    #     emb_2 = self.encoder_withouchannel(a.to(device))
    #     emb = torch.concat([emb,emb_2], dim = 1)
    #     # print(emb.shape)
    #     x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
    #     emb = emb.view((bs, -1))
    #     # print("emb",emb.shape)
    #     y_hat = torch.zeros(bs, 2)
    #
    #     ############################
    #     #  y_hat 的逻辑改一下
    #     # y_hat = self.clf(emb)
    #     for i in range(bs):
    #         y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
    #         # y_hat_temp = self.clf_list[0](emb[i])
    #         if i == 0:
    #             y_hat = y_hat_temp.reshape(1,2)
    #         else:
    #             y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
    #     ############################1,2
    #     loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
    #     self.optimizer_encoder.zero_grad()
    #     self.optimizer_encoder_withouchannel.zero_grad()
    #     self.optimizer_decoder.zero_grad()
    #     self.optimizer_proj.zero_grad()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].zero_grad()
    #     loss.backward()
    #     self.optimizer_encoder.step()
    #     self.optimizer_encoder_withouchannel.step()
    #     self.optimizer_decoder.step()
    #     self.optimizer_proj.step()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].step()
    #     result = {"loss": ((y - y_hat) ** 2).mean()}
    #     return result

    def update(self, z, x, y, a, epoch, device):  # x:channel; y:(x,y);
        # decay = 0.5
        # if epoch == 85:# 每迭代20次，更新一次学习率
        #     for params in self.optimizer_encoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_encoder_withouchannel.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_decoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_proj.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for i in range(self.num_domain):
        #         for params in self.optimizer_clf_list[i].param_groups:
        #             # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #             params['lr'] *= decay

        bs = x.shape[0]
        emb = self.encoder(x.to(device)) #CFR
        # emb = self.encoder_withouchannel(a.to(device)) #ADP
        emb = emb.view((bs, -1))

        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
        self.optimizer_encoder.zero_grad()
        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1)) #CFR
        # emb = self.encoder_withouchannel(a.to(device)).view((bs, -1)) #ADP

        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat


class DS_AE_CFRorADP(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(DS_AE_CFRorADP, self).__init__()
        self.hparams = hparams
        self.num_domain = 500# 6 选定有多少个domain
        self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))#输入通道是2
        # self.encoder = nn.Sequential(torchvision.models.resnet18(pretrained=False))  # 输入通道是3
        # self.encoder = nn.Sequential(torchvision.models.alexnet())
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )#4096
        self.encoder_withouchannel = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                                           torchvision.models.resnet18(pretrained = False) # 输入通道是1
        )#输出4096
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder = StepLR(self.optimizer_encoder, step_size=85, gamma=0.3)
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder_withouchannel = StepLR(self.optimizer_encoder_withouchannel, step_size=85, gamma=0.3)
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_decoder = StepLR(self.optimizer_decoder, step_size=85, gamma=0.3)
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_proj = StepLR(self.optimizer_proj, step_size=85, gamma=0.3)
        self.optimizer_clf_list = []
        self.scheduler_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            self.scheduler_clf_list.append(StepLR(self.optimizer_clf_list[i], step_size=85, gamma=0.3))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    # def update(self, z, x, y, a, device):  # x:channel; y:(x,y);
    #
    #     bs = x.shape[0]
    #     emb = self.encoder(x.to(device))
    #     emb = emb.view((bs, -1))
    #     emb_2 = self.encoder_withouchannel(a.to(device))
    #     emb = torch.concat([emb,emb_2], dim = 1)
    #     # print(emb.shape)
    #     x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
    #     emb = emb.view((bs, -1))
    #     # print("emb",emb.shape)
    #     y_hat = torch.zeros(bs, 2)
    #
    #     ############################
    #     #  y_hat 的逻辑改一下
    #     # y_hat = self.clf(emb)
    #     for i in range(bs):
    #         y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
    #         # y_hat_temp = self.clf_list[0](emb[i])
    #         if i == 0:
    #             y_hat = y_hat_temp.reshape(1,2)
    #         else:
    #             y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
    #     ############################1,2
    #     loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
    #     self.optimizer_encoder.zero_grad()
    #     self.optimizer_encoder_withouchannel.zero_grad()
    #     self.optimizer_decoder.zero_grad()
    #     self.optimizer_proj.zero_grad()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].zero_grad()
    #     loss.backward()
    #     self.optimizer_encoder.step()
    #     self.optimizer_encoder_withouchannel.step()
    #     self.optimizer_decoder.step()
    #     self.optimizer_proj.step()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].step()
    #     result = {"loss": ((y - y_hat) ** 2).mean()}
    #     return result

    def update(self, z, x, y, a, epoch, device):  # x:channel; y:(x,y);
        # decay = 0.5
        # if epoch == 85:# 每迭代20次，更新一次学习率
        #     for params in self.optimizer_encoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_encoder_withouchannel.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_decoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_proj.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for i in range(self.num_domain):
        #         for params in self.optimizer_clf_list[i].param_groups:
        #             # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #             params['lr'] *= decay

        bs = x.shape[0]
        # emb = self.encoder(x.to(device)) #CFR
        emb = self.encoder_withouchannel(a.to(device)) #ADP
        emb = emb.view((bs, -1))

        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
        self.optimizer_encoder.zero_grad()
        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        # emb = self.encoder(x.to(device)).view((bs, -1)) #CFR
        emb = self.encoder_withouchannel(a.to(device)).view((bs, -1)) #ADP

        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat


class DS_AE_manmade_DCNNclassification(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(DS_AE_manmade_DCNNclassification, self).__init__()
        self.hparams = hparams
        self.num_domain = 500# 6 选定有多少个domain
        self.encoder = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
                                     timm.create_model('resnet18', pretrained=False, num_classes=146))
        # self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))#输入通道是2
        # self.encoder = nn.Sequential(torchvision.models.resnet18(pretrained=False))  # 输入通道是3
        # self.encoder = nn.Sequential(torchvision.models.alexnet())
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )#4096
        self.encoder_withouchannel = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                                           timm.create_model('resnet18', pretrained=False, num_classes=146) # 输入通道是1
        )
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder = StepLR(self.optimizer_encoder, step_size=85, gamma=0.3)
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder_withouchannel = StepLR(self.optimizer_encoder_withouchannel, step_size=85, gamma=0.3)
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_decoder = StepLR(self.optimizer_decoder, step_size=85, gamma=0.3)
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_proj = StepLR(self.optimizer_proj, step_size=85, gamma=0.3)
        self.optimizer_clf_list = []
        self.scheduler_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            self.scheduler_clf_list.append(StepLR(self.optimizer_clf_list[i], step_size=85, gamma=0.3))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    # def update(self, z, x, y, a, device):  # x:channel; y:(x,y);
    #
    #     bs = x.shape[0]
    #     emb = self.encoder(x.to(device))
    #     emb = emb.view((bs, -1))
    #     emb_2 = self.encoder_withouchannel(a.to(device))
    #     emb = torch.concat([emb,emb_2], dim = 1)
    #     # print(emb.shape)
    #     x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
    #     emb = emb.view((bs, -1))
    #     # print("emb",emb.shape)
    #     y_hat = torch.zeros(bs, 2)
    #
    #     ############################
    #     #  y_hat 的逻辑改一下
    #     # y_hat = self.clf(emb)
    #     for i in range(bs):
    #         y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
    #         # y_hat_temp = self.clf_list[0](emb[i])
    #         if i == 0:
    #             y_hat = y_hat_temp.reshape(1,2)
    #         else:
    #             y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
    #     ############################1,2
    #     loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
    #     self.optimizer_encoder.zero_grad()
    #     self.optimizer_encoder_withouchannel.zero_grad()
    #     self.optimizer_decoder.zero_grad()
    #     self.optimizer_proj.zero_grad()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].zero_grad()
    #     loss.backward()
    #     self.optimizer_encoder.step()
    #     self.optimizer_encoder_withouchannel.step()
    #     self.optimizer_decoder.step()
    #     self.optimizer_proj.step()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].step()
    #     result = {"loss": ((y - y_hat) ** 2).mean()}
    #     return result

    def update(self, z, x, y, a, epoch, device):  # x:channel; y:(x,y);
        # decay = 0.5
        # if epoch == 85:# 每迭代20次，更新一次学习率
        #     for params in self.optimizer_encoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_encoder_withouchannel.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_decoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_proj.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for i in range(self.num_domain):
        #         for params in self.optimizer_clf_list[i].param_groups:
        #             # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #             params['lr'] *= decay

        bs = x.shape[0]
        # sigma = self.encoder(x.to(device)) #CDF
        sigma = self.encoder_withouchannel(a.to(device)) #ADP
        sigma = sigma.view((bs, -1))
        criterion = torch.nn.CrossEntropyLoss().to(device)
        loss = criterion(sigma, z)
        labels_hat = torch.max(sigma, 1)[1]
        correct = torch.sum(z == labels_hat)


        ############################1,2
        self.optimizer_encoder.zero_grad()
        self.optimizer_encoder_withouchannel.zero_grad()
        # self.optimizer_decoder.zero_grad()
        # self.optimizer_proj.zero_grad()
        # for i in range(self.num_domain):
        #     self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_encoder_withouchannel.step()
        # self.optimizer_decoder.step()
        # self.optimizer_proj.step()
        # for i in range(self.num_domain):
        #     self.optimizer_clf_list[i].step()
        result = correct
        # result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        # sigma = self.encoder(x.to(device)).view((bs, -1)) #CFR
        sigma = self.encoder_withouchannel(a.to(device)).view((bs, -1)) #ADP
        labels_hat = torch.max(sigma, 1)[1]
        correct = torch.sum(z == labels_hat)

        return correct


#CFR和ADP结合成一张输入图片
class DS_AE_combineCFRADP(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(DS_AE_combineCFRADP, self).__init__()
        self.hparams = hparams
        self.num_domain = 60# 6 选定有多少个domain
        # self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))#输入通道是2
        self.encoder = nn.Sequential(torchvision.models.resnet18(pretrained=False))  # 输入通道是3
        # self.encoder = nn.Sequential(torchvision.models.alexnet())
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )#4096
        # self.encoder_withouchannel = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
        #                                    torchvision.models.resnet18(pretrained = False)) # 输入通道是1#输出4096
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder = StepLR(self.optimizer_encoder, step_size=85, gamma=0.3)
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_decoder = StepLR(self.optimizer_decoder, step_size=85, gamma=0.3)
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_proj = StepLR(self.optimizer_proj, step_size=85, gamma=0.3)
        self.optimizer_clf_list = []
        self.scheduler_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            self.scheduler_clf_list.append(StepLR(self.optimizer_clf_list[i], step_size=85, gamma=0.3))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    # def update(self, z, x, y, a, device):  # x:channel; y:(x,y);
    #
    #     bs = x.shape[0]
    #     emb = self.encoder(x.to(device))
    #     emb = emb.view((bs, -1))
    #     emb_2 = self.encoder_withouchannel(a.to(device))
    #     emb = torch.concat([emb,emb_2], dim = 1)
    #     # print(emb.shape)
    #     x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
    #     emb = emb.view((bs, -1))
    #     # print("emb",emb.shape)
    #     y_hat = torch.zeros(bs, 2)
    #
    #     ############################
    #     #  y_hat 的逻辑改一下
    #     # y_hat = self.clf(emb)
    #     for i in range(bs):
    #         y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
    #         # y_hat_temp = self.clf_list[0](emb[i])
    #         if i == 0:
    #             y_hat = y_hat_temp.reshape(1,2)
    #         else:
    #             y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
    #     ############################1,2
    #     loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
    #     self.optimizer_encoder.zero_grad()
    #     self.optimizer_encoder_withouchannel.zero_grad()
    #     self.optimizer_decoder.zero_grad()
    #     self.optimizer_proj.zero_grad()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].zero_grad()
    #     loss.backward()
    #     self.optimizer_encoder.step()
    #     self.optimizer_encoder_withouchannel.step()
    #     self.optimizer_decoder.step()
    #     self.optimizer_proj.step()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].step()
    #     result = {"loss": ((y - y_hat) ** 2).mean()}
    #     return result

    def update(self, z, x, y, epoch, device):  # x:channel; y:(x,y);
        # decay = 0.5
        # if epoch == 85:# 每迭代20次，更新一次学习率
        #     for params in self.optimizer_encoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_encoder_withouchannel.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_decoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_proj.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for i in range(self.num_domain):
        #         for params in self.optimizer_clf_list[i].param_groups:
        #             # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #             params['lr'] *= decay

        bs = x.shape[0]
        emb = self.encoder(x.to(device))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
        self.optimizer_encoder.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, device):
        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1))
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat


class effiformerv2_DS_AE_CFRADP(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(effiformerv2_DS_AE_CFRADP, self).__init__()
        self.hparams = hparams
        self.num_domain = 500# 6 选定有多少个domain
        # self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))#输入通道是2
        # self.encoder = nn.Sequential(torchvision.models.resnet18(pretrained=False))  # 输入通道是3
        # self.encoder = nn.Sequential(torchvision.models.alexnet())
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )#4096
        self.encoder = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
                                     creat('efficientformerv2_l', pretrained=True, num_classes=100))  # 输入通道是2
        # self.encoder = creat('vit_base_patch16_224', pretrained=True, num_classes=1000)
        # self.encoder_withouchannel = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
        #                                    torchvision.models.resnet18(pretrained = False) # 输入通道是1
        # )#输出4096
        # self.encoder_withouchannel = creat('vit_base_patch16_224', pretrained=True, num_classes=1000)
        self.encoder_withouchannel = nn.Sequential(nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1),
                                                   creat('efficientformerv2_l', pretrained=True,
                                                         num_classes=1000)) # 输入通道是1
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(100 + 1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder = StepLR(self.optimizer_encoder, step_size=85, gamma=0.3)
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder_withouchannel = StepLR(self.optimizer_encoder_withouchannel, step_size=85, gamma=0.3)
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_decoder = StepLR(self.optimizer_decoder, step_size=85, gamma=0.3)
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_proj = StepLR(self.optimizer_proj, step_size=85, gamma=0.3)
        self.optimizer_clf_list = []
        self.scheduler_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            self.scheduler_clf_list.append(StepLR(self.optimizer_clf_list[i], step_size=85, gamma=0.3))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    # def update(self, z, x, y, a, device):  # x:channel; y:(x,y);
    #
    #     bs = x.shape[0]
    #     emb = self.encoder(x.to(device))
    #     emb = emb.view((bs, -1))
    #     emb_2 = self.encoder_withouchannel(a.to(device))
    #     emb = torch.concat([emb,emb_2], dim = 1)
    #     # print(emb.shape)
    #     x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
    #     emb = emb.view((bs, -1))
    #     # print("emb",emb.shape)
    #     y_hat = torch.zeros(bs, 2)
    #
    #     ############################
    #     #  y_hat 的逻辑改一下
    #     # y_hat = self.clf(emb)
    #     for i in range(bs):
    #         y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
    #         # y_hat_temp = self.clf_list[0](emb[i])
    #         if i == 0:
    #             y_hat = y_hat_temp.reshape(1,2)
    #         else:
    #             y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
    #     ############################1,2
    #     loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
    #     self.optimizer_encoder.zero_grad()
    #     self.optimizer_encoder_withouchannel.zero_grad()
    #     self.optimizer_decoder.zero_grad()
    #     self.optimizer_proj.zero_grad()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].zero_grad()
    #     loss.backward()
    #     self.optimizer_encoder.step()
    #     self.optimizer_encoder_withouchannel.step()
    #     self.optimizer_decoder.step()
    #     self.optimizer_proj.step()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].step()
    #     result = {"loss": ((y - y_hat) ** 2).mean()}
    #     return result

    def update(self, z, x, y, a, epoch, device):  # x:channel; y:(x,y);
        # decay = 0.5
        # if epoch == 85:# 每迭代20次，更新一次学习率
        #     for params in self.optimizer_encoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_encoder_withouchannel.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_decoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_proj.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for i in range(self.num_domain):
        #         for params in self.optimizer_clf_list[i].param_groups:
        #             # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #             params['lr'] *= decay

        bs = x.shape[0]
        emb = self.encoder(x.to(device))
        emb = emb.view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb,emb_2], dim = 1)
        # print(emb.shape)
        # x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
        self.optimizer_encoder.zero_grad()
        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb, emb_2], dim=1)
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat


class effiformerv2_DS_AE_CFRperfectADP(nn.Module):
    '''
    Domain-Specific Auto-Encoder
    '''

    def __init__(self, hparams, device):  # (channel, z, UE_x, UE_y)
        super(effiformerv2_DS_AE_CFRperfectADP, self).__init__()
        self.hparams = hparams
        self.num_domain = 120# 6 选定有多少个domain
        # self.encoder = nn.Sequential(nn.Conv2d(2,3,kernel_size=3,stride=1,padding=1), torchvision.models.resnet18(pretrained=False))#输入通道是2
        self.encoder = nn.Sequential(nn.Conv2d(2, 3, kernel_size=3, stride=1, padding=1),
                                     creat('efficientformerv2_l', pretrained=True, num_classes=1000))  # 输入通道是2
        # self.encoder = nn.Sequential(torchvision.models.resnet18(pretrained=False))  # 输入通道是3
        # self.encoder = nn.Sequential(torchvision.models.alexnet())
        # self.encoder = nn.Sequential(
        #     nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        #     nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(),
        # )#4096
        self.encoder_withouchannel = nn.Sequential(
            torch.nn.Linear(4, 64),
            nn.ReLU(),
            torch.nn.Linear(64, 512),
            nn.ReLU(),
            torch.nn.Linear(512, 1000)
        )#输出4096
        #暂时不知道选用什么结构进行逆推
        self.proj = torch.nn.Linear(1000 + 1000, 7 * 7 * 64)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # output_padding = (kernel_size-1)/2
            nn.BatchNorm2d(32),  # out_size = (in_size-1)*stride - 2 * padding + kernel_size + output_padding
            nn.ReLU(),  # 6*2-2+3+1=14
            nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 28
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=16, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 56
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=8, kernel_size=3, stride=2,
                               padding=1, output_padding=1),  # 112
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.ConvTranspose2d(in_channels=8, out_channels=2, kernel_size=3, stride=2,
                               padding=1, output_padding=1)  # 224
        )
        self.clf_list = []
        for i in range(self.num_domain):
            self.clf_list.append(torch.nn.Linear(1000 + 1000, 2))#线性
        #     self.clf_list.append(nn.Sequential(
        #     torch.nn.Linear(1000 + 1000, 512),
        #     nn.ReLU(),
        #     torch.nn.Linear(512, 64),
        #     nn.ReLU(),
        #     torch.nn.Linear(64, 2)
        # ))
            self.clf_list[i] = self.clf_list[i].to(device)
        m = 0.95
        self.optimizer_encoder = torch.optim.Adam(
            self.encoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder = StepLR(self.optimizer_encoder, step_size=85, gamma=0.3)
        self.optimizer_encoder_withouchannel = torch.optim.Adam(
            self.encoder_withouchannel.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_encoder_withouchannel = StepLR(self.optimizer_encoder_withouchannel, step_size=85, gamma=0.3)
        self.optimizer_decoder = torch.optim.Adam(
            self.decoder.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_decoder = StepLR(self.optimizer_decoder, step_size=85, gamma=0.3)
        self.optimizer_proj = torch.optim.Adam(
            self.proj.parameters()
            , lr=self.hparams["lr"]
        )
        self.scheduler_proj = StepLR(self.optimizer_proj, step_size=85, gamma=0.3)
        self.optimizer_clf_list = []
        self.scheduler_clf_list = []
        for i in range(self.num_domain):
            self.optimizer_clf_list.append(torch.optim.Adam(
                self.clf_list[i].parameters()
                , lr=self.hparams["lr"]
            ))
            self.scheduler_clf_list.append(StepLR(self.optimizer_clf_list[i], step_size=85, gamma=0.3))
            # self.optimizer_clf_list[i] = self.optimizer_clf_list[i].to(device)

    # def __info__(self):
    #     print("\t Number of samples: {}".format(len(self.image_path)))

    # def update(self, z, x, y, a, device):  # x:channel; y:(x,y);
    #
    #     bs = x.shape[0]
    #     emb = self.encoder(x.to(device))
    #     emb = emb.view((bs, -1))
    #     emb_2 = self.encoder_withouchannel(a.to(device))
    #     emb = torch.concat([emb,emb_2], dim = 1)
    #     # print(emb.shape)
    #     x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
    #     emb = emb.view((bs, -1))
    #     # print("emb",emb.shape)
    #     y_hat = torch.zeros(bs, 2)
    #
    #     ############################
    #     #  y_hat 的逻辑改一下
    #     # y_hat = self.clf(emb)
    #     for i in range(bs):
    #         y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
    #         # y_hat_temp = self.clf_list[0](emb[i])
    #         if i == 0:
    #             y_hat = y_hat_temp.reshape(1,2)
    #         else:
    #             y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
    #     ############################1,2
    #     loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
    #     self.optimizer_encoder.zero_grad()
    #     self.optimizer_encoder_withouchannel.zero_grad()
    #     self.optimizer_decoder.zero_grad()
    #     self.optimizer_proj.zero_grad()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].zero_grad()
    #     loss.backward()
    #     self.optimizer_encoder.step()
    #     self.optimizer_encoder_withouchannel.step()
    #     self.optimizer_decoder.step()
    #     self.optimizer_proj.step()
    #     for i in range(self.num_domain):
    #         self.optimizer_clf_list[i].step()
    #     result = {"loss": ((y - y_hat) ** 2).mean()}
    #     return result

    def update(self, z, x, y, a, epoch, device):  # x:channel; y:(x,y);
        # decay = 0.5
        # if epoch == 85:# 每迭代20次，更新一次学习率
        #     for params in self.optimizer_encoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_encoder_withouchannel.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_decoder.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for params in self.optimizer_proj.param_groups:
        #         # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #         params['lr'] *= decay
        #     for i in range(self.num_domain):
        #         for params in self.optimizer_clf_list[i].param_groups:
        #             # 遍历Optimizer中的每一组参数，将该组参数的学习率 * decay
        #             params['lr'] *= decay

        bs = x.shape[0]
        emb = self.encoder(x.to(device))
        emb = emb.view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb,emb_2], dim = 1)
        # print(emb.shape)
        x_hat = self.decoder(self.proj(emb).reshape(bs,64,7,7))
        emb = emb.view((bs, -1))
        # print("emb",emb.shape)
        y_hat = torch.zeros(bs, 2)

        ############################
        #  y_hat 的逻辑改一下
        # y_hat = self.clf(emb)
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1,2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1,2)],dim = 0)
        ############################1,2
        loss = ((y - y_hat) ** 2).mean() #+ ((x - x_hat) ** 2).mean()  # ?
        self.optimizer_encoder.zero_grad()
        self.optimizer_encoder_withouchannel.zero_grad()
        self.optimizer_decoder.zero_grad()
        self.optimizer_proj.zero_grad()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].zero_grad()
        loss.backward()
        self.optimizer_encoder.step()
        self.optimizer_encoder_withouchannel.step()
        self.optimizer_decoder.step()
        self.optimizer_proj.step()
        for i in range(self.num_domain):
            self.optimizer_clf_list[i].step()
        result = {"loss": ((y - y_hat) ** 2).mean()}
        return result

    def predict(self, z, x, a, device):
        bs = x.shape[0]
        emb = self.encoder(x.to(device)).view((bs, -1))
        emb_2 = self.encoder_withouchannel(a.to(device))
        emb = torch.concat([emb, emb_2], dim=1)
        y_hat = torch.zeros(bs, 2)
        ############################
        #  y_hat 的逻辑改一下
        for i in range(bs):
            y_hat_temp = self.clf_list[z[i]](emb[i])#这里要更改，如果是CNN就只需要1个就行
            # y_hat_temp = self.clf_list[0](emb[i])
            if i == 0:
                y_hat = y_hat_temp.reshape(1, 2)
            else:
                y_hat = torch.concat([y_hat, y_hat_temp.reshape(1, 2)], dim=0)
        # for zidx in z:
        #     if zidx >= 4 and zidx <= 6:
        #         zidx = zidx - 1
        #     y_hat[zidx] = self.clf_list[zidx](emb)
        ############################

        return y_hat



# if __name__ == '__main__':
#     # ds-ae-com
#     lr = 0.001
#     hparams = {'lr': lr,
#                'temperature': 0.75}
#     cudaIdx = "cuda:2"  # GPU card index
#     device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")
#     model = DS_AE_CFRADP(hparams, device).to(device, non_blocking=True)  # 多头+cnn
#     LR = 0.01
#     optimizer = Adam(model.parameters(),lr = LR)
#     lr_list = []
#     for epoch in range(100):
#         if epoch % 5 == 0:
#             for p in optimizer.param_groups:
#                 p['lr'] *= 0.9#注意这里
#         lr_list.append(optimizer.state_dict()['param_groups'][0]['lr'])
#     plt.plot(range(100),lr_list,color = 'r')