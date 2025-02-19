'''
Copyright (C) 2021. Huawei Technologies Co., Ltd.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
'''

import os
import torch
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import torchvision
from torchvision import transforms
from matplotlib import pyplot as plt
import copy
import pandas as pd
from openpyxl import load_workbook


class Recoder(object):#在python 3 中已经默认就帮你加载了object了（即便你没有写上object）
    def __init__(self):
        self.last = 0
        self.values = []
        self.nums = []

    def update(self, val, n=1):#更新值和对应的索引
        self.last = val
        self.values.append(val)
        self.nums.append(n)

    def avg(self):
        sum = np.sum(np.asarray(self.values) * np.asarray(self.nums))
        count = np.sum(np.asarray(self.nums))
        return sum / count

    def var(self):
        return np.var(self.values)


def seed_everything(seed: int):
    # random.seed(seed)
    # np.random.seed(seed)
    # torch.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False

def toNP(tensor):
    return tensor.detach().cpu().numpy()#detach阻断反向传播, 返回值仍为tensor; cpu()将变量放在cpu上，仍为tensor; 将tensor转换为numpy


def makeDIRs(folder):  # 创建models/runCase_1和results/runCase_1这两个文件夹（包含子文件夹
    if not os.path.exists(f'models/{folder}/'):
        os.makedirs(f'models/{folder}/')
    if not os.path.exists(f'results/{folder}/'):
        os.makedirs(f'results/{folder}/')


def checkPoint(runCase, epoch, epochs, model, Train, Valid, saveModelInterval,
               saveLossInterval):  # 保存每一次epoch的模型结构权重数值s
    if epoch % saveModelInterval == 0 or epoch == epochs:
        torch.save(model.state_dict(), f'models/{runCase}/DoraNet_' + str(epoch) + '.pth')
    if epoch % saveLossInterval == 0 or epoch == epochs:
        np.save(f'results/{runCase}/Train.npy', np.asarray(Train))
        np.save(f'results/{runCase}/Valid.npy', np.asarray(Valid))


def readEnvLink(folder, carrierFreq, imageSize, scenario): #?
    transform = transforms.Compose([transforms.Grayscale(),#将图像转换为灰度图像，默认通道数为1，通道数为3时，RGB三个通道的值相等
                                    transforms.ToTensor(),
                                    ])#将图像转换为灰度图像, 然后转换成张量
    sub_path = folder
    linklocA_path = os.path.join(sub_path, 'Info.npy')
    linklocA = np.load(linklocA_path, allow_pickle=True, encoding='latin1')
    linklocA[:2] = np.floor(linklocA[:2])#用来标正图片？意义在哪里没看明白
    env_path = os.path.join(sub_path, 'environment.png')
    env = Image.open(env_path, mode='r')
    ImgSize = linklocA[:2].astype(np.int64)
    if ImgSize[0] > ImgSize[1]:
        top = -(ImgSize[0] - ImgSize[1]) // 2
        left = 0
    else:
        left = -(ImgSize[1] - ImgSize[0]) // 2
        top = 0
    env = env.resize(ImgSize)#按照Info.npy第一、第二个数调整图片大小
    envNew = Image.new(env.mode, (np.max(ImgSize), np.max(ImgSize)), (255, 255, 255))#创建新图片
    envNew.paste(env, (int(-left), int(-top)))
    env.close()
    envNew = 1 - transform(envNew)#?
    if torch.max(envNew) == 0:
        pass
    else:
        envNew = envNew / torch.max(envNew)
    envNew = transforms.functional.resize(envNew, imageSize)#*将envNew变成imagesize的尺寸
    linklocA[2::2] = (linklocA[2::2] - left) / np.max(ImgSize)
    linklocA[3::2] = (linklocA[3::2] - (ImgSize[1] - top - np.max(ImgSize))) / np.max(ImgSize)
    P = np.load(os.path.join(sub_path, 'Path.npy'), allow_pickle=True, encoding='latin1')
    Pitem = P.item()
    H = np.load(os.path.join(sub_path, f'H_{carrierFreq}_G.npy'), allow_pickle=True, encoding='latin1')
    Hitem = H.item()
    sights = []
    distances = []
    gains = []
    angles = []
    if scenario == 1:
        bsMax = 5
        ueMax = 30
    else:
        bsMax = 1
        ueMax = 10000
    for bsIdx in range(bsMax):
        for ueIdx in range(ueMax):
            Plink = Pitem[f'bs{bsIdx}_ue{ueIdx}'] if scenario == 1 else Pitem[f'bs{bsIdx}_ue{ueIdx:05d}']
            Hlink = Hitem[f'bs{bsIdx}_ue{ueIdx}'] if scenario == 1 else Hitem[f'bs{bsIdx}_ue{ueIdx:05d}']
            tau = Plink['taud']
            firstPath = np.argmin(tau)#最小值对应的索引，也就是最早到达的路径对应的索引
            doa = Plink['doa']
            dod = Plink['dod']
            phiDiff = dod[firstPath, 1] - doa[firstPath, 1]
            if scenario == 1:
                BSloc = linklocA[2:].reshape(150, 4)[bsIdx * 30, :2]
                UEloc = linklocA[2:].reshape(150, 4)[ueIdx, 2:4]
            else:
                BSloc = linklocA[2:4]
                UEloc = linklocA[4:].reshape(10000, 2)[ueIdx, :]
            dis1 = ((UEloc[1] - BSloc[1]) ** 2 + (UEloc[0] - BSloc[0]) ** 2 + (
                        (6 - 1.5) * 2 / np.max(ImgSize)) ** 2) ** 0.5 * 0.5 * np.max(ImgSize)
            dis2 = np.min(tau) * 0.3 #?
            distances.append(dis2 * 2 / np.max(ImgSize))
            if np.round(phiDiff, 5) == np.round(np.pi, 5) and np.round(dis1, 5) == np.round(dis2, 5):
                sight = 1  # line of sight
            else:
                sight = 0  # non line of sight
            gains.append(10 * np.log10(np.sum(np.abs(Hlink) ** 2)))
            sights.append(sight)
            angles.append([dod[firstPath, 1], doa[firstPath, 1]])#选出第一条到达路径的2d平面的到达角
    return envNew, linklocA, np.asarray(sights), np.asarray(distances), np.asarray(angles), np.asarray(gains)

def readEnvLink_cases(folder, path, carrierFreq, imageSize, scenario, casesToTrain):#在readEnvLink的基础之上，筛选出满足caseToTrain的数据
    transform = transforms.Compose([transforms.Grayscale(),#将图像转换为灰度图像，默认通道数为1，通道数为3时，RGB三个通道的值相等
                                    transforms.ToTensor(),
                                    ])#将图像转换为灰度图像, 然后转换成张量
    sub_path = os.path.join(folder, path)
    linklocA_path = os.path.join(sub_path, 'Info.npy')
    linklocA = np.load(linklocA_path, allow_pickle=True, encoding='latin1')
    linklocA[:2] = np.floor(linklocA[:2])#用来标正图片？意义在哪里没看明白
    env_path = os.path.join(sub_path, 'environment.png')
    env = Image.open(env_path, mode='r')
    ImgSize = linklocA[:2].astype(np.int64)
    if ImgSize[0] > ImgSize[1]:
        top = -(ImgSize[0] - ImgSize[1]) // 2
        left = 0
    else:
        left = -(ImgSize[1] - ImgSize[0]) // 2
        top = 0
    env = env.resize(ImgSize)#按照Info.npy第一、第二个数调整图片大小
    envNew = Image.new(env.mode, (np.max(ImgSize), np.max(ImgSize)), (255, 255, 255))#创建新图片
    envNew.paste(env, (int(-left), int(-top)))
    env.close()
    envNew = 1 - transform(envNew)#?
    if torch.max(envNew) == 0:
        pass
    else:
        envNew = envNew / torch.max(envNew)
    envNew = transforms.functional.resize(envNew, imageSize)#*将envNew变成imagesize的尺寸
    linklocA[2::2] = (linklocA[2::2] - left) / np.max(ImgSize)
    linklocA[3::2] = (linklocA[3::2] - (ImgSize[1] - top - np.max(ImgSize))) / np.max(ImgSize)
    P = np.load(os.path.join(sub_path, 'Path.npy'), allow_pickle=True, encoding='latin1')
    Pitem = P.item()
    H = np.load(os.path.join(sub_path, f'H_{carrierFreq}_G.npy'), allow_pickle=True, encoding='latin1')
    Hitem = H.item()
    sights = []
    distances = []
    gains = []
    angles = []
    if scenario == 1:
        bsMax = 5
        ueMax = 30
    else:
        bsMax = 1
        ueMax = 10000
    for bsIdx in range(bsMax):
        for ueIdx in range(ueMax):
            Plink = Pitem[f'bs{bsIdx}_ue{ueIdx}'] if scenario == 1 else Pitem[f'bs{bsIdx}_ue{ueIdx:05d}']
            Hlink = Hitem[f'bs{bsIdx}_ue{ueIdx}'] if scenario == 1 else Hitem[f'bs{bsIdx}_ue{ueIdx:05d}']
            tau = Plink['taud']
            firstPath = np.argmin(tau)#最小值对应的索引，也就是最早到达的路径对应的索引
            doa = Plink['doa']
            dod = Plink['dod']
            phiDiff = dod[firstPath, 1] - doa[firstPath, 1]
            if scenario == 1:
                BSloc = linklocA[2:].reshape(150, 4)[bsIdx * 30, :2]
                UEloc = linklocA[2:].reshape(150, 4)[ueIdx, 2:4]
            else:
                BSloc = linklocA[2:4]
                UEloc = linklocA[4:].reshape(10000, 2)[ueIdx, :]
            dis1 = ((UEloc[1] - BSloc[1]) ** 2 + (UEloc[0] - BSloc[0]) ** 2 + (
                        (6 - 1.5) * 2 / np.max(ImgSize)) ** 2) ** 0.5 * 0.5 * np.max(ImgSize)
            dis2 = np.min(tau) * 0.3 #?
            if np.round(phiDiff, 5) == np.round(np.pi, 5) and np.round(dis1, 5) == np.round(dis2, 5):
                sight = 1  # line of sight
            else:
                sight = 0  # non line of sight
            ifcases = (sight == casesToTrain) if casesToTrain > -1 else (sight > casesToTrain)
            if not ifcases:
                break
            distances.append(dis2 * 2 / np.max(ImgSize))
            gains.append(10 * np.log10(np.sum(np.abs(Hlink) ** 2)))
            sights.append(sight)
            angles.append([dod[firstPath, 1], doa[firstPath, 1]])
    return envNew, linklocA, np.asarray(sights), np.asarray(distances), np.asarray(angles), np.asarray(gains)

def readChannel(readImage, path, bsNo, ueNo, scenario):#提取norm_channel
    if readImage:
        channelPath = path + 'bs' + str(bsNo) + '_' + (
            f'ue{ueNo}.png' if scenario == 1 else f'ue{ueNo:05d}.png')
        H = Image.open(channelPath)
        H = np.asarray(H, dtype=float)#(64, 64, 2)
        # H /= 255.
        H = (H[:, :, 0] / 255. + 1j * H[:, :, 1] / 255. - 0.5 - 1j * 0.5) * 2#(64, 64)
    else:
        channelPath = f'{path}/array/bs{bsNo}_' + (
            f'ue{ueNo}.npy' if scenario == 1 else f'ue{ueNo:05d}.npy')
        H = np.load(channelPath, allow_pickle=True)
        H = H[0, :, :] + 1j * H[1, :, :]
    return H#(64,64) 64值CFR，高天线对，宽sample_frequency

def readChannel_firstArrive(readImage, path, curEnv, bsNo, ueNo, scenario):
    '''32*2(Nr*Nt), channel at frequency 28'''
    if readImage:
        channelPath = f'{path}/{curEnv}/image/bs{bsNo}_' + (
            f'ue{ueNo}_2.png' if scenario == 1 else f'ue{ueNo:05d}.png')
        # channel2 process
    else:
        channelPath = f'{path}/{curEnv}/array/bs{bsNo}_' + (
            f'ue{ueNo}_2.npy' if scenario == 1 else f'ue{ueNo:05d_2}.npy')
        H1 = np.load(channelPath, allow_pickle=True)
        H2 = np.concatenate((np.real(H1)[None, ...], np.imag(H1)[None, ...]), axis=0)
        #draw one of the figures
        # plt.title('H_28.0sample')
        # plt.imshow(array3)
        # plt.axis("off")
        # plt.savefig('H_28.0sample.png')
    return H1, H2

def readChannel_CE(curEnv, bsNo):
    linkStr = f'curEnv{curEnv}_bs{bsNo}'
    path = os.path.join('data', linkStr + '.npy')
    H = np.load(path, allow_pickle=True)#32*60
    return H

def Y_sgn(readImage, path, curEnv, bsNo, ueNo, scenario):
    channel21, channel22 = readChannel_firstArrive(readImage, generatedFolder, self.paths[curEnv], BSlist[bsIdx],
                                                   UElist[ueIdx], scenario)

    dftmtx = np.fft.fft(np.eye(channel21))


def antennaPosition(N, spacing, Basis):
    N0, N1, N2 = N
    p0 = spacing[0] * np.linspace(-(N0 - 1) * 0.5, (N0 - 1) * 0.5, N0)[None, :] * Basis[:, 0:1]
    p1 = spacing[1] * np.linspace(-(N1 - 1) * 0.5, (N1 - 1) * 0.5, N1)[None, :] * Basis[:, 1:2]
    p2 = spacing[2] * np.linspace(-(N2 - 1) * 0.5, (N2 - 1) * 0.5, N2)[None, :] * Basis[:, 2:3]
    p = p0[:, :, None, None] + p1[:, None, :, None] + p2[:, None, None, :]#?
    position = p.reshape((3, np.prod(N)))#行是xyz，列是多少个天线
    return position


def arrayResponse(angle, position, sortedPath):#?
    rx = np.sin(angle[sortedPath, 0]) * np.cos(angle[sortedPath, 1])#sin有点问题，不应该是cos？
    ry = np.sin(angle[sortedPath, 0]) * np.sin(angle[sortedPath, 1])
    rz = np.cos(angle[sortedPath, 0])
    r = np.concatenate((rx[:, None], ry[:, None], rz[:, None]), axis=-1)
    r = np.dot(r, position)#?
    response = np.exp(1j * 2 * np.pi * r)
    # print(response)
    return response


def backward(optimizer, loss):
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()




def readEnvLink_1(folder, path, carrierFreq, imageSize, scenario):
    transform = transforms.Compose([transforms.Grayscale(),#将图像转换为灰度图像，默认通道数为1，通道数为3时，RGB三个通道的值相等
                                    transforms.ToTensor(),
                                    ])#将图像转换为灰度图像, 然后转换成张量
    sub_path = os.path.join(folder, path)
    linklocA_path = os.path.join(sub_path, 'Info.npy')
    linklocA = np.load(linklocA_path, allow_pickle=True, encoding='latin1')
    linklocA[:2] = np.floor(linklocA[:2])#用来标正图片？意义在哪里没看明白
    env_path = os.path.join(sub_path, 'environment.png')
    env = Image.open(env_path, mode='r')
    ImgSize = linklocA[:2].astype(np.int64)
    if ImgSize[0] > ImgSize[1]:
        top = -(ImgSize[0] - ImgSize[1]) // 2
        left = 0
    else:
        left = -(ImgSize[1] - ImgSize[0]) // 2
        top = 0
    env = env.resize(ImgSize)#按照Info.npy第一、第二个数调整图片大小
    envNew = Image.new(env.mode, (np.max(ImgSize), np.max(ImgSize)), (255, 255, 255))#创建新图片
    envNew.paste(env, (int(-left), int(-top)))
    env.close()
    envNew = 1 - transform(envNew)#?
    if torch.max(envNew) == 0:
        pass
    else:
        envNew = envNew / torch.max(envNew)
    envNew = transforms.functional.resize(envNew, imageSize)#*将envNew变成imagesize的尺寸
    linklocA[2::2] = (linklocA[2::2] - left) / np.max(ImgSize)
    linklocA[3::2] = (linklocA[3::2] - (ImgSize[1] - top - np.max(ImgSize))) / np.max(ImgSize)
    P = np.load(os.path.join(sub_path, 'Path.npy'), allow_pickle=True, encoding='latin1')
    Pitem = P.item()
    H = np.load(os.path.join(sub_path, f'H_{carrierFreq}_G.npy'), allow_pickle=True, encoding='latin1')
    Hitem = H.item()
    sights = []
    distances = []
    gains = []
    angles = []
    if scenario == 1:
        bsMax = 5
        ueMax = 30
    else:
        bsMax = 1
        ueMax = 10000
    for bsIdx in range(bsMax):
        for ueIdx in range(ueMax):
            Plink = Pitem[f'bs{bsIdx}_ue{ueIdx}'] if scenario == 1 else Pitem[f'bs{bsIdx}_ue{ueIdx:05d}']
            Hlink = Hitem[f'bs{bsIdx}_ue{ueIdx}'] if scenario == 1 else Hitem[f'bs{bsIdx}_ue{ueIdx:05d}']
            tau = Plink['taud']
            firstPath = np.argmin(tau)#最小值对应的索引，也就是最早到达的路径对应的索引
            doa = Plink['doa']
            dod = Plink['dod']
            phiDiff = dod[firstPath, 1] - doa[firstPath, 1]
            if scenario == 1:
                BSloc = linklocA[2:].reshape(150, 4)[bsIdx * 30, :2]
                UEloc = linklocA[2:].reshape(150, 4)[ueIdx, 2:4]
            else:
                BSloc = linklocA[2:4]
                UEloc = linklocA[4:].reshape(10000, 2)[ueIdx, :]
            dis1 = ((UEloc[1] - BSloc[1]) ** 2 + (UEloc[0] - BSloc[0]) ** 2 + (
                        (6 - 1.5) * 2 / np.max(ImgSize)) ** 2) ** 0.5 * 0.5 * np.max(ImgSize)
            dis2 = np.min(tau) * 0.3 #?
            distances.append(dis2 * 2 / np.max(ImgSize))
            if np.round(phiDiff, 5) == np.round(np.pi, 5) and np.round(dis1, 5) == np.round(dis2, 5):
                sight = 1  # line of sight
            else:
                sight = 0  # non line of sight
            gains.append(10 * np.log10(np.sum(np.abs(Hlink) ** 2)))
            sights.append(sight)
            angles.append([dod[firstPath, 1], doa[firstPath, 1]])
    return envNew, linklocA, np.asarray(sights), np.asarray(distances), np.asarray(angles), np.asarray(gains)


def to_xls(A, Aname, sheetName):#Aname = 'A.xlsx', sheetName = 'page_1'
    data = pd.DataFrame(A)

    writer = pd.ExcelWriter(Aname)  # 写入Excel文件
    data.to_excel(writer, sheetName, float_format='%.5f')  # ‘page_1’是写入excel的sheet名
    writer.save()

    writer.close()
# 有些问题下面这个程序
# def to_xls(A, Aname, sheetName):#Aname = 'A.xlsx', sheetName = 'page_1'
#     data = pd.DataFrame(A)
#     book = load_workbook(Aname)
#     writer = pd.ExcelWriter(Aname, engine='openpyxl')  # 写入Excel文件
#     writer.book = book
#     data.to_excel(writer, sheetName, float_format='%.5f')  # ‘page_1’是写入excel的sheet名
#     writer.save()
#
#     writer.close()


def readChannel_2(readImage, path, curEnv, bsNo, ueNo, scenario):#提取norm_channel
    if readImage:
        channelPath = f'{path}/{curEnv}/image/bs{bsNo}_' + (
            f'ue{ueNo}.png' if scenario == 1 else f'ue{ueNo:05d}.png')
        H = Image.open(channelPath)
        H = np.asarray(H, dtype=float)#(64, 64, 2)
        H /= 255.
        # H = (H[:, :, 0] / 255. + 1j * H[:, :, 1] / 255. - 0.5 - 1j * 0.5) * 2#(64, 64)
    else:
        channelPath = f'{path}/{curEnv}/array/bs{bsNo}_' + (
            f'ue{ueNo}.npy' if scenario == 1 else f'ue{ueNo:05d}.npy')
        H = np.load(channelPath, allow_pickle=True)
        H = H[0, :, :] + 1j * H[1, :, :]
    return H#(64,64) 64值CFR，高天线对，宽sample_frequency


class EarlyStopping():
    """
    在测试集上的损失连续几个epochs不再降低的时候，提前停止
    val_loss: 测试集/验证集上这个epoch的损失

    """

    def __init__(self, patience=5, tol=0.0005):

        '''
        patience: 连续patience个epochs上损失不再降低的时候，停止迭代
        tol: 阈值，当新损失与旧损失之前的差异小于tol值时，认为模型不再提升
        '''
        self.patience = patience
        self.tol = tol
        self.counter = 0
        self.lowest_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.lowest_loss == None:
            self.lowest_loss = val_loss
        elif self.lowest_loss - val_loss > self.tol:
            self.lowest_loss = val_loss
            self.counter = 0
        elif self.lowest_loss - val_loss < self.tol:
            self.counter += 1
            print("\t NOTICE: Early stopping counter {} of {}".format(self.counter, self.patience))
            if self.counter >= self.patience:
                print('\t NOTICE: Early Stopping Actived')
                self.early_stop = True
        return self.early_stop


def loss_image(Train_Loss,Test_Loss,imgname):#results文件夹下面
    """Display the results of training and testing"""
    # G_Loss = torch.tensor(G_Loss, device='cpu')
    # D_Loss = torch.tensor(D_Loss, device='cpu')
    PATH = "results"
    # imgname = "NLOS_2_MLP"#NLOS_scenario_model
    if not os.path.exists(os.path.join(PATH, "loss_img")):
        os.makedirs(os.path.join(PATH, "loss_img"))#创建文件夹

    plt.figure(figsize=(10, 20))
    plt.title("Training loss and Test loss")
    plt.plot(Train_Loss, label="Train_Loss")#G
    plt.plot(Test_Loss, label="Test_Loss")#D
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(PATH, "loss_img", imgname + '.jpg'))
    plt.show()


def acc_image(Train_Loss,Test_Loss,imgname):#results文件夹下面
    """Display the results of training and testing"""
    # G_Loss = torch.tensor(G_Loss, device='cpu')
    # D_Loss = torch.tensor(D_Loss, device='cpu')
    PATH = "results"
    # imgname = "NLOS_2_MLP"#NLOS_scenario_model
    if not os.path.exists(os.path.join(PATH, "acc_img")):
        os.makedirs(os.path.join(PATH, "acc_img"))#创建文件夹

    plt.figure(figsize=(10, 20))
    plt.title("Training Acc and Test Acc")
    plt.plot(Train_Loss, label="Train_Acc")#G
    plt.plot(Test_Loss, label="Test_Acc")#D
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(os.path.join(PATH, "acc_img", imgname + '.jpg'))
    plt.show()

#让每个数据集随机显示5张图像
def plotsample(data):#只能够接受tensor格式的图像
    fig, axs = plt.subplots(1,5,figsize=(10,10)) #建立子图
    for i in range(5):
        num = random.randint(0,len(data)-1) #首先选取随机数，随机选取五次
        #抽取数据中对应的图像对象，make_grid函数可将任意格式的图像的通道数升为3，而不改变图像原始的数据
        #而展示图像用的imshow函数最常见的输入格式也是3通道
        npimg = torchvision.utils.make_grid(data[num][0]).numpy()
        nplabel = data[num][1] #提取标签
        #将图像由(3, weight, height)转化为(weight, height, 3)，并放入imshow函数中读取
        axs[i].imshow(np.transpose(npimg, (1, 2, 0)))
        # axs[i].imshow(npimg)
        axs[i].set_title(nplabel) #给每个子图加上标签
        axs[i].axis("off") #消除每个子图的坐标轴

#生成VF以生成后续的ADP
def generate_matrix(Nt_x, Nt_y, Nc):
    V_x = np.zeros((Nt_x, Nt_x), dtype=np.complex128)
    V_y = np.zeros((Nt_y, Nt_y), dtype=np.complex128)
    F = np.zeros((Nc, Nc), dtype=np.complex128)
    for z in range(Nt_x):
        for q in range(Nt_x):
#             V[z,q] = z + q * 1j
            V_x[z,q] = 1/np.sqrt(Nt_x) * np.exp(-2 * np.pi * 1j * (z * (q - Nt_x/2)) / Nt_x)
    for z in range(Nt_y):
        for q in range(Nt_y):
#             V[z,q] = z + q * 1j
            V_y[z,q] = 1/np.sqrt(Nt_y) * np.exp(-2 * np.pi * 1j * (z * (q - Nt_y/2)) / Nt_y)
    for z in range(Nc):
        for q in range(Nc):
            F[z,q] = 1/np.sqrt(Nc) * np.exp(-2 * np.pi * 1j * z * q / Nc)
    return V_x, V_y, F