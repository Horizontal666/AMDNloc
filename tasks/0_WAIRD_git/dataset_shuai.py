from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import torch, torchvision
from parameters_waird import *
import cv2
from torchvision import transforms
import matplotlib.pyplot as plt
import random  # 为了plotsamples而导入
import pandas as pd
from util_waird import *

# 确保plotsample已运行


# train_path = os.path.join('/workspace/wbh/DeepMIMO-5GNR/DeepMIMO_python/data/label', 'train_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv')
# test_path = os.path.join('/workspace/wbh/DeepMIMO-5GNR/DeepMIMO_python/data/label', 'test_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv')
target_path = "/workspace/wbh/DeepMIMO-5GNR/DeepMIMO_python/data/label/target.txt"
# root = "../../channel/withPhase" #改为path
#存放全部的target信息
imageSize = 256


class ImgDataset(Dataset):
    """
    自定义数据集，用于读取celebA数据集中的个体识别（identity recognition）数据的标签和图像
    图像格式为png
    """

    def __init__(self, root_dir, file, transform=None):
        """
        参数说明：
            csv_file (字符串): 标签csv/txt的具体地址
            root_dir (string): 所有图片所在的根目录
            transform (callable, optional): 选填，需要对样本进行的预处理
        """
        super().__init__()
        # 读取train文件，提取每一类别的具体数目，用以进行分类操作。
        df_train = pd.read_csv(train_path)
        # 打印image number in the class
        self.class_z = []
        self.class_num = []
        for i in df_train['final_combination'].unique():
            self.class_z.append(i)
            self.class_num.append(len(df_train[df_train['final_combination'] == i]))

        #txt读取方式
        # lines = open(file, 'r').readlines()
        # lines_target = open(target_path, 'r').readlines()
        #读取csv文件，从csv文件的标题下面开始读
        lines = np.loadtxt(file, delimiter=',', skiprows=1, dtype=str)
        # lines_target = np.loadtxt("/workspace/wbh/DeepMIMO-5GNR-localpycharm/data/label/target_00032_1500.csv", delimiter=',', skiprows=1, dtype=str)#00032的
        lines_target = np.loadtxt(template_result_file_finalCom_largerthan10, delimiter=',', skiprows=1, dtype=str)
        # lines_target = np.loadtxt(template_result_file_finalCom, delimiter=',', skiprows=1, dtype=str)
        # #生成用于ADP的V和F矩阵
        # N_antenna = Nt[0] * Nt[1] * Nt[2] * Nr[0] * Nr[1] * Nr[2] #64
        # self.V_x, self.V_y, self.F = generate_matrix(N_antenna, sampledCarriers)
        self.image_path = []
        self.labels = []
        self.ueloc_x = []
        self.ueloc_y = []
        self.ueloc_x_target = []
        self.ueloc_y_target = []
        self.distance = []
        self.pathloss = []
        self.distance_target = []
        self.pathloss_target = []
        self.z = []
        self.z_CFR = []
        self.z_ADP = []
        self.z_CFR_manmadearea = []
        self.DoD_phi = []
        self.DoD_theta = []
        self.ToA = []
        self.phase = []
        self.power = []
        self.DoD_phi_target = []
        self.DoD_theta_target = []
        self.ToA_target = []
        self.phase_target = []
        self.power_target = []
        self.imageSize = imageSize
        # for line_target in lines_target:#读取txt的
        #     _, x, y, dis, pat = line_target.strip().split(',')
        #     self.ueloc_x_target.append(float(x))
        #     self.ueloc_y_target.append(float(y))
        #     self.distance_target.append(float(dis))
        #     self.pathloss_target.append(float(pat))
        for line_target in lines_target:#读取csv的
            # _, _, scale, x, y, phi, theta, dis, gain = line_target#00032
            _, _, x, y, _, _, _, phi, theta, dis, gain, _, _, _, _ = line_target
            self.ueloc_x_target.append(float(x))
            self.ueloc_y_target.append(float(y))
            self.distance_target.append(float(dis))
            self.pathloss_target.append(float(gain))
            self.DoD_phi_target.append(float(phi))
            self.DoD_theta_target.append(float(theta))
        for line in lines:
            # image_path, ueloc_x, ueloc_y = line.strip().split(',')#txt读取
            image_path, classnum_CFR, ueloc_x, ueloc_y, _, _, _, DoD_phi, DoD_theta, distance, pathloss, classnum_ADP, _, classnum, classnum_CFR_manmadearea= line#csv读取
            self.image_path.append(image_path)
            # self.labels.append(int(label))
            self.z.append(int(classnum))
            self.z_CFR.append(int(classnum_CFR))
            self.z_ADP.append(int(classnum_ADP))
            self.z_CFR_manmadearea.append(int(classnum_CFR_manmadearea))
            self.ueloc_x.append(float(ueloc_x))
            self.ueloc_y.append(float(ueloc_y))
            self.distance.append(float(distance))
            self.pathloss.append(float(pathloss))
            self.DoD_phi.append(float(DoD_phi))
            self.DoD_theta.append(float(DoD_theta))
        self.root_dir = root_dir
        self.transform = transform
        #save before generalize
        # self.ueloc_x_target = self.ueloc_x
        # self.ueloc_y_target = self.ueloc_y

        #generalize lists
        distancemin = min(self.distance_target)
        distancemax = max(self.distance_target)
        for i, x in enumerate(self.distance):
            self.distance[i] = (x - distancemin) / (distancemax - distancemin)
            # self.distance[i] = x / distancemax
        pathlossmin = min(self.pathloss_target)
        pathlossmax = max(self.pathloss_target)
        for i, x in enumerate(self.pathloss):
            self.pathloss[i] = (x - pathlossmin) / (pathlossmax - pathlossmin)
            # self.pathloss[i] = x / pathlossmax
        DoD_phimin = min(self.DoD_phi_target)
        DoD_phimax = max(self.DoD_phi_target)
        for i, x in enumerate(self.DoD_phi):
            self.DoD_phi[i] = (x - DoD_phimin) / (DoD_phimax - DoD_phimin)
        DoD_thetamin = min(self.DoD_theta_target)
        DoD_thetamax = max(self.DoD_theta_target)
        for i, x in enumerate(self.DoD_theta):
            self.DoD_theta[i] = (x - DoD_thetamin) / (DoD_thetamax - DoD_thetamin)
        # ToAmin = min(self.ToA_target)
        # ToAmax = max(self.ToA_target)
        # for i, x in enumerate(self.ToA):
        #     self.ToA[i] = (x - ToAmin) / (ToAmax - ToAmin)
        # phasemin = min(self.phase_target)
        # phasemax = max(self.phase_target)
        # for i, x in enumerate(self.phase):
        #     self.phase[i] = (x - phasemin) / (phasemax - phasemin)
        # powermin = min(self.power_target)
        # powermax = max(self.power_target)
        # for i, x in enumerate(self.power):
        #     self.power[i] = (x - powermin) / (powermax - powermin)
        # # xmin = min(self.ueloc_x_target)
        # # xmax = max(self.ueloc_x_target)
        # # for i, x in enumerate(self.ueloc_x):
        # #     # self.ueloc_x[i] = (x - xmin) / (xmax - xmin)
        # #     self.ueloc_x[i] = x / xmax
        # # ymin = min(self.ueloc_y_target)
        # # ymax = max(self.ueloc_y_target)
        # # for i, x in enumerate(self.ueloc_y):
        # #     # self.ueloc_y[i] = (x - ymin) / (ymax - ymin)
        # #     self.ueloc_y[i] = x / ymax

    def __len__(self):
        # 展示数据中总共有多少个样本
        return len(self.image_path)

    def __info__(self):
        print("CustomData")
        print("\t Number of samples: {}".format(len(self.image_path)))
        # print("\t Number of classes: {}".format(len(np.unique(self.labels))))
        print("\t root_dir: {}".format(self.root_dir))
        print("\t imgdic: {}".format(generatedFolderNlos_SNR))
        print("\t Minimum of ueloc_x: {}".format(min(self.ueloc_x_target)))
        print("\t Dataset of ueloc_x: {}".format(self.ueloc_x))

    def __getitem__(self, idx):
        # lr = 3e-3  # learning rate
        # 保证idx不是一个tensor
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # 图像目录

        imgdic = os.path.join(self.root_dir, 'image', self.image_path[idx]) #NLOS, NLOS+LOS
        # imgdic = os.path.join(generatedFolderNlos_SNR, self.image_path[idx]) #NLOS SNR
        # 提取出的，索引为idx的图像的像素值矩阵
        #这三行是针对WithPhase的
        channel_2D = Image.open(imgdic)
        channel_2D = np.asarray(channel_2D, dtype=float)  # (64, 64, 3)
        #归一化出channel_2D
        channel_temp = (channel_2D[:, :, 0] / 255. + 1j * channel_2D[:, :, 1] / 255. - 0.5 - 1j * 0.5) * 2
        channel_temp = channel_temp / np.max(np.abs(channel_temp))
        channel_temp = np.concatenate((np.real(channel_temp)[:, :, None], np.imag(channel_temp)[:, :, None]), axis=2) # (64, 64, 2)
        #resize channel
        width = 224
        height = 224
        dim = (width, height)
        # resize image
        resized = cv2.resize(channel_2D, dim, interpolation=cv2.INTER_AREA)
        channel = resized.transpose(2, 0, 1).astype(np.float32) # (2or3 224 224)
        channel = torch.tensor(channel, dtype=torch.float32)
        # resize
        resized_temp = cv2.resize(channel_temp, dim, interpolation=cv2.INTER_AREA)
        channel_2channels = resized_temp.transpose(2, 0, 1).astype(np.float32)  # (2 224 224)
        channel_2channels = torch.tensor(channel_2channels, dtype=torch.float32)
        # #if not resize
        # channel = channel_2D#如果不resize的话
        # channel = channel.transpose(2, 0, 1).astype(np.float32)

        #生成ADP
        # ADPdic = os.path.join(self.root_dir, 'ADP', self.image_path[idx]) #NLOS的
        ADPdic = os.path.join(generatedFolderNlos_SNR_ADP, self.image_path[idx]) #NLOS SNR
        ADP = Image.open(ADPdic)
        ADP = np.asarray(ADP, dtype=float)  # (64, 64)
        resized_ADP = cv2.resize(ADP, dim, interpolation=cv2.INTER_AREA)
        ADP = resized_ADP.astype(np.float32)
        ADP = torch.tensor(ADP[None,:,:], dtype=torch.float32)
        combineCFRADP = torch.cat((channel_2channels, ADP), dim=0)
        # print(combineCFRADP.shape)


        # #读取的是数组
        # arraydic = os.path.join(self.root_dir, self.image_path[idx].strip().split('.')[0] + '.npy')
        # H = np.load(arraydic, allow_pickle=True)
        # H = H[0, :, :] + 1j * H[1, :, :]#读取出了最原始的H
        # G = self.V.T.conjugate() @ H @ self.F
        # ADP = np.abs(G)
        # ADP = ADP / np.max(ADP)
        # ADP = torch.tensor(ADP, dtype=torch.float32)

        # label = self.labels[idx]
        # label = torch.tensor([label])
        UE_x = self.ueloc_x[idx]
        UE_x = torch.tensor([UE_x])
        UE_y = self.ueloc_y[idx]
        UE_y = torch.tensor([UE_y])
        z = self.z[idx]
        z_CFR = self.z_CFR[idx]
        z_ADP = self.z_ADP[idx]
        z_CFR_manmadearea = self.z_CFR_manmadearea[idx]
        # #line当中的'class'没有z=3的
        # if z >= 4 and z <= 6:
        #         z = z - 1
        z = torch.tensor([z])
        z_CFR = torch.tensor([z_CFR])
        z_ADP = torch.tensor([z_ADP])
        # classnum = self.class_num[self.class_z.index(z)]
        # classnum = torch.tensor([classnum])
        UE_distance = self.distance[idx]
        UE_distance = torch.tensor([UE_distance])
        UE_gain = self.pathloss[idx]
        UE_gain = torch.tensor([UE_gain])
        dod_phi = self.DoD_phi[idx]
        dod_phi = torch.tensor([dod_phi])
        dod_theta = self.DoD_theta[idx]
        dod_theta = torch.tensor([dod_theta])


        if self.transform != None:
            channel = self.transform(channel)
            channel_2channels = self.transform(channel_2channels)
        # sample = (channel, label)
        # sample = {'channel': channel,
        #           'z':z,
        #           'x':UE_x,
        #           'y':UE_y,
        #           'lr':lr}
        # sample = (channel, z, UE_x, UE_y)#这里的UE_x/y的float只能是四位小数，但是保持了拥有的和预测的都是同一个值
        sample = (channel_2channels, z, UE_x, UE_y, UE_distance, UE_gain, dod_phi, dod_theta, ADP, combineCFRADP, z_CFR, z_ADP, z_CFR_manmadearea)#这里的UE_x/y的float只能是四位小数，但是保持了拥有的和预测的都是同一个值
        return sample
        #scale:[0.7710843086242676]全部都是


if __name__ == '__main__':
    train_data = ImgDataset(path, train_path)
    test_data = ImgDataset(path, test_path)
    print(train_data[0][8].shape)
    # # print(train_data[0]['lr'])
    # # print(test_data[1452][0].shape)
    # print(train_data.__len__())
    # print(test_data.__len__())
    # print(train_data.__info__())
    # print(test_data.__info__())
