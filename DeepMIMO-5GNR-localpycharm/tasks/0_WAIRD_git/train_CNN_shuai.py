'''
根据dataset的数值来进行训练
'''

import os
import sys
sys.path.append("../../")
from util_waird import *
from parameters_waird import *
import pandas as pd

from sipml_shuai import *
# from dataset import DoraSet
from dataset_shuai import ImgDataset
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
writer=SummaryWriter(log_dir='logs')

#导入pytorch一个完整流程所需的可能全部的包


import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torchvision import transforms as T
from torchvision import models as m
from torch.utils.data import DataLoader
#导入作为辅助工具的各类包
import matplotlib.pyplot as plt
from time import time
import numpy as np
import gc
import torch

#一些系统参数
casesToTrain = 0 # 1 for los, 0 for nlos, -1 for all
saveLossInterval = 1 # intervals to save loss
saveModelInterval = 1 # intervals to save model
cudaIdx = "cuda:0" # GPU card index
num_workers = 8 # workers for dataloader
evaluation = False # evaluation only if True
loadRUNcase = f'/runCase_{casesToTrain}/DoraNet_10.pth' # path of model to be loaded if evalution is True
torch.backends.cudnn.enabled = False

'''
parameters to be optimized 要调的参数
'''
epochs = 150 #10
# batchSize = 32 # NLOS [1,8,8] 32
batchSize = 16 # NLOS [1,64,1] 16
# batchSize = 64 # LOS [1,64,1]
# batchSize = 80 # NLOS+LOS 80
lr = 3e-3 # learning rate
epoch_stop = 5
# gamma = 0
# wd = 0
tol = 10**(-5)
model_path = "results/model_state_dict"
modelname = "CNN_Regression"
imgname = "CNN_Regression"

# train_path = os.path.join('/data2/wbh/DeepMIMO-5GNR/DeepMIMO_python/data/label', 'train_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv')
# test_path = os.path.join('/data2/wbh/DeepMIMO-5GNR/DeepMIMO_python/data/label', 'test_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv')
# root = "/data2/wbh/DeepMIMO-5GNR/DeepMIMO_python/channel/withPhase"


def run(isTrain, data_loader, model, epoch, device, imageSize):#一个batch运行的程序，不区分train/test
    # model.train() if isTrain else model.eval()
    setStr = 'Train' if isTrain else 'Valid'
    losses = Recoder()
    Acc = Recoder()
    # ueAcc = Recoder()
    ueAcc = 0.0
    # scale = float(0.7710843086242676)
    cnt = 0
    loss_s = 0.0
    offset_s = 0.0
    acc_s = 0.0
    total = 0
    offset = torch.tensor([0.0])
    p_ueloc_x = []
    p_ueloc_y = []
    sumoffset = []
    # data_csv = open("/data2/wbh/DoraSet_code/data/dataset_2_00032_5_10_15_30_1000_label/dataset_2_00032_5_10_15_30_1000_target_labelUELoc.txt", 'r')
    # lines = data_csv.readlines()
    # for i, (channels, UE_x, UE_y, UE_distance, UE_pathloss) in enumerate(data_loader):#change according to dataset
    for i, (channels, z, UE_x, UE_y, UE_distance, UE_gain, dod_phi, dod_theta, ADP, combineCFRADP, z_CFR, z_ADP, z_CFR_manmadearea) in enumerate(data_loader):  # change according to dataset; (channel, z, UE_x, UE_y)
        p_ueloc_x = []
        p_ueloc_y = []

        #change data to gpu
        channels = channels.to(device, non_blocking=True)
        z = z.to(device, non_blocking=True)
        z_CFR = z_CFR.to(device, non_blocking=True)
        UE_x = UE_x.to(device, non_blocking=True)
        UE_y = UE_y.to(device, non_blocking=True)
        # UE_distance = UE_distance.to(device, non_blocking=True)
        # UE_pathloss = UE_pathloss.to(device, non_blocking=True)
        # xmin = xmin.to(device, non_blocking=True)
        # xmax = xmax.to(device, non_blocking=True)
        # ymin = ymin.to(device, non_blocking=True)
        # ymax = ymax.to(device, non_blocking=True)

        bs = channels.shape[0]  # 记录样本量
        y = torch.cat([UE_x, UE_y], dim=1)

        if isTrain:
            p_UEloc = model.update(z, channels, y, device)
            # p_UEloc = model.update(z, ADP, y, device) #将ADP换成channels，然后选择simpl_shuai.py中的对应encoder并更改update和predict函数
            print(
                # 'Epoch{}:[{}/{}({:.0f}%)]'.format(epoch,trainedsamples,allsamples,100*trainedsamples/allsamples))
                f'{setStr} Epoch: [{epoch}][{i}/{len(data_loader)}] ---- offset {p_UEloc["loss"]:.4f}meters')
        if not isTrain:
            y_hat = model.predict(z, channels, device)
            # y_hat = model.predict(z, ADP, device)
            offset = ((y - y_hat) ** 2).mean()

            # 将每个数据进行保存以之后绘画CDF图
            px = pd.DataFrame(y_hat.detach().cpu().numpy())
            py = pd.DataFrame(y.cpu().numpy())
            # 行拼接
            if i == 0:
                py_hat1 = px.copy(deep=True)
                py1 = py.copy(deep=True)
            else:
                py_hat1 = pd.concat([py_hat1, px])
                py1 = pd.concat([py1, py])
            # sumoffset_temp = np.sum((y - y_hat) ** 2, axis=1)
            # sumoffset.extend(sumoffset_temp)
            # print(sumoffset) #nd2db
            print(
                # 'Epoch{}:[{}/{}({:.0f}%)]'.format(epoch,trainedsamples,allsamples,100*trainedsamples/allsamples))
                f'{setStr} Epoch: [{epoch}][{i}/{len(data_loader)}] ---- offset {offset:.4f}meters')


        # offset = torch.mean(((p_UEloc[:, 0] - UE_x) ** 2 + (p_UEloc[:, 1] - UE_y) ** 2) ** 0.5)


        # 每个batch都打印

        # print(
        #     # 'Epoch{}:[{}/{}({:.0f}%)]'.format(epoch,trainedsamples,allsamples,100*trainedsamples/allsamples))
        #     f'{setStr} Epoch: [{epoch}][{i}/{len(data_loader)}] ---- offset {p_UEloc["loss"]:.4f}meters')
    # 清理GPU内存
        cnt += 1
        if isTrain:
            loss_s += p_UEloc['loss'].item()
        if not isTrain:
            loss_s += offset.item()
        # offset_s += offset.item()
        total += bs
    del channels
    gc.collect()
    torch.cuda.empty_cache()
    if isTrain:
        return loss_s / cnt
    return loss_s / cnt, py_hat1, py1 #loss_s / cnt


def showResults(envImg, p_UEloc, UEloc, BSloc, batchIdx, imageSize):
    fig, ax = plt.subplots()
    ax.imshow(1 - envImg[batchIdx, 0, :, :], cmap='gray')
    ax.scatter(imageSize * BSloc[batchIdx, 0], imageSize - imageSize * BSloc[batchIdx, 1], marker='^', c='red',
               label='BS location')
    ax.scatter(imageSize * UEloc[batchIdx, 0], imageSize - imageSize * UEloc[batchIdx, 1], marker='o', c='blue',
               label='real UE location')
    ax.scatter(imageSize * p_UEloc[batchIdx, 0], imageSize - imageSize * p_UEloc[batchIdx, 1], marker='o', c='green',
               label='predicted UE location')
    ax.legend()
    plt.show()
    fig.savefig("wholeImg.png")


def main():
    # 参数、模型、实例化
    seed_everything(42)
    # transform
    # tf = transforms.Compose([
    #     transforms.Grayscale(num_output_channels=3),  # 3为三通道，1为单通道
    #     transforms.ToTensor()])
    valid_dataset = ImgDataset(path, test_path)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batchSize, shuffle=False,
                                               num_workers=num_workers)
    imageSize = valid_dataset.imageSize#256
    device = torch.device(cudaIdx if torch.cuda.is_available() else "cpu")

    #ds-ae-com
    hparams = {'lr': lr,
               'temperature': 0.75}

    # choose the model
    # model = DS_AE_CNN(hparams, device).to(device, non_blocking=True)
    # model = DS_AE_CL(hparams, device).to(device, non_blocking=True)#多头+对比学习+cnn
    # model = AE_CL(hparams, device).to(device, non_blocking=True)#对比学习+cnn
    model = CNN(hparams, device).to(device, non_blocking=True)#cnn

    resultsfolder = os.path.join(savedatadir, case, 'CFR')
    # resultsfolder = os.path.join(savedatadir, case, 'ADP')
    print("resultsfolder:", resultsfolder)

    if os.path.exists(resultsfolder):
        pass
    else:
        # 创建文件夹
        os.mkdir(resultsfolder)

    Train_loss_loc = os.path.join(resultsfolder,
                                  'Train_loss_' + str(model.__class__.__name__) + '_' + str(sim_threshold) + '_' + str(
                                      sim_threshold_betweenTemp) + '_' + str(lr) + '_' + str(batchSize) + '_' + str(
                                      hparams['lr']) + '_' + str(hparams['temperature']) + '.csv')
    Test_loss_loc = os.path.join(resultsfolder,
                                 'Test_loss_' + str(model.__class__.__name__) + '_' + str(sim_threshold) + '_' + str(
                                     sim_threshold_betweenTemp) + '_' + str(lr) + '_' + str(batchSize) + '_' + str(
                                     hparams['lr']) + '_' + str(hparams['temperature']) + '.csv')

    # #读取此时model的名字
    # model_name = model.__class__.__name__
    # criterion = torch.nn.CrossEntropyLoss().to(device)
    Valid = []
    Train_Loss = []
    Test_Loss = []
    Train_Acc = []
    Test_Acc = []
    highestacc = None
    early_stopping = EarlyStopping(tol=tol)
    start = time()  # 计算训练时间


    # trainloss, testloss = full_procedure()
    if not evaluation:
        train_dataset = ImgDataset(path, train_path)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batchSize, shuffle=True,
                                                   num_workers=num_workers)
        # optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        Train = []
        runCase = f'runCase_{casesToTrain}'
        makeDIRs(runCase)
    else:
        model.load_state_dict(torch.load(f'models{loadRUNcase}'))
        print('*****')
        optimizer = []

    for epoch in range(1, epochs + 1):

        if (not evaluation):
            LossT = run(True, train_loader, model, epoch, device, imageSize)
            Train_Loss.append(LossT)
            LossV, py_hat1, py1 = run(False, valid_loader, model, epoch, device, imageSize)
            Test_Loss.append(LossV)


            # checkPoint(runCase, epoch, epochs, model, Train_Loss, Test_Loss, saveModelInterval, saveLossInterval)
        # else:
        #     with torch.no_grad():
        #         LossV= run(False, valid_loader, model, criterion, [], epoch, device, imageSize)
        #         print(LossV)
        else:
            break



        # 对每一个epoch打印训练和测试的结果
        # 训练集上的损失，测试集上的损失，训练集上的准确率，测试集上的准确率
        print(
            '\t Train Loss:{:.3f}m, Test Loss:{:.3f}m'.format(LossT
                                                              , LossV))

        # 如果测试集准确率出现新高/测试集loss出现新低，那我会保存现在的这一组权重
        if highestacc == None:
            highestacc = LossV
            df_y_hat = py_hat1
            df_y = py1
        if highestacc > LossV:
            highestacc = LossV
            df_y_hat = py_hat1
            df_y = py1
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            torch.save(model.state_dict(), os.path.join(model_path, str(model.__class__.__name__) + ".pt"))
            print("\t Weight Saved")

        # 提前停止
        early_stop = early_stopping(LossV)
        if early_stop == 'True':
            break

        if epoch == epoch_stop or epoch % 50 == 0:
            # 将Train Loss列表变为dataframe数据
            Train_Loss_pd = pd.DataFrame(Train_Loss)
            Train_Loss_pd.to_csv(Train_loss_loc, index=False, header=False)
            # 将Test Loss列表变为dataframe数据
            Test_Loss_pd = pd.DataFrame(Test_Loss)
            Test_Loss_pd.to_csv(Test_loss_loc, index=False, header=False)
            # 将目前最好的一次testloss的每一个bs的数据保存以后续计算CDF
            # print(df_y_hat)
            df_y_hat.to_csv(os.path.join(resultsfolder, 'df_y_hat_' + str(model.__class__.__name__) + '_' + str(
                sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(lr) + '_' + str(
                batchSize) + '_' + str(hparams['lr']) + '_' + str(hparams['temperature']) + '.csv'), index=False,
                            header=False)
            # 将目前最好的一次testloss的每一个bs的数据对应的xy保存以后续计算CDF
            # print(df_y)
            df_y.to_csv(os.path.join(resultsfolder,
                                     'df_y_' + str(model.__class__.__name__) + '_' + str(sim_threshold) + '_' + str(
                                         sim_threshold_betweenTemp) + '_' + str(lr) + '_' + str(batchSize) + '_' + str(
                                         hparams['lr']) + '_' + str(hparams['temperature']) + '.csv'), index=False,
                        header=False)
            if epoch == 150 or epoch == 400:
                df_y_hat.to_csv(os.path.join(resultsfolder, 'df_y_hat_' + str(epoch) + '_' + str(model.__class__.__name__) + '_' + str(
                    sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(lr) + '_' + str(
                    batchSize) + '_' + str(hparams['lr']) + '_' + str(hparams['temperature']) + '.csv'), index=False,
                                header=False)
                df_y.to_csv(os.path.join(resultsfolder,
                                         'df_y_' + str(epoch) + '_' + str(model.__class__.__name__) + '_' + str(sim_threshold) + '_' + str(
                                             sim_threshold_betweenTemp) + '_' + str(lr) + '_' + str(
                                             batchSize) + '_' + str(hparams['lr']) + '_' + str(
                                             hparams['temperature']) + '.csv'), index=False, header=False)

            print('time:{}', format(time() - start))

        # if epoch == 60:
        #     # 我写的稍微有点复杂的绘图
        #     Test_Loss = torch.tensor(Test_Loss, device='cpu')
        #     Train_Loss = torch.tensor(Train_Loss, device='cpu')
        #     dict = {'Train_Loss': Train_Loss.cpu(), 'Test_Loss': Test_Loss.cpu()}  # save the last training
        #     np.save('results/dict1.npy', dict)
        #     dict = np.load('results/dict1.npy', allow_pickle=True)  # Train_Loss,Test_Loss
        #     loss_image(dict.item()['Train_Loss'], dict.item()['Test_Loss'], imgname)
    # #将Train Loss列表变为dataframe数据
    # Train_Loss = pd.DataFrame(Train_Loss)
    # Train_Loss.to_csv(Train_loss_loc, index=False, header=False)
    # # 将Test Loss列表变为dataframe数据
    # Test_Loss = pd.DataFrame(Test_Loss)
    # Test_Loss.to_csv(Test_loss_loc, index=False, header=False)



    print('time:{}', format(time() - start))
    # showResults(envImg, p_UEloc, UEloc, BSloc, 0, imageSize)

    # # 我写的稍微有点复杂的绘图
    # Test_Loss = torch.tensor(Test_Loss, device='cpu')
    # Train_Loss = torch.tensor(Train_Loss, device='cpu')
    # dict = {'Train_Loss': Train_Loss.cpu(), 'Test_Loss': Test_Loss.cpu()}  # save the last training
    # np.save('results/dict1.npy', dict)
    # dict = np.load('results/dict1.npy', allow_pickle=True)  # Train_Loss,Test_Loss
    # loss_image(dict.item()['Train_Loss'], dict.item()['Test_Loss'], imgname)


if __name__ == '__main__':
    print(f'maxpathnum: {maxPathNum}')
    print('power:', Pattern_t['Power'])
    print('SNR:', SNR)
    main()