# '''
# 生成generatedFolder当中的png信道图片
# '''
#
# import os
# from util import *
# from tqdm import tqdm
# from parameters import *
# from torch.utils.data import Dataset, DataLoader
# from matplotlib import pyplot as plt
#
#
# class DoraSet(Dataset):
#     def __init__(self):
#         self.fcGHz = float(carrierFreq.replace('_', '.'))
#         self.fGHz = np.linspace(-0.5 * BWGHz, 0.5 * BWGHz, sampledCarriers) + self.fcGHz
#         self.Hs = {}
#         self.Ps = {}
#         self.envs = []
#         path_list = os.listdir(scenarioFolder)
#         path_list.sort(key=lambda x: int(x))
#         for i in tqdm(path_list[:ENVnum]):
#             self.envs.append(i)
#             envPath = os.path.join(scenarioFolder, i)
#             H = np.load(os.path.join(envPath, f'H_{carrierFreq}_G.npy'), allow_pickle=True, encoding='latin1')
#             self.Hs[len(self.envs) - 1] = H.item()#precision,but i'm confused what difference between item and items
#             P = np.load(os.path.join(envPath, 'Path.npy'), allow_pickle=True, encoding='latin1')
#             self.Ps[len(self.envs) - 1] = P.item()
#             if saveAsArray:
#                 arrayFolder = os.path.join(generatedFolder, i, 'array')
#                 if not os.path.exists(arrayFolder):
#                     os.makedirs(arrayFolder)
#             if saveAsImage:
#                 imageFolder = os.path.join(generatedFolder, i, 'image')
#                 if not os.path.exists(imageFolder):
#                     os.makedirs(imageFolder)
#
#     def __getitem__(self, idx): #idx:想获得的第几个idx的后续值，getitem来进行提取
#         curEnv = idx // (BSnum * UEnum)#case
#         curLink = idx % (BSnum * UEnum)
#         bsIdx = curLink // UEnum
#         ueIdx = curLink % UEnum
#         linkStr = f'bs{BSlist[bsIdx]}_' + (f'ue{UElist[ueIdx]}' if scenario == 1 else f'ue{UElist[ueIdx]:05d}')
#         H = self.Hs[curEnv][linkStr] #*第几个场景的第几个基站-用户对应信道
#         P = self.Ps[curEnv][linkStr]
#         tau = np.asarray(P['taud'])#单位ns
#         sortedPath = np.argsort(tau)[:maxPathNum]#Taud从小到大排列对应值的索引
#         doa = np.asarray(P['doa'])
#         dod = np.asarray(P['dod'])
#         pos_r = antennaPosition(Nr, spacing_r, Basis_r)
#         res_r = arrayResponse(doa, pos_r, sortedPath)
#         pos_t = antennaPosition(Nt, spacing_t, Basis_t)
#         res_t = arrayResponse(dod, pos_t, sortedPath)
#         pow_t = 10 ** (Pattern_t['Power'] / 10)#转成mW
#         norm_H = H * (pow_t ** 0.5) / (subcarriers ** 0.5)
#         ofdm_H = norm_H[sortedPath, None] * np.exp(-2 * 1j * np.pi * tau[sortedPath, None] * self.fGHz[None, :])
#         CFR = np.sum(ofdm_H[:, None, None, :] * res_t[:, :, None, None] * res_r[:, None, :, None],
#                      axis=0)  # dimensions in (Nt,Nr,Nf)?这个式子expression
#         channel = np.reshape(CFR, (-1, sampledCarriers))  # dimensions in (Nt*Nr,Nf)?this 2 equations i can't write
#         channel2 = CFR[:, :, 31]#27.99963GHz
#         if saveAsArray:
#             arrayPath = os.path.join(generatedFolder, self.envs[curEnv], 'array', linkStr + '.npy')
#             array = np.concatenate((np.real(channel)[None, ...], np.imag(channel)[None, ...]), axis=0)
#             np.save(arrayPath, array)
#             # #channel2
#             # arrayPath2 = os.path.join(generatedFolder, self.envs[curEnv], 'array', linkStr + '_2.npy')
#             # # array2 = np.concatenate((np.real(channel2)[None, ...], np.imag(channel2)[None, ...]), axis=0)
#             # np.save(arrayPath2, channel2)#ndarray 2*32 (Nt,Nr)
#         if saveAsImage:
#             norm_channel = channel / np.max(np.abs(channel))  # to save as image, should be normalized first
#             image = (norm_channel / 2 + 0.5 + 0.5 * 1j) * 255 #norm-channel/2后实部虚部都在0-0.5之间，为了图片处理+0.5+0.5i
#             image = np.concatenate(
#                 (np.uint8(np.round(np.real(image)))[:, :, None], np.uint8(np.round(np.imag(image)))[:, :, None]),
#                 axis=-1)#取整,
#             image = Image.fromarray(image, mode='LA') #灰度+透明，2通道
#             imagePath = os.path.join(generatedFolder, self.envs[curEnv], 'image', linkStr + '.png')
#             image.save(imagePath)
#             #channel2 process
#             norm_channel2 = channel2 / np.max(np.abs(channel2))
#             # array2 = np.concatenate((np.real(norm_channel2)[None, ...], np.imag(norm_channel2)[None, ...]), axis=0)
#             array3 = torch.from_numpy(np.real(norm_channel2))
#             plt.title('H_28.0sample')
#             plt.imshow(array3)
#             plt.axis("off")
#             plt.savefig('H_28.0sample.png')
#         return channel
#
#     def __len__(self):
#         return ENVnum * BSnum * UEnum
#
#
# if __name__ == "__main__":
#     print('Init...')
#     doraset = DoraSet()
#     generator = DataLoader(doraset, batch_size=100, shuffle=False, num_workers=numCores, pin_memory=True)
#     print('Generating...')
#     for channel in tqdm(generator):
#         pass
#     print('Done!')
#


'''
只根据固定的比如00743scenarioFolder生成对应场景的的generatedFolder当中的png信道图片
主要的几个参数不在parameters_waird里面，在自己的main函数里面
'''

import os
from util_waird import *
from tqdm import tqdm
from parameters_waird import *
from torch.utils.data import Dataset, DataLoader
from matplotlib import pyplot as plt


class DoraSet(Dataset):
    def __init__(self, env_num, generatedFolder_dir):
        #load parameters
        # generatedFolder = f'/workspace/wbh/DoraSet_code/data/generated_{scenario}_{carrierFreq}_{maxPathNum}_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}_{int(BWGHz * 1000)}_{sampledCarriers}/'#和parameters_waird.py的路径不一样，不然会重复
        print('generatedFolder_dir:', generatedFolder_dir)
        self.envs = env_num
        envPath = '/workspace/wbh/DoraSet_code/data/scenario_2/' + self.envs

        self.fcGHz = float(carrierFreq.replace('_', '.'))
        self.fGHz = np.linspace(-0.5 * BWGHz, 0.5 * BWGHz, sampledCarriers) + self.fcGHz
        self.Hs = {}
        self.Ps = {}

        H = np.load(os.path.join(envPath, f'H_{carrierFreq}_G.npy'), allow_pickle=True, encoding='latin1')
        self.Hs[0] = H.item()
        P = np.load(os.path.join(envPath, 'Path.npy'), allow_pickle=True, encoding='latin1')
        self.Ps[0] = P.item()
        if saveAsArray:
            self.arrayFolder = os.path.join(generatedFolder_dir, 'array')
            if not os.path.exists(self.arrayFolder):
                os.makedirs(self.arrayFolder)
            print(self.arrayFolder)
        if saveAsImage:
            self.imageFolder = os.path.join(generatedFolder_dir, 'image')
            if not os.path.exists(self.imageFolder):
                os.makedirs(self.imageFolder)
            print(self.imageFolder)
        if saveAsAmpImage:
            self.ampimageFolder = os.path.join(generatedFolder_dir, 'ampimage')
            if not os.path.exists(self.ampimageFolder):
                os.makedirs(self.ampimageFolder)


    def __getitem__(self, idx): #idx:想获得的第几个idx的后续值，getitem来进行提取
        curEnv = 0
        curLink = idx % (BSnum * UEnum)
        bsIdx = curLink // UEnum
        ueIdx = curLink % UEnum
        linkStr = f'bs{BSlist[bsIdx]}_' + (f'ue{UElist[ueIdx]}' if scenario == 1 else f'ue{UElist[ueIdx]:05d}')
        H = self.Hs[curEnv][linkStr] #*第几个场景的第几个基站-用户对应信道
        P = self.Ps[curEnv][linkStr]
        tau = np.asarray(P['taud'])#单位ns
        sortedPath = np.argsort(tau)[:maxPathNum]#Taud从小到大排列对应值的索引
        doa = np.asarray(P['doa'])
        dod = np.asarray(P['dod'])
        pos_r = antennaPosition(Nr, spacing_r, Basis_r)
        res_r = arrayResponse(doa, pos_r, sortedPath)
        pos_t = antennaPosition(Nt, spacing_t, Basis_t)
        res_t = arrayResponse(dod, pos_t, sortedPath)
        pow_t = 10 ** (Pattern_t['Power'] / 10)#转成mW
        norm_H = H * (pow_t ** 0.5) / (subcarriers ** 0.5)
        ofdm_H = norm_H[sortedPath, None] * np.exp(-2 * 1j * np.pi * tau[sortedPath, None] * self.fGHz[None, :])
        CFR = np.sum(ofdm_H[:, None, None, :] * res_t[:, :, None, None] * res_r[:, None, :, None],
                     axis=0)  # dimensions in (Nt,Nr,Nf)?这个式子expression
        channel = np.reshape(CFR, (-1, sampledCarriers))  # dimensions in (Nt*Nr,Nf)?this 2 equations i can't write
        channel2 = CFR[:, :, 31]#27.99963GHz
        if saveAsArray:
            arrayPath = os.path.join(self.arrayFolder, linkStr + '.npy')
            array = np.concatenate((np.real(channel)[None, ...], np.imag(channel)[None, ...]), axis=0)
            np.save(arrayPath, array)


            # #channel2
            # arrayPath2 = os.path.join(generatedFolder_dir, self.envs[curEnv], 'array', linkStr + '_2.npy')
            # # array2 = np.concatenate((np.real(channel2)[None, ...], np.imag(channel2)[None, ...]), axis=0)
            # np.save(arrayPath2, channel2)#ndarray 2*32 (Nt,Nr)
        if saveAsImage:
            norm_channel = channel / np.max(np.abs(channel))  # to save as image, should be normalized first
            image = (norm_channel / 2 + 0.5 + 0.5 * 1j) * 255 #norm-channel/2后实部虚部都在0-0.5之间，为了图片处理+0.5+0.5i
            image = np.concatenate(
                (np.uint8(np.round(np.real(image)))[:, :, None], np.uint8(np.round(np.imag(image)))[:, :, None]),
                axis=-1)#取整,
            image = Image.fromarray(image, mode='LA') #灰度+透明，2通道
            imagePath = os.path.join(self.imageFolder, linkStr + '.png')
            image.save(imagePath)
            #channel2 process
            norm_channel2 = channel2 / np.max(np.abs(channel2))
            # array2 = np.concatenate((np.real(norm_channel2)[None, ...], np.imag(norm_channel2)[None, ...]), axis=0)
            array3 = torch.from_numpy(np.real(norm_channel2))
            # plt.title('H_28.0sample')
            # plt.imshow(array3)
            # plt.axis("off")
            # plt.savefig('H_28.0sample.png')
        if saveAsAmpImage:
            #channel的每一个元素都取幅值
            channel = np.abs(channel) / np.max(np.abs(channel))
            imagePath = os.path.join(self.ampimageFolder, linkStr + '.png')
            Image.fromarray(np.uint8(channel * 255)).convert('RGB').save(imagePath)

        return channel

    def __len__(self):
        return ENVnum * BSnum * UEnum


if __name__ == "__main__":
    if not os.path.exists(generatedFolder_dir):
        os.makedirs(generatedFolder_dir)
    print('Init...')
    doraset = DoraSet(case, generatedFolder_dir)
    generator = DataLoader(doraset, batch_size=100, shuffle=False, num_workers=numCores, pin_memory=True)
    print('Generating...')
    for channel in tqdm(generator):
        pass
    print('Done!')


