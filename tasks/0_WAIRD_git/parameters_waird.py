
'''
Copyright (C) 2021. Huawei Technologies Co., Ltd.
This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
'''

import warnings
import numpy as np
import os
import multiprocessing
from PIL import Image
warnings.filterwarnings('ignore')

numCores = multiprocessing.cpu_count() # number of cores to do generation job, decrease this number if your computer has less memory or available cores
# numCores = 25 #40目前最好的设置是50，60的话速度变化不大，CPU占用80

# parameters for generating dataset
carrierFreq = '60_0' # for example, 2_6 for 2.6G, 60_0 for 60.0G 28_0
BWGHz = 0.05 # bandwidth in GHz 0.04608
subcarriers = 384 # number of subcarriers
carrierSampleInterval = 6 # sample subcarriers with this interval to save computation time
sampledCarriers = int(subcarriers/carrierSampleInterval) # number of sampled subcarriers for deep learning
Nt = [1, 64, 1] # BS antenna array in [x,y,z] axis, e.g., [1, 8, 8], [1, 32, 4]  [8, 1, 1]
Nr = [1, 1, 1] # UE antenna array in [x,y,z] axis, e.g., [2, 2, 1], [4, 2, 1]
spacing_t = [0.5, 0.5, 0.5] # transmitter antenna spacing in wavelength
spacing_r = [0.5, 0.5, 0.5] # receiver antenna spacing in wavelength
Pattern_t = {'Power':0} # omni antenna type for default, transmitter power 0 dBm
Basis_t = np.eye(3) # antenna basis rotation, no rotation for default
Basis_r = np.eye(3) # antenna basis rotation, no rotation for default
saveAsArray = True # save channel as numpy array if True
saveAsImage = True # save channel as image if True
saveAsAmpImage = False
maxPathNum = 10 # should be >0, max Path number for every BS-UE link, a large number such as 1000 means no limits 1000
num_tau = 10 #template_final_largerthan最后一步将数据集中小于等于num_tau的给去掉
SNR = 35 #dB，和gentarget_waird.ipynb保持一致

scenario = 2 # select a scenario to generate channel, the detailed description of scenarios are listed below
# scenarioFolder = f'data/scenario_{scenario}/' # folder of scenario primary
# scenarioFolder = f'/workspace/wbh/DoraSet_code/data/scenario_2_00032/10'
scenarioFolder_2 = f'data/scenario_2_00032/'
# generatedFolder = f'data/generated_{scenario}_{carrierFreq}_{maxPathNum}_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}_{int(BWGHz*1000)}_{sampledCarriers}/'
# generatedFolder = '/workspace/wbh/DoraSet_code/data/generated_2_00032/10/image'
generatedFolder_CE_GAN = f'data/generated_{scenario}_{carrierFreq}_{maxPathNum}_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}_{int(BWGHz*1000)}_{sampledCarriers}_{2}/'
generatedFolder_2 = f'data/generated_2_00032/'
if scenario==1:
    # scenario_1: sparse UE drop in lots of environments
    # max 10000 envs, 5 BS and 30 UE drops can be selected for every environment
    ENVnum = 1000 # number of environments to pick, max is 10000 first1000
    BSlist = list(range(5)) # BS index range 0~4 per environment, e.g., [0] picks BS_0, [2,4] picks BS_2 and BS_4
    UElist = list(range(30)) # UE index range 0~29 per environment, e.g., [0] picks UE_0, [2,17,26] picks UE_2, UE_17 and UE_26
    ENVlist = list(range(ENVnum))
    BSnum = len(BSlist) # number of BS per environment, max is 5
    UEnum = len(UElist) # number of UE per environment, max is 30
elif scenario==2:
    # scenario_2: dense UE drop in some environments
    # max 100 envs, 1 BS and 10000 UE drops can be selected for every environment
    ENVnum = 30 # number of environments to pick, max is 100
    BSlist = list(range(1)) # BS index range 0~0 per environment, e.g., [0] picks BS_0
    UElist = list(range(10000)) # UE index range 0~9999 per environment, e.g., [0] picks UE_0, [2,170,2600] picks UE_2, UE_170 and UE_2600
    BSnum = len(BSlist) # number of BS per environment, max is 1
    UEnum = len(UElist) # number of UE per environment, max is 10000
else:
    raise('More scenarios are in preparation.')

'''
gentarget_waird
'''
case = '00743'
# case = '00247'
# case = '00670'
scenarioFolder = '/workspace/wbh/DoraSet_code/data/scenario_2/' + case
generatedFolder_dir = f'/workspace/wbh/DoraSet_code/data/generated_{scenario}_{carrierFreq}_{maxPathNum}_' + str(Pattern_t['Power']) + f'_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}_{int(BWGHz * 1000)}_{sampledCarriers}/' + case
generatedFolder = os.path.join(generatedFolder_dir, 'image/')#形式： '/workspace/wbh/DoraSet_code/data/generated_2_60_0_10_1_8_8_1_2_2_50_64/' + case + '/image/'
targetfile = '/workspace/wbh/DeepMIMO-5GNR-localpycharm/data/label/target_' + case + '_10000.csv' #和f'_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}'无关
targetfile_los = '/workspace/wbh/DeepMIMO-5GNR-localpycharm/data/label/targetLOS_' + case + '_10000.csv' #和f'_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}'无关
targetfile_nlos = '/workspace/wbh/DeepMIMO-5GNR-localpycharm/data/label/targetNLOS_' + case + '_10000.csv' #和f'_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}无关
generatedFolderNlos_dir = f'/workspace/wbh/DoraSet_code/data/generatedNLOS_{scenario}_{carrierFreq}_{maxPathNum}_' + str(Pattern_t['Power']) + f'_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}_{int(BWGHz * 1000)}_{sampledCarriers}/' + case
generatedFolderNlos = os.path.join(generatedFolderNlos_dir, 'image/') #没有信噪比的形况
generatedFolderNlos_array = os.path.join(generatedFolderNlos_dir, 'array/')
generatedFolderlos_dir = f'/workspace/wbh/DoraSet_code/data/generatedLOS_{scenario}_{carrierFreq}_{maxPathNum}_' + str(Pattern_t['Power']) + f'_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}_{int(BWGHz * 1000)}_{sampledCarriers}/' + case
generatedFolderlos = os.path.join(generatedFolderlos_dir, 'image/') #没有信噪比的形况
generatedFolderlos_array = os.path.join(generatedFolderlos_dir, 'array/')
#NLOS SNR系列
generatedFolderNlos_SNR_dir = os.path.join(generatedFolderNlos_dir, 'SNR/')
generatedFolderNlos_SNR = os.path.join(generatedFolderNlos_SNR_dir, 'image_' + str(SNR) + '/')
generatedFolderNlos_SNR_array = os.path.join(generatedFolderNlos_SNR_dir, 'array_' + str(SNR) + '/')
generatedFolderNlos_SNR_ADP = os.path.join(generatedFolderNlos_SNR_dir, 'ADP_' + str(SNR) + '/')
# generatedFolderNlos = '/workspace/wbh/DoraSet_code/data/generated_2_00032/10/image/'#这个不是00032的，00032没找到


'''
Parameters from 0_regressionTemplate
The same in the jupyter-notebook: lightly/compute_offset
'''
# # train_prior_shuai.py或者train_cnn_shuai.py需要不注释下面三行
# path = generatedFolder_dir #NLOS+LOS的话只用更改这里
path = generatedFolderNlos_dir #这个是NLOS的图片，训练时候用到了
# path = generatedFolderlos_dir #这个是LOS的图片，训练时候用到了
path_list = os.listdir(os.path.join(path,'image')) #对应path = generatedFolderNlos_dir，跑generator.py时候注释掉，不然会显示程序有问题
num_images = len(path_list)#，跑generator.py时候注释掉，不然会显示程序有问题
# # 跑generator.py或者template_matching.py需要注释掉上面三行，然后不注释下面这一行。不然程序会有问题
# num_images = 0

# Define the path to the image folder
# path = '/workspace/wbh/DoraSet_code/data/generated_2_00032/10/image/' #这个是所有的图片，不知道WAIRD后续的程序有没有用到这个
# path = generatedFolder #这个是所有的图片，跑generator.py时候注释掉，不然会显示程序有问题
# path = '/workspace/wbh/DoraSet_code/data/generated_2_60_0_10_1_8_8_1_1_1_50_64/00247/image/' #之前的，没怎么用
# Define the number of images to classify
# path_list = os.listdir(os.path.join(path, 'ADP')) #00743的是这样。之后要改成新的样子，把adp移到上一级目录
# path_list.sort(key=lambda x: int(x.split('.')[0]))  # 没有必要用。对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型

#template_matching.py的参数
sim_threshold = 0.99 #组内相似度：templatematching.py
sim_threshold_betweenTemp = 0.99 #组间相似度：这个参数并没有在template_matching.py中用到
dist_threshold = 100
stop_threshold = num_images+500
template_size_x = 8
template_size_y = 16


savedatadir = f'/workspace/wbh/DeepMIMO-5GNR-localpycharm/tasks/0_WAIRD/results/generated_{scenario}_{carrierFreq}_{maxPathNum}_' + str(Pattern_t['Power']) + f'_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}_{int(BWGHz * 1000)}_{sampledCarriers}/'
# savedatadir = f'/workspace/wbh/DeepMIMO-5GNR-localpycharm/tasks/0_WAIRD/results/generated_{scenario}_{carrierFreq}_{maxPathNum}_' + str(Pattern_t['Power']) + f'_{Nt[0]}_{Nt[1]}_{Nt[2]}_{Nr[0]}_{Nr[1]}_{Nr[2]}_{int(BWGHz * 1000)}_{sampledCarriers}/SNR_{SNR}/' #NLOS SNR
saveImagesforOverleafdir = f'/workspace/wbh/DeepMIMO-5GNR-localpycharm/tasks/0_WAIRD/results/draw/WAIRD/'

# classified_images_file = 'classified_images_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv'
# classified_images_file = 'classified_images_'  + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv'
classified_images_file = 'classified_images_2templates_'  + case + '_' + str(maxPathNum) + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + f'{Nr[0]}_{Nr[1]}_{Nr[2]}_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv'
classified_file = os.path.join(savedatadir, classified_images_file)

# template_image_file = 'template_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv'
template_image_file = 'template_2templates_' + case + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + f'{Nr[0]}_{Nr[1]}_{Nr[2]}_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv'
template_file = os.path.join(savedatadir, template_image_file)
# template_folder = 'template_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(dist_threshold)
template_folder1 = 'template_1_' + case + '_' + str(maxPathNum) + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + f'{Nr[0]}_{Nr[1]}_{Nr[2]}_' + str(sim_threshold) + '_' + str(dist_threshold)
template_folder2 = 'template_2_' + case + '_' + str(maxPathNum) + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + f'{Nr[0]}_{Nr[1]}_{Nr[2]}_' + str(sim_threshold) + '_' + str(dist_threshold)
template_folder_loc1 = os.path.join(savedatadir, template_folder1)
template_folder_loc2 = os.path.join(savedatadir, template_folder2)
template_result_file = 'templateResults_' + case + '_' + str(maxPathNum) + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + f'{Nr[0]}_{Nr[1]}_{Nr[2]}_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv'
template_result_file_aftersimilarity = os.path.join(savedatadir, 'templateResults_after_' + case + '_' + str(maxPathNum) + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + f'{Nr[0]}_{Nr[1]}_{Nr[2]}_' + str(sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(dist_threshold) + '.csv')
template_result_file_aftersimilarity_kmeans = os.path.join(savedatadir, 'templateResults_after_kmeans_' + case + '_' + str(maxPathNum) + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(dist_threshold) + '.csv')
template_result_file_finalCom = os.path.join(savedatadir, 'templateResults_final_'+ case + '_' + str(maxPathNum) + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + f'{Nr[0]}_{Nr[1]}_{Nr[2]}_' + str(sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(dist_threshold) + '.csv')
template_result_file_finalCom_largerthan10 = os.path.join(savedatadir, 'templateResults_final_' + str(num_tau) + '_' + case + '_' + str(maxPathNum) + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + f'{Nr[0]}_{Nr[1]}_{Nr[2]}_' + str(sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(dist_threshold) + '.csv')
distribution_folder = 'distribution_' + case + '_' + str(maxPathNum) + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + f'{Nr[0]}_{Nr[1]}_{Nr[2]}_' + str(sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(dist_threshold)
distribution_folder_loc = os.path.join('/workspace/wbh/DeepMIMO-5GNR-localpycharm/channel/', distribution_folder)
#for training and testing
# train_path = os.path.join('/workspace/wbh/DeepMIMO-5GNR-localpycharm/data/label/', 'train_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(dist_threshold) + '.csv') #00032
# test_path = os.path.join('/workspace/wbh/DeepMIMO-5GNR-localpycharm/data/label/', 'test_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(dist_threshold) + '.csv') #00032
train_path = os.path.join(savedatadir, 'train_' + case + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(dist_threshold) + '.csv')
test_path = os.path.join(savedatadir, 'test_' + case + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(dist_threshold) + '.csv')
#different from the jupyter/workspace/wbh/lightly/templateResults_final_10_8_16_2287_0.99_0.99_100.csv not found.
# template_result_file_finalCom_largerthan10_loc = 'target_' + case + f'_{Nr[0]}_{Nr[1]}_{Nr[2]}_{Nt[0]}_{Nt[1]}_{Nt[2]}_' + str(num_images)

saveImagesforOverleaf = os.path.join(saveImagesforOverleafdir, 'ScenesVisualize_' + case + '_' + str(maxPathNum) + '_' + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + f'{Nr[0]}_{Nr[1]}_{Nr[2]}_' + str(sim_threshold) + '_' + str(sim_threshold_betweenTemp) + '_' + str(dist_threshold) + '.pdf')