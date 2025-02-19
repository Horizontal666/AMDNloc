'''1st
Define the path to the image folder.
Define the template as a 32x32 white square.
Define the threshold for considering two images to be of the same class (in this case, 0.9).
Initialize an empty list to store the classified images.
Loop through all the image files in the folder.
For each image, find the best match between the template and the image using the cv2.matchTemplate() function.
If the maximum correlation coefficient is above the threshold, add the image to the list of classified images.
Save the list of classified images to a CSV file using the csv module.
'''
# import os
# import cv2
# import numpy as np
# import csv
#
# # Define the path to the image folder
# path = '/workspace/wbh/DeepMIMO_python/channel/withPhase_32*64'
#
# # Define the template
# template = np.zeros((32, 32), dtype=np.uint8)
# template[...] = 255
#
# # Define the threshold for considering two images to be of the same class
# threshold = 0.7
#
# # Initialize the list of classified images
# classified_images = []
#
# # Loop through all the image files in the folder
# for filename in os.listdir(path):
#     if filename.endswith('.png'):
#         # Load the image
#         img = cv2.imread(os.path.join(path, filename), cv2.IMREAD_GRAYSCALE)
#
#         # Find the best match between the template and the image
#         result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#         _, max_val, _, _ = cv2.minMaxLoc(result)
#
#         # If the maximum correlation coefficient is above the threshold, add the image to the list of classified images
#         if max_val >= threshold:
#             classified_images.append((filename, 'class%d' % len(classified_images)))
#
# # Save the list of classified images to a CSV file
# with open('classified_images.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['filename', 'class'])
#     writer.writerows(classified_images)

'''2nd
Here's what the modified code does:

Define the path to the image folder.
Define the template size as a tuple (32, 32).
Define the threshold for considering two images to be of the same class (in this case, 0.9).
Initialize an empty list to store the classified images.
Loop through all the image files in the folder.
For each image, initialize the template to be the first 32 rows and columns of the image.
Check if the image has already been classified as part of another class.
For each existing class, load the image for the class and check if the template matches using the cv2.matchTemplate() function.
If the maximum correlation coefficient is above the threshold, classify the current image as the current class and break out of the loop.
If the image has not been classified, create a new class for it.
Save the list of classified images to a CSV file using the csv module.
'''
# import os
# import cv2
# import numpy as np
# import csv
#
# # Define the path to the image folder
# path = '/workspace/wbh/DeepMIMO_python/channel/withPhase_32*64_1500'
#
# # Define the template size
# template_size = (16, 16)
#
# # Define the threshold for considering two images to be of the same class
# threshold = 0.9
#
# # Initialize the list of classified images
# classified_images = []
#
# # Loop through all the image files in the folder
# path_list = os.listdir(path)
# path_list.sort(key=lambda x:int(x.split('.')[0])) #对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
# for filename in path_list:
# # if filename.endswith('.png'):
#     # Load the image
#     img = cv2.imread(os.path.join(path, filename))
#
#     # Initialize the template to be the first 32 rows and columns of the image
#     template = img[:template_size[0], :template_size[1]]
#
#     # Flag to indicate if the image has been classified
#     classified = False
#
#     # Check if the image has already been classified as part of another class
#     for i in range(len(classified_images)):
#         # Load the image for the current class
#         class_img = cv2.imread(os.path.join(path, classified_images[i][0]))
#
#         # Check if the template matches the current class image
#         result = cv2.matchTemplate(class_img, template, cv2.TM_CCOEFF_NORMED)
#         _, max_val, _, _ = cv2.minMaxLoc(result)
#
#         # If the maximum correlation coefficient is above the threshold, classify the current image as the current class
#         if max_val >= threshold:
#             classified_images.append((filename, classified_images[i][1]))
#             classified = True
#             break
#
#     # If the image has not been classified, create a new class for it
#     if not classified:
#         classified_images.append((filename, '%d' % len(classified_images)))
#
# # Save the list of classified images to a CSV file
# with open('classified_images.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['filename', 'class'])
#     writer.writerows(classified_images)


'''3rd
following 2nd
注意：这里没有加入连续两个template没有匹配的判断
Here's what the modified code does:

Define the path to the image folder.
Define the path to the target file.
Define the template size as a tuple (32, 32).
Define the threshold for considering two images to be of the same class (in this case, 1).
Load the target coordinates from the target file into a list of tuples.
Initialize an empty list to store the classified images.
Loop through all the image files in the folder.
For each image, initialize the template to be the first 32 rows and columns of the image.
Load the target coordinates for the
'''
# import os
# import cv2
# import numpy as np
# import csv
#
# # Define the path to the image folder
# path = '/workspace/wbh/DeepMIMO_python/channel/withPhase_32*64_1500'
#
# # Define the number of images to classify
# path_list = os.listdir(path)
# path_list.sort(key=lambda x:int(x.split('.')[0])) #对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
# num_images = len(path_list)
#
# #load the thresholds
# sim_threshold = 0.55
# dist_threshold = 0.7
# stop_threshold = num_images+500
#
# # Initialize the list of classified images
# classified_images = [stop_threshold for x in range(0, num_images)]
# classified = []
#
#
# # Load the target coordinates
# with open('/workspace/wbh/DeepMIMO_python/data/label/target.txt', 'r') as f:
#     lines = f.readlines()
#     targets = []
#     for line in lines:
#         line = line.strip().split(',')
#         targets.append((float(line[1]), float(line[2])))
#
#
# # Loop through the first num_images image files in the folder
# class_num = 0
#
# # classify images
# for i,imgname in enumerate(path_list):
# # for i in range(num_images):
#     # Define the template for this iteration
#     template = None
#     if i == 0:
#         template = cv2.imread(os.path.join(path, imgname))[:32, :32]
#
#     else:
#         # prev_class = classified_images[i][1]
#         # prev_filename = classified_images[i-1][0]
#         if classified_images[i] != stop_threshold:
#             continue
#         template = cv2.imread(os.path.join(path, imgname))[:32, :32]
#
#     # Load the target coordinates for the template
#     class_x, class_y = targets[i]
#
#     # Put the template in the csv file
#     classified_images[i] = class_num
#     classified.append((imgname, class_num, class_x, class_y, 0))#if max_val == 0, then it is template
#
#     # Loop through all the remaining image files in the folder
#     for j in range(i+1, num_images):
#         if classified_images[j] != stop_threshold:
#             continue
#         # Load the image
#         filename = path_list[j]
#         img = cv2.imread(os.path.join(path, filename)) #32*64
#
#         # Find the best match between the template and the image
#         result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#         _, max_val, _, _ = cv2.minMaxLoc(result)
#
#         # If the maximum correlation coefficient is above the threshold, add the image to the list of similar images
#         if max_val >= sim_threshold:
#             # Load the target coordinates for the current image
#             x, y = targets[j]
#
#             # Check if the l2-norm between the template coordinates and the class coordinates is less than the threshold
#             dist = np.sqrt((x - class_x) ** 2 + (y - class_y) ** 2)
#
#             # If the distance is less than the threshold, classify the current image as the current class
#             if dist < dist_threshold:
#                 classified_images[j] = class_num
#                 classified.append((filename, class_num, x, y, max_val))
#     class_num += 1
#                 # similar_images.append(filename)
#
#     # # If there are at least 1499 similar images, classify them as the same class and add them to the list of classified images
#     # if len(similar_images) >= 1499:
#     #     classified_images.append(('1.png' if i == 0 else prev_filename, 'class%d' % (i+1)))
#     #     for filename in similar_images:
#     #         classified_images.append((filename, 'class%d' % (i+1)))
#     # else:
#     #     classified_images.append(('1.png' if i == 0 else prev_filename, 'not classified'))
#
# # Save the list of classified images to a CSV file
# classified_images_file = 'classified_images_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv'
# with open(classified_images_file, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['filename', 'class', 'x', 'y', 'max_val'])
#     writer.writerows(classified)



# '''4th
# following 3rd
# 这个是一度使用最多的版本，这个版本是只计算对比了一个template的max_val。如果max_val大于阈值，就认为是同一个类别，然后计算l2距离，如果l2距离小于阈值，就认为是同一个类别。
# Here's what the modified code does:
#
# Define the path to the image folder.
# Define the path to the target file.
# Define the template size as a tuple (32, 32).
# Define the threshold for considering two images to be of the same class (in this case, 1).
# Load the target coordinates from the target file into a list of tuples.
# Initialize an empty list to store the classified images.
# Loop through all the image files in the folder.
# For each image, initialize the template to be the first 32 rows and columns of the image.
# Load the target coordinates for the
# '''
# import os
# import cv2
# import numpy as np
# import csv
# import pandas as pd
#
# # Define the path to the image folder
# path = '/workspace/wbh/DeepMIMO_python/channel/withPhase_32*64_1500'
#
# # Define the number of images to classify
# path_list = os.listdir(path)
# path_list.sort(key=lambda x:int(x.split('.')[0])) #对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
# num_images = len(path_list)
#
# #load the thresholds
# sim_threshold = 0.96
# dist_threshold = 100 #100 means no distance limit
# stop_threshold = num_images+500
# template_size_x = 8
# template_size_y = 16
#
# # Define the csv images
# classified_images_file = 'classified_images_'  + str(template_size_x) + '_' + str(template_size_y) + '_' + str(num_images) + '_' + str(sim_threshold) + '_' + str(dist_threshold) + '.csv'
# classified_file = os.path.join('/workspace/wbh/lightly', classified_images_file)
#
#
# # Initialize the list of classified images
# classified_images = [stop_threshold for x in range(0, num_images)]
# ifTemplate = [0 for x in range(0, num_images)]
# classified = []
#
#
# # Load the target coordinates
# with open('/workspace/wbh/DeepMIMO_python/data/label/target_1500.txt', 'r') as f:
#     lines = f.readlines()
#     targets = []
#     for line in lines:
#         line = line.strip().split(',')
#         targets.append((float(line[1]), float(line[2])))
#
#
# # Loop through the first num_images image files in the folder
# class_num = 0
#
# # classify images
# for i,imgname in enumerate(path_list):
# # for i in range(num_images):
#     # Define the template for this iteration
#     template = None
#     # Define whether template class match images
#     templateClassified = False
#     if i == 0:
#         template = cv2.imread(os.path.join(path, imgname))[:template_size_x, :template_size_y]
#
#
#     else:
#         # prev_class = classified_images[i][1]
#         # prev_filename = classified_images[i-1][0]
#         if classified_images[i] != stop_threshold:
#             continue
#         template = cv2.imread(os.path.join(path, imgname))[:template_size_x, :template_size_y]
#
#     # if template
#     ifTemplate[i] = 1
#
#     # Load the target coordinates for the template
#     class_x, class_y = targets[i]
#
#     # Put the template in the csv file
#     classified_images[i] = class_num
#     classified.append([imgname, class_num, class_x, class_y, 0, ifTemplate[i]])#if max_val == 0, then it is template
#
#     # Loop through all the remaining image files in the folder
#     for j in range(i+1, num_images):
#         if classified_images[j] != stop_threshold:
#             continue
#         # Load the image
#         filename = path_list[j]
#         img = cv2.imread(os.path.join(path, filename)) #32*64
#
#         # Find the best match between the template and the image
#         result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#         _, max_val, _, _ = cv2.minMaxLoc(result)
#
#
#         # If the maximum correlation coefficient is above the threshold, add the image to the list of similar images
#         if max_val >= sim_threshold:
#             # Load the target coordinates for the current image
#             x, y = targets[j]
#
#             # Check if the l2-norm between the template coordinates and the class coordinates is less than the threshold
#             dist = np.sqrt((x - class_x) ** 2 + (y - class_y) ** 2)
#
#             # If the distance is less than the threshold, classify the current image as the current class
#             if dist < dist_threshold:
#                 classified_images[j] = class_num
#                 classified.append([filename, class_num, x, y, max_val, ifTemplate[j]])
#                 templateClassified = True
#
#     if templateClassified == False:
#         for k in range(len(classified) - 1): #escape the last line
#             filename = classified[k][0]
#             img = cv2.imread(os.path.join(path, filename))  # 32*64
#
#             # Find the best match between the template and the image
#             result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#             _, max_val, _, _ = cv2.minMaxLoc(result)
#
#             # If the maximum correlation coefficient is above the threshold, add the image to the list of similar images
#             if max_val >= sim_threshold - 0.1:#if it only considers the max_val, it should add the -0.1
#                 # Load the target coordinates for the current image
#                 x = classified[k][2]
#                 y = classified[k][3]
#
#                 # Check if the l2-norm between the template coordinates and the class coordinates is less than the threshold
#                 dist = np.sqrt((x - class_x) ** 2 + (y - class_y) ** 2)
#
#                 # If the distance is less than the threshold, classify the current image as the current class
#                 if dist < dist_threshold - 0.1:
#                     # it is template
#                     classified[k][5] = 2 #ifTemplate
#                     # it is classified
#                     classified[-1][5] = 3
#                     classified[-1][1] = classified[k][1]
#                     classified[-1][4] = max_val
#                     templateClassified = True
#                     break
#
#
#     ##very good program but hear is dealing with the csv rather than classified
#     # #if there is no matching images of the template
#     # if templateClassified == False:
#     #     with open(classified_file, 'r+') as f:
#     #         reader = csv.reader(f)
#     #         lines = list(reader)
#     #         for i, cols in enumerate(lines):
#     #             # Load the image
#     #             filename = cols[0]
#     #             img = cv2.imread(os.path.join(path, filename))  # 32*64
#     #
#     #             # Find the best match between the template and the image
#     #             result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#     #             _, max_val, _, _ = cv2.minMaxLoc(result)
#     #
#     #             # If the maximum correlation coefficient is above the threshold, add the image to the list of similar images
#     #             if max_val >= sim_threshold:
#     #                 # Load the target coordinates for the current image
#     #                 x = cols[2]
#     #                 y = cols[3]
#     #
#     #                 # Check if the l2-norm between the template coordinates and the class coordinates is less than the threshold
#     #                 dist = np.sqrt((x - class_x) ** 2 + (y - class_y) ** 2)
#     #
#     #                 # If the distance is less than the threshold, classify the current image as the current class
#     #                 if dist < dist_threshold - 0.1:
#     #                     # it is template
#     #                     cols[5] = '1' #ifTemplate
#     #                     # it is classified
#     #                     lines[-1][5] = '2'
#     #                     lines[-1][1] = cols[1]
#     #                     break
#     #     # Move the file pointer to the beginning of the file
#     #     f.seek(0)
#     #
#     #     # Write the updated rows back to the file
#     #     writer = csv.writer(f)
#     #     writer.writerows(lines)
#     #
#     #     # Truncate the file to remove any remaining rows
#     #     f.truncate()
#
#
#     class_num += 1
#                 # similar_images.append(filename)
#
#     # # If there are at least 1499 similar images, classify them as the same class and add them to the list of classified images
#     # if len(similar_images) >= 1499:
#     #     classified_images.append(('1.png' if i == 0 else prev_filename, 'class%d' % (i+1)))
#     #     for filename in similar_images:
#     #         classified_images.append((filename, 'class%d' % (i+1)))
#     # else:
#     #     classified_images.append(('1.png' if i == 0 else prev_filename, 'not classified'))
#
# # Save the list of classified images to a CSV file
# # with open(classified_images_file, 'w', newline='') as csvfile:
# #     writer = csv.writer(csvfile)
# #     writer.writerow(['filename', 'class', 'x', 'y', 'max_val', 'ifTemplate'])
# #     writer.writerows(classified)
#
# #将classified从list转化为dataframe结构并保存下来
# df = pd.DataFrame(classified, columns=['filename', 'class', 'x', 'y', 'max_val', 'ifTemplate'])
# df.to_csv(classified_images_file, index=False)
#
# #读取csv文件并转化为dataframe结构
# df = pd.read_csv(classified_images_file)
# #将dataframe结构转化为list
# classified = df.values.tolist()


'''5th-1 NLOS的计算都是用这个版本
following 4th
这个版本是将一个图片作为模板后，其他图片与该图的前后16行16列分别计算max_val，如果2个max_val都大于某一个值，就认为是同一个类别。
可以调整的逻辑是：
1. 距离的限制if dist < dist_threshold:
2. 
可以调整的参数是：
1. 16行16列的大小
2. 2个max_val的阈值
3. 2个max_val的阈值的差值
Here's what the modified code does:

Define the path to the image folder.
Define the path to the target file.
Define the template size as a tuple (32, 32).
Define the threshold for considering two images to be of the same class (in this case, 1).
Load the target coordinates from the target file into a list of tuples.
Initialize an empty list to store the classified images.
Loop through all the image files in the folder.
For each image, initialize the template to be the first 32 rows and columns of the image.
Load the target coordinates for the
'''
import os
import cv2
import numpy as np
import csv
import pandas as pd
from tqdm import tqdm
from parameters_waird import *

# Define the path to the image folder
# path = '/workspace/wbh/DeepMIMO_python/channel/withPhase_32*64_1500'
# target_path = '/workspace/wbh/DeepMIMO_python/data/label/target_1500.txt'
# path = '/data2/wbh/DeepMIMO-5GNR/DeepMIMO_python/channel/withPhase_32*64'#100%|██████████| 58294/58294 [50:10<00:00, 19.36it/s]
# target_path = '/data2/wbh/DeepMIMO-5GNR-localpycharm/data/label/target_generated_2_00032.txt'#00032的target

# 读取图片的路径
path = generatedFolderNlos #NLOS
# path = generatedFolderlos #LOS
# path = generatedFolderNlos_SNR #NLOS SNR
print('Images Folder:', path)
target_path = targetfile_nlos #NLOS, NLOS SNR
# target_path = targetfile_los #LOS
# 存储的路径：更改parameters_waird.py当中的savedatadir


# # Define the number of images to classify
# path_list = os.listdir(path)
# # path_list.sort(key=lambda x:int(x.split('.')[0])) #对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
# num_images = len(path_list)
# define the path_list and load the target coordinates
path_list = []
targets = []
# path_list.sort(key=lambda x: int(x.split('.')[0]))  # 对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
num_images = len(os.listdir(path))
#读取数据
#txt读取方式
# for line in open(target_path, 'r', encoding="utf8").readlines():
#     line = line.strip().split(',')
#     path_list.append(line[0])
#     targets.append((float(line[2]), float(line[3])))
#csv读取方式
for i, line in pd.read_csv(target_path).iterrows():
    path_list.append(line[0])
    targets.append((float(line[3]), float(line[4])))
#load the thresholds from parameters_waird.py
# sim_threshold = 0.985 #load from parameters_waird.py
# dist_threshold = 100 #100 means no distance limit
# stop_threshold = num_images+500
# template_size_x = 4 #之前是8
# template_size_y = 8 #之前是16



# Initialize the list of classified images
classified_images = [stop_threshold for x in range(0, num_images)]
ifTemplate = [0 for x in range(0, num_images)]
classified = []


# # Load the target coordinates
# with open(target_path, 'r') as f:
# # with open('/workspace/wbh/DeepMIMO_python/data/label/target.txt', 'r') as f:
#     lines = f.readlines()
#     targets = []
#     for line in lines:
#         line = line.strip().split(',')
#         targets.append((float(line[1]), float(line[2])))


# Loop through the first num_images image files in the folder
class_num = 0

# classify images
for i, imgname in enumerate(tqdm(path_list)):#add 进度条
# for i in range(num_images):
    # Define the template for this iteration
    template = None
    # Define whether template class match images
    templateClassified = False
    if i == 0:
        template1 = cv2.imread(os.path.join(path, imgname))[:template_size_x, :template_size_y]
        template2 = cv2.imread(os.path.join(path, imgname))[-template_size_x:, -template_size_y:]


    else:
        # prev_class = classified_images[i][1]
        # prev_filename = classified_images[i-1][0]
        if classified_images[i] != stop_threshold:
            continue
        template1 = cv2.imread(os.path.join(path, imgname))[:template_size_x, :template_size_y]
        template2 = cv2.imread(os.path.join(path, imgname))[-template_size_x:, -template_size_y:]

    # if template
    ifTemplate[i] = 1

    # Load the target coordinates for the template
    class_x, class_y = targets[i]

    # Put the template in the csv file
    classified_images[i] = class_num
    classified.append([imgname, class_num, class_x, class_y, 0, 0, ifTemplate[i]])#if max_val == 0, then it is template

    # Loop through all the remaining image files in the folder
    for j in range(i+1, num_images):
        if classified_images[j] != stop_threshold:
            continue
        # Load the image
        filename = path_list[j]
        img = cv2.imread(os.path.join(path, filename)) #32*64

        # Find the best match between the template and the image
        result1 = cv2.matchTemplate(img, template1, cv2.TM_CCOEFF_NORMED)
        _, max_val1, _, _ = cv2.minMaxLoc(result1)
        result2 = cv2.matchTemplate(img, template2, cv2.TM_CCOEFF_NORMED)
        _, max_val2, _, _ = cv2.minMaxLoc(result2)


        # If the maximum correlation coefficient is above the threshold, add the image to the list of similar images
        if max_val1 >= sim_threshold and max_val2 >= sim_threshold:
            # Load the target coordinates for the current image
            x, y = targets[j]

            # Check if the l2-norm between the template coordinates and the class coordinates is less than the threshold
            dist = np.sqrt((x - class_x) ** 2 + (y - class_y) ** 2)

            # If the distance is less than the threshold, classify the current image as the current class
            if dist < dist_threshold:
                classified_images[j] = class_num
                classified.append([filename, class_num, x, y, max_val1, max_val2, ifTemplate[j]])
                templateClassified = True

    #if the template doesn't match any images, then to see the images before the template
    if templateClassified == False:
        for k in range(len(classified) - 1): #escape the last line
            filename = classified[k][0]
            img = cv2.imread(os.path.join(path, filename))  # 32*64

            # Find the best match between the template and the image
            result1 = cv2.matchTemplate(img, template1, cv2.TM_CCOEFF_NORMED)
            _, max_val1, _, _ = cv2.minMaxLoc(result1)
            result2 = cv2.matchTemplate(img, template2, cv2.TM_CCOEFF_NORMED)
            _, max_val2, _, _ = cv2.minMaxLoc(result2)

            # If the maximum correlation coefficient is above the threshold, add the image to the list of similar images
            if max_val1 >= sim_threshold and max_val2 >= sim_threshold:#if it only considers the max_val, it should add the -0.1
                # Load the target coordinates for the current image
                x = classified[k][2]
                y = classified[k][3]

                # Check if the l2-norm between the template coordinates and the class coordinates is less than the threshold
                dist = np.sqrt((x - class_x) ** 2 + (y - class_y) ** 2)

                # If the distance is less than the threshold, classify the current image as the current class
                if dist < dist_threshold - 0.1:
                    # it is template
                    classified[k][6] = 2 #ifTemplate
                    # it is classified
                    classified[-1][6] = 3
                    classified[-1][1] = classified[k][1]
                    classified[-1][4] = max_val1
                    classified[-1][5] = max_val2
                    templateClassified = True
                    break


    ##very good program but hear is dealing with the csv rather than classified
    # #if there is no matching images of the template
    # if templateClassified == False:
    #     with open(classified_file, 'r+') as f:
    #         reader = csv.reader(f)
    #         lines = list(reader)
    #         for i, cols in enumerate(lines):
    #             # Load the image
    #             filename = cols[0]
    #             img = cv2.imread(os.path.join(path, filename))  # 32*64
    #
    #             # Find the best match between the template and the image
    #             result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    #             _, max_val, _, _ = cv2.minMaxLoc(result)
    #
    #             # If the maximum correlation coefficient is above the threshold, add the image to the list of similar images
    #             if max_val >= sim_threshold:
    #                 # Load the target coordinates for the current image
    #                 x = cols[2]
    #                 y = cols[3]
    #
    #                 # Check if the l2-norm between the template coordinates and the class coordinates is less than the threshold
    #                 dist = np.sqrt((x - class_x) ** 2 + (y - class_y) ** 2)
    #
    #                 # If the distance is less than the threshold, classify the current image as the current class
    #                 if dist < dist_threshold - 0.1:
    #                     # it is template
    #                     cols[5] = '1' #ifTemplate
    #                     # it is classified
    #                     lines[-1][5] = '2'
    #                     lines[-1][1] = cols[1]
    #                     break
    #     # Move the file pointer to the beginning of the file
    #     f.seek(0)
    #
    #     # Write the updated rows back to the file
    #     writer = csv.writer(f)
    #     writer.writerows(lines)
    #
    #     # Truncate the file to remove any remaining rows
    #     f.truncate()


    class_num += 1
                # similar_images.append(filename)

    # # If there are at least 1499 similar images, classify them as the same class and add them to the list of classified images
    # if len(similar_images) >= 1499:
    #     classified_images.append(('1.png' if i == 0 else prev_filename, 'class%d' % (i+1)))
    #     for filename in similar_images:
    #         classified_images.append((filename, 'class%d' % (i+1)))
    # else:
    #     classified_images.append(('1.png' if i == 0 else prev_filename, 'not classified'))

# Save the list of classified images to a CSV file
# with open(classified_images_file, 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['filename', 'class', 'x', 'y', 'max_val', 'ifTemplate'])
#     writer.writerows(classified)

#将classified从list转化为dataframe结构并保存下来
df = pd.DataFrame(classified, columns=['filename', 'class', 'x', 'y', 'max_val_pre', 'max_val_pro', 'ifTemplate'])
if not os.path.exists(savedatadir):
    os.makedirs(savedatadir)
df.to_csv(classified_file, index=False)

#读取csv文件并转化为dataframe结构
df = pd.read_csv(classified_file)
#将dataframe结构转化为list
classified = df.values.tolist()


'''5th-2 NLOS+LOS的计算都是用这个版本?
following 4th
这个版本是将一个图片作为模板后，其他图片与该图的前后16行16列分别计算max_val，如果2个max_val都大于某一个值，就认为是同一个类别。
可以调整的逻辑是：
1. 距离的限制if dist < dist_threshold:
2. 
可以调整的参数是：
1. 16行16列的大小
2. 2个max_val的阈值
3. 2个max_val的阈值的差值
Here's what the modified code does:

Define the path to the image folder.
Define the path to the target file.
Define the template size as a tuple (32, 32).
Define the threshold for considering two images to be of the same class (in this case, 1).
Load the target coordinates from the target file into a list of tuples.
Initialize an empty list to store the classified images.
Loop through all the image files in the folder.
For each image, initialize the template to be the first 32 rows and columns of the image.
Load the target coordinates for the
'''
# import os
# import cv2
# import numpy as np
# import csv
# import pandas as pd
# from tqdm import tqdm
# from parameters_waird import *
#
# # Define the path to the image folder
# # path = '/workspace/wbh/DeepMIMO_python/channel/withPhase_32*64_1500'
# # target_path = '/workspace/wbh/DeepMIMO_python/data/label/target_1500.txt'
# # path = '/data2/wbh/DeepMIMO-5GNR/DeepMIMO_python/channel/withPhase_32*64'#100%|██████████| 58294/58294 [50:10<00:00, 19.36it/s]
# # target_path = '/data2/wbh/DeepMIMO-5GNR-localpycharm/data/label/target_generated_2_00032.txt'#00032的target
# # path = generatedFolderNlos #NLOS
# # target_path = targetfile_nlos
# path = generatedFolder #LOS
# target_path = targetfile
#
#
#
# # # Define the number of images to classify
# # path_list = os.listdir(path)
# # # path_list.sort(key=lambda x:int(x.split('.')[0])) #对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
# # num_images = len(path_list)
# # define the path_list and load the target coordinates
# path_list = []
# targets = []
# # path_list.sort(key=lambda x: int(x.split('.')[0]))  # 对‘.’进行切片，并取列表的第一个值（左边的文件名）转化整数型
# num_images = len(os.listdir(path))
# #读取数据
# #txt读取方式
# # for line in open(target_path, 'r', encoding="utf8").readlines():
# #     line = line.strip().split(',')
# #     path_list.append(line[0])
# #     targets.append((float(line[2]), float(line[3])))
# #csv读取方式
# for i, line in pd.read_csv(target_path).iterrows():
#     path_list.append(line[0])
#     targets.append((float(line[3]), float(line[4])))
# #load the thresholds
# sim_threshold = 0.99
# dist_threshold = 100 #100 means no distance limit
# stop_threshold = num_images+500
# template_size_x = 4
# template_size_y = 8
#
#
#
# # Initialize the list of classified images
# classified_images = [stop_threshold for x in range(0, num_images)]
# ifTemplate = [0 for x in range(0, num_images)]
# classified = []
#
#
# # # Load the target coordinates
# # with open(target_path, 'r') as f:
# # # with open('/workspace/wbh/DeepMIMO_python/data/label/target.txt', 'r') as f:
# #     lines = f.readlines()
# #     targets = []
# #     for line in lines:
# #         line = line.strip().split(',')
# #         targets.append((float(line[1]), float(line[2])))
#
#
# # Loop through the first num_images image files in the folder
# class_num = 0
#
# # classify images
# for i, imgname in enumerate(tqdm(path_list)):#add 进度条
# # for i in range(num_images):
#     # Define the template for this iteration
#     template = None
#     # Define whether template class match images
#     templateClassified = False
#     if i == 0:
#         template1 = cv2.imread(os.path.join(path, imgname))[:template_size_x, :template_size_y]
#         template2 = cv2.imread(os.path.join(path, imgname))[-template_size_x:, -template_size_y:]
#
#
#     else:
#         # prev_class = classified_images[i][1]
#         # prev_filename = classified_images[i-1][0]
#         if classified_images[i] != stop_threshold:
#             continue
#         template1 = cv2.imread(os.path.join(path, imgname))[:template_size_x, :template_size_y]
#         template2 = cv2.imread(os.path.join(path, imgname))[-template_size_x:, -template_size_y:]
#
#     # if template
#     ifTemplate[i] = 1
#
#     # Load the target coordinates for the template
#     class_x, class_y = targets[i]
#
#     # Put the template in the csv file
#     classified_images[i] = class_num
#     classified.append([imgname, class_num, class_x, class_y, 0, 0, ifTemplate[i]])#if max_val == 0, then it is template
#
#     # Loop through all the remaining image files in the folder
#     for j in range(i+1, num_images):
#         if classified_images[j] != stop_threshold:
#             continue
#         # Load the image
#         filename = path_list[j]
#         img = cv2.imread(os.path.join(path, filename)) #32*64
#
#         # Find the best match between the template and the image
#         result1 = cv2.matchTemplate(img, template1, cv2.TM_CCOEFF_NORMED)
#         _, max_val1, _, _ = cv2.minMaxLoc(result1)
#         result2 = cv2.matchTemplate(img, template2, cv2.TM_CCOEFF_NORMED)
#         _, max_val2, _, _ = cv2.minMaxLoc(result2)
#
#
#         # If the maximum correlation coefficient is above the threshold, add the image to the list of similar images
#         if max_val1 >= sim_threshold and max_val2 >= sim_threshold:
#             # Load the target coordinates for the current image
#             x, y = targets[j]
#
#             # Check if the l2-norm between the template coordinates and the class coordinates is less than the threshold
#             dist = np.sqrt((x - class_x) ** 2 + (y - class_y) ** 2)
#
#             # If the distance is less than the threshold, classify the current image as the current class
#             if dist < dist_threshold:
#                 classified_images[j] = class_num
#                 classified.append([filename, class_num, x, y, max_val1, max_val2, ifTemplate[j]])
#                 templateClassified = True
#
#     #if the template doesn't match any images, then to see the images before the template
#     if templateClassified == False:
#         for k in range(len(classified) - 1): #escape the last line
#             filename = classified[k][0]
#             img = cv2.imread(os.path.join(path, filename))  # 32*64
#
#             # Find the best match between the template and the image
#             result1 = cv2.matchTemplate(img, template1, cv2.TM_CCOEFF_NORMED)
#             _, max_val1, _, _ = cv2.minMaxLoc(result1)
#             result2 = cv2.matchTemplate(img, template2, cv2.TM_CCOEFF_NORMED)
#             _, max_val2, _, _ = cv2.minMaxLoc(result2)
#
#             # If the maximum correlation coefficient is above the threshold, add the image to the list of similar images
#             if max_val1 >= sim_threshold and max_val2 >= sim_threshold:#if it only considers the max_val, it should add the -0.1
#                 # Load the target coordinates for the current image
#                 x = classified[k][2]
#                 y = classified[k][3]
#
#                 # Check if the l2-norm between the template coordinates and the class coordinates is less than the threshold
#                 dist = np.sqrt((x - class_x) ** 2 + (y - class_y) ** 2)
#
#                 # If the distance is less than the threshold, classify the current image as the current class
#                 if dist < dist_threshold - 0.1:
#                     # it is template
#                     classified[k][6] = 2 #ifTemplate
#                     # it is classified
#                     classified[-1][6] = 3
#                     classified[-1][1] = classified[k][1]
#                     classified[-1][4] = max_val1
#                     classified[-1][5] = max_val2
#                     templateClassified = True
#                     break
#
#
#     ##very good program but hear is dealing with the csv rather than classified
#     # #if there is no matching images of the template
#     # if templateClassified == False:
#     #     with open(classified_file, 'r+') as f:
#     #         reader = csv.reader(f)
#     #         lines = list(reader)
#     #         for i, cols in enumerate(lines):
#     #             # Load the image
#     #             filename = cols[0]
#     #             img = cv2.imread(os.path.join(path, filename))  # 32*64
#     #
#     #             # Find the best match between the template and the image
#     #             result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
#     #             _, max_val, _, _ = cv2.minMaxLoc(result)
#     #
#     #             # If the maximum correlation coefficient is above the threshold, add the image to the list of similar images
#     #             if max_val >= sim_threshold:
#     #                 # Load the target coordinates for the current image
#     #                 x = cols[2]
#     #                 y = cols[3]
#     #
#     #                 # Check if the l2-norm between the template coordinates and the class coordinates is less than the threshold
#     #                 dist = np.sqrt((x - class_x) ** 2 + (y - class_y) ** 2)
#     #
#     #                 # If the distance is less than the threshold, classify the current image as the current class
#     #                 if dist < dist_threshold - 0.1:
#     #                     # it is template
#     #                     cols[5] = '1' #ifTemplate
#     #                     # it is classified
#     #                     lines[-1][5] = '2'
#     #                     lines[-1][1] = cols[1]
#     #                     break
#     #     # Move the file pointer to the beginning of the file
#     #     f.seek(0)
#     #
#     #     # Write the updated rows back to the file
#     #     writer = csv.writer(f)
#     #     writer.writerows(lines)
#     #
#     #     # Truncate the file to remove any remaining rows
#     #     f.truncate()
#
#
#     class_num += 1
#                 # similar_images.append(filename)
#
#     # # If there are at least 1499 similar images, classify them as the same class and add them to the list of classified images
#     # if len(similar_images) >= 1499:
#     #     classified_images.append(('1.png' if i == 0 else prev_filename, 'class%d' % (i+1)))
#     #     for filename in similar_images:
#     #         classified_images.append((filename, 'class%d' % (i+1)))
#     # else:
#     #     classified_images.append(('1.png' if i == 0 else prev_filename, 'not classified'))
#
# # Save the list of classified images to a CSV file
# # with open(classified_images_file, 'w', newline='') as csvfile:
# #     writer = csv.writer(csvfile)
# #     writer.writerow(['filename', 'class', 'x', 'y', 'max_val', 'ifTemplate'])
# #     writer.writerows(classified)
#
# #将classified从list转化为dataframe结构并保存下来
# df = pd.DataFrame(classified, columns=['filename', 'class', 'x', 'y', 'max_val_pre', 'max_val_pro', 'ifTemplate'])
# df.to_csv(classified_file, index=False)
#
# #读取csv文件并转化为dataframe结构
# df = pd.read_csv(classified_file)
# #将dataframe结构转化为list
# classified = df.values.tolist()





