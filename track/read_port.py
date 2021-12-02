import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2


def read_data(file_list, path, color_queue):

    folder_path = path
    # init = False
    bypass = ['car10', 'carLight']

    for scene in file_list:

        if scene in bypass:
            continue

        # if scene != 'hotkettle':
        #     continue

        print(scene)

        # if scene == 'car':
        #     init = True

        path = folder_path + '/' + scene

        # color_img info
        color_path = '/'.join([path, 'color'])
        color_list = os.listdir(color_path)
        color_list.sort()

        # length of each target_img
        img_num = len(color_list)

        gt_path = '/'.join([path, 'groundtruth.txt'])

        # import ground_truth data
        with open(gt_path, "r") as f:

            gt_file = f.read()  # Read file
            gt_val_list = gt_file.split('\n')
            # print(data)

        for idx in range(img_num):

            gt_val = gt_val_list[idx].split(',')
            gt_val = tuple([int(float(i)) for i in gt_val])

            # get color picture
            color_img = os.path.join(color_path, color_list[idx])
            color_image = cv2.imread(color_img)

            # process gt_val
            color_gt_val = ((min(gt_val[0], gt_val[4]), min(gt_val[1], gt_val[5])),
                            (max(gt_val[0], gt_val[4]), max(gt_val[1], gt_val[5])))

            color_queue.put((color_image, color_gt_val, scene))
