import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2


def read_data(file_list, path, color_queue, ir_queue):

    folder_path = path

    init = False

    for target in file_list:

        if target == 'car10':
            continue

        if target == 'car':
            init = True

        if not init:
            continue

        path = folder_path + '/' + target

        # color_img info
        color_path = '/'.join([path, 'color'])
        color_list = os.listdir(color_path)
        color_list.sort()

        # length of each target_img
        img_num = len(color_list)

        # ir_img info
        ir_path = '/'.join([path, 'ir'])
        ir_list = os.listdir(ir_path)
        ir_list.sort()

        gt_path = '/'.join([path, 'groundtruth.txt'])

        # import groundtruth data
        with open(gt_path, "r") as f:

            gt_file = f.read()  # Read file
            gt_val_list = gt_file.split('\n')
            # print(data)

        for idx in range(img_num):
            gt_val = gt_val_list[idx].split(',')
            gt_val = tuple([int(float(i)) for i in gt_val])

            # get color and ir picture by cv2
            color_img = os.path.join(color_path, color_list[idx])
            color_image = cv2.imread(color_img)

            ir_img = os.path.join(ir_path, ir_list[idx])
            ir_image = cv2.imread(ir_img)

            # process gt_val
            color_gt_val = ((min(gt_val[0], gt_val[4]), min(gt_val[1], gt_val[5])),
                            (max(gt_val[0], gt_val[4]), max(gt_val[1], gt_val[5])))

            color_queue.put((color_image, color_gt_val))
            ir_queue.put(ir_image)
