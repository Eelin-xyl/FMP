import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2


def read_data(file_list, path, color_queue):

    folder_path = path

    for target in file_list:

        if target == 'car10':
            continue

        path = folder_path + '/' + target

        # color_img info
        color_path = '/'.join([path, 'color'])
        color_list = os.listdir(color_path)
        color_list.sort()

        # length of each target_img
        img_num = len(color_list)

        # SelectROI
        first_image = cv2.imread(os.path.join(color_path, color_list[0]))
        roi = cv2.selectROI('SelectROI', first_image, True, False)
        # cv2.destroyWindow('SelectROI')

        for idx in range(img_num):

            # get color and ir picture by cv2
            color_img = os.path.join(color_path, color_list[idx])
            color_image = cv2.imread(color_img)

            color_queue.put((color_image, roi))
