import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2


def read_data(file_list, path, color_queue, ir_queue):

    folder_path = path
    # init = False
    bypass = ['car10', 'carLight']

    for scene in file_list:

        if scene in bypass:
            continue

        # if scene != 'caraftertree':
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

        # ir_img info
        ir_path = '/'.join([path, 'ir'])
        ir_list = os.listdir(ir_path)
        ir_list.sort()

        # SelectROI
        first_image = cv2.imread(os.path.join(color_path, color_list[0]))
        roi = cv2.selectROI('SelectROI', first_image, True, False)
        cv2.destroyWindow('SelectROI')

        for idx in range(img_num):

            # get color and ir picture by cv2
            color_img = os.path.join(color_path, color_list[idx])
            color_image = cv2.imread(color_img)

            ir_img = os.path.join(ir_path, ir_list[idx])
            ir_image = cv2.imread(ir_img)

            color_queue.put((color_image, roi))
            ir_queue.put(ir_image)
