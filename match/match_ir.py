import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
from tools import covert_img, sensor_miss_area


def track_ir(tracker_model, ir_queue, ir_res_queue, tmp_queue):

    ir_tracker = tracker_model()

    while True:

        if not ir_queue.empty() and not tmp_queue.empty():

            ir_image = ir_queue.get()
            ir_res_image = ir_image.copy()
            tmp_image, gt_val = tmp_queue.get()

            ir_image = covert_img(ir_image)
            exp_val = sensor_miss_area(ir_image, gt_val)
            ir_image = ir_image[exp_val[0][1]:exp_val[1][1], exp_val[0][0]:exp_val[1][0]]

            w, h = tmp_image.shape[::-1]
            res = cv2.matchTemplate(ir_image, tmp_image, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            # （如果模板方法是平方差或者归一化平方差，要用min_loc）
            bottom_right = (top_left[0] + w, top_left[1] + h)
            # figure out the relative coordinate
            top_left = (top_left[0] + exp_val[0][0], top_left[1] + exp_val[0][1])
            bottom_right = (bottom_right[0] + exp_val[0][0], bottom_right[1] + exp_val[0][1])
            cv2.rectangle(ir_res_image, top_left, bottom_right, (0, 255, 255), 2)
            cv2.rectangle(ir_res_image, (exp_val[0][0], exp_val[0][1]), (exp_val[1][0], exp_val[1][1]),
                          (255, 0, 0), thickness=2)

            ir_res_queue.put(ir_res_image)
