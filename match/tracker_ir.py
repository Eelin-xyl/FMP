import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
from tools import cover_img, sensor_miss_area


def track_ir(tracker_model, ir_queue, ir_res_queue, label_queue):

    ir_tracker = tracker_model()

    while True:

        if not ir_queue.empty() and not label_queue.empty():

            ir_image = ir_queue.get()
            ir_res_image = ir_image.copy()
            tmp_image, gt_val = label_queue.get()

            ir_image = cover_img(ir_image)
            exp_val = sensor_miss_area(ir_image, gt_val)

            w, h = tmp_image.shape[::-1]
            res = cv2.matchTemplate(ir_image, tmp_image, cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
            top_left = max_loc
            # （如果模板方法是平方差或者归一化平方差，要用min_loc）
            bottom_right = (top_left[0] + w, top_left[1] + h)
            cv2.rectangle(ir_res_image, top_left, bottom_right, (255, 0, 0), 2)
            cv2.rectangle(ir_res_image, (exp_val[0][0], exp_val[0][1]), (exp_val[1][0], exp_val[1][1]),
                          (0, 0, 255), thickness=2)

            # cv2.rectangle(ir_image, (label_val[0][0], label_val[0][1]), (label_val[1][0], label_val[1][1]),
            #               (0, 255, 0), thickness=2)

            ir_res_queue.put(ir_res_image)
