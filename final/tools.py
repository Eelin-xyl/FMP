import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2


def covert_img(raw_img):

    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    # raw_img = cv2.adaptiveThreshold(raw_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
    # raw_img = cv2.GaussianBlur(raw_img, (3, 3), 0)  # 用高斯滤波处理原图像降噪
    # raw_img = cv2.Canny(raw_img, 50, 150)

    return raw_img


def sensor_miss_area(ir_image, gt_val):

    expand = 1
    x1 = max(0, int(gt_val[0][0] - (gt_val[1][0] - gt_val[0][0]) / 2 * expand))
    y1 = max(0, int(gt_val[0][1] - (gt_val[1][1] - gt_val[0][1]) / 2 * expand))
    x2 = min(ir_image.shape[1], int(gt_val[1][0] + (gt_val[1][0] - gt_val[0][0]) / 2 * expand))
    y2 = min(ir_image.shape[0], int(gt_val[1][1] + (gt_val[1][1] - gt_val[0][1]) / 2 * expand))
    exp_val = ((x1, y1), (x2, y2))

    return exp_val


def show_res(tracker_name, matcher_name, color_res_queue, ir_res_queue):

    while True:

        if not color_res_queue.empty() and not ir_res_queue.empty():

            color_res = color_res_queue.get()
            ir_res = ir_res_queue.get()

            cv2.imshow(tracker_name + ' - Color', color_res[0])

            cv2.namedWindow('COLOR - Tmp', 0)
            cv2.resizeWindow('COLOR - Tmp', int(color_res[1].shape[1] * 3), int(color_res[1].shape[0] * 3))
            cv2.imshow('COLOR - Tmp', color_res[1])

            cv2.imshow(matcher_name + ' - Ir', ir_res[0])

            cv2.namedWindow('IR - Area', 0)
            cv2.resizeWindow('IR - Area', int(ir_res[1].shape[1] * 3), int(ir_res[1].shape[0] * 3))
            cv2.imshow('IR - Area', ir_res[1])

            cv2.waitKey(40)
