import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2


def is_cross(m, n):
    # m = [
    #         (x1, y1),     LeftTop
    #         (x2, y2)      RightBottom
    #     ]
    # n = [
    #         (a1, b1),     LeftTop
    #         (a2, b2)      RightBottom
    #     ]

    x1 = m[0][0]
    y1 = m[0][1]
    x2 = m[1][0]
    y2 = m[1][1]
    a1 = n[0][0]
    b1 = n[0][1]
    a2 = n[1][0]
    b2 = n[1][1]

    if x2 < a1 or a2 < x1:
        return False

    if y2 < b1 or b2 < y1:
        return False

    return True

    # w = [m[0][0], m[1][0], n[0][0], n[1][0]]
    # w.sort()
    # h = [m[0][1], m[1][1], n[0][1], n[1][1]]
    # h.sort()
    #
    # return abs(w[1] - w[2]) * abs(h[1] - h[2])


def cover_img(raw_img):

    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    raw_img = cv2.adaptiveThreshold(raw_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                                    5, 3)

    return raw_img


def sensor_miss_area(ir_image, gt_val):

    expand = 1
    x1 = max(0, int(gt_val[0][0] - (gt_val[1][0] - gt_val[0][0]) / 2 * expand))
    y1 = max(0, int(gt_val[0][1] - (gt_val[1][1] - gt_val[0][1]) / 2 * expand))
    x2 = min(ir_image.shape[0], int(gt_val[1][0] + (gt_val[1][0] - gt_val[0][0]) / 2 * expand))
    y2 = min(ir_image.shape[1], int(gt_val[1][1] + (gt_val[1][1] - gt_val[0][1]) / 2 * expand))
    # target_area = ir_image[y1:y2, x1:x2]
    exp_val = ((x1, y1), (x2, y2))

    return exp_val


def show_res(tracker_name, color_res_queue, ir_res_queue):

    while True:

        if not color_res_queue.empty() and not ir_res_queue.empty():

            color_res = color_res_queue.get()
            ir_res = ir_res_queue.get()
            cv2.imshow(tracker_name + ' - color', color_res)
            cv2.imshow(tracker_name + ' - ir', ir_res)
            cv2.waitKey(40)
