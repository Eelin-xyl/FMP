import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
from tools import covert_img, sensor_miss_area, intersection, cal_iou


def matcher_ir(matcher_model, ir_queue, ir_res_queue, tmp_queue):

    ir_matcher = matcher_model
    sum_iou = 0
    count_iou = 0

    while True:

        if not ir_queue.empty() and not tmp_queue.empty():

            ir_image = ir_queue.get()
            tmp_image, gt_val = tmp_queue.get()

            exp_val = sensor_miss_area((ir_image.shape[0], ir_image.shape[1]), gt_val)
            area_image = ir_image[exp_val[0][1]:exp_val[1][1], exp_val[0][0]:exp_val[1][0]]
            raw_image = area_image.copy()
            area_image = covert_img(area_image)

            h, w = tmp_image.shape[:2]

            res = cv2.matchTemplate(area_image, tmp_image, ir_matcher)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

            # if match_tmp method is SQDIFF or SQDIFF_NORMED, using min_loc
            top_left = max_loc
            bottom_right = (top_left[0] + w, top_left[1] + h)

            # figure out the relative coordinate
            top_left = (top_left[0] + exp_val[0][0], top_left[1] + exp_val[0][1])
            bottom_right = (bottom_right[0] + exp_val[0][0], bottom_right[1] + exp_val[0][1])
            cv2.rectangle(ir_image, top_left, bottom_right, (0, 255, 255), 2)
            cv2.rectangle(ir_image, (exp_val[0][0], exp_val[0][1]), (exp_val[1][0], exp_val[1][1]),
                          (255, 0, 0), thickness=2)

            sum_iou += cal_iou((top_left, bottom_right), gt_val)
            count_iou += 1
            print(sum_iou / count_iou)

            ir_res_queue.put((ir_image, raw_image, area_image))
