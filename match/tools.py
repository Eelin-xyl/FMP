import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2


def is_cross(box1, box2):

    # box1 = (
    #         (x1, y1),     LeftTop
    #         (x2, y2)      RightBottom
    #     )
    # box2 = (
    #         (a1, b1),     LeftTop
    #         (a2, b2)      RightBottom
    #     )

    (x1, y1), (x2, y2) = box1
    (a1, b1), (a2, b2) = box2

    if x2 < a1 or a2 < x1:
        return False

    if y2 < b1 or b2 < y1:
        return False

    return True


def cal_iou(box1, box2):

    # box1 = (
    #         (x1, y1),     LeftTop
    #         (x2, y2)      RightBottom
    #     )
    # box2 = (
    #         (a1, b1),     LeftTop
    #         (a2, b2)      RightBottom
    #     )

    (x1, y1), (x2, y2) = box1
    (a1, b1), (a2, b2) = box2

    # top_left of i
    i1 = max(x1, a1)
    i2 = max(y1, b1)

    # bottom_right of i
    i3 = min(x2, a2)
    i4 = min(y2, b2)

    # cal area
    s1 = (x2 - x1) * (y2 - y1)
    s2 = (a2 - a1) * (b2 - b1)

    # sum area
    s = s1 + s2

    # cal intersection
    inter_area = (i3 - i1) * (i4 - i2)

    iou = inter_area / (s - inter_area)

    return iou


def intersection(box1, box2):

    w = [box1[0][0], box1[1][0], box2[0][0], box2[1][0]]
    w.sort()
    h = [box1[0][1], box1[1][1], box2[0][1], box2[1][1]]
    h.sort()

    return abs(w[1] - w[2]) * abs(h[1] - h[2])


def covert_img(raw_img):
    raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
    # raw_img = cv2.adaptiveThreshold(raw_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 3)
    # raw_img = cv2.GaussianBlur(raw_img, (3, 3), 0)
    raw_img = cv2.Canny(raw_img, 50, 150)

    # cv2.TM_CCOEFF_NORMED
    # miou 0.3673803935096333 (only gray) 0.37293078716287253 (color without medianBlur)
    # miou 0.536817262918913 (gray and canny) 0.524499019328563 (color without medianBlur)
    # miou 0.5056186418634039 (gauss and canny) 0.5120990164005463 (color without medianBlur)
    # miou 0.4686126916064594 (adaptivaethreshould and canny) 0.45016259260479696 (color without medianBlur)

    return raw_img


def sensor_miss_area(size, gt_val):

    expand = 1
    h, w = size
    x1 = max(0, int(gt_val[0][0] - (gt_val[1][0] - gt_val[0][0]) / 2 * expand))
    y1 = max(0, int(gt_val[0][1] - (gt_val[1][1] - gt_val[0][1]) / 2 * expand))
    x2 = min(w, int(gt_val[1][0] + (gt_val[1][0] - gt_val[0][0]) / 2 * expand))
    y2 = min(h, int(gt_val[1][1] + (gt_val[1][1] - gt_val[0][1]) / 2 * expand))
    exp_val = ((x1, y1), (x2, y2))

    return exp_val


def show_res(tracker_name, matcher_name, color_res_queue, ir_res_queue):

    while True:

        if not color_res_queue.empty() and not ir_res_queue.empty():

            color_res_image, target_image, tmp_image = color_res_queue.get()
            ir_image, raw_image, area_image = ir_res_queue.get()

            cv2.imshow(tracker_name + ' - Color', color_res_image)

            # Process blank_color
            blank_color = np.zeros((300, 600, 3), np.uint8)
            blank_color.fill(0)

            k_color = 300 / max(target_image.shape[0], target_image.shape[1])
            target_image = cv2.resize(target_image, None, fx=k_color, fy=k_color, interpolation=cv2.INTER_LINEAR)
            # target_image = cv2.resize(target_image, None, fx=k_color, fy=k_color, interpolation=cv2.INTER_CUBIC)

            tmp_image = cv2.resize(tmp_image, None, fx=k_color, fy=k_color, interpolation=cv2.INTER_LINEAR)
            # tmp_image = cv2.resize(tmp_image, None, fx=k_color, fy=k_color, interpolation=cv2.INTER_CUBIC)

            blank_color[0:target_image.shape[0], 0:target_image.shape[1]] = target_image
            tmp_image = cv2.cvtColor(tmp_image, cv2.COLOR_GRAY2BGR)
            blank_color[0: tmp_image.shape[0], 300: 300 + tmp_image.shape[1]] = tmp_image

            cv2.imshow('COLOR - TARGET - TMP', blank_color)

            cv2.imshow(matcher_name + ' - Ir', ir_image)

            # Process blank_ir
            blank_ir = np.zeros((300, 600, 3), np.uint8)
            blank_ir.fill(0)

            k_ir = 300 / max(raw_image.shape[0], raw_image.shape[1])
            raw_image = cv2.resize(raw_image, None, fx=k_ir, fy=k_ir, interpolation=cv2.INTER_LINEAR)
            # raw_image = cv2.resize(raw_image, None, fx=k_ir, fy=k_ir, interpolation=cv2.INTER_CUBIC)

            area_image = cv2.resize(area_image, None, fx=k_ir, fy=k_ir, interpolation=cv2.INTER_LINEAR)
            # area_image = cv2.resize(area_image, None, fx=k_ir, fy=k_ir, interpolation=cv2.INTER_CUBIC)

            blank_ir[0:raw_image.shape[0], 0:raw_image.shape[1]] = raw_image
            area_image = cv2.cvtColor(area_image, cv2.COLOR_GRAY2BGR)
            blank_ir[0: area_image.shape[0], 300: 300 + area_image.shape[1]] = area_image

            cv2.imshow('IR - RAW - AREA', blank_ir)

            cv2.waitKey(40)
            # cv2.waitKey(-1)
