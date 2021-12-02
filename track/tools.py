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


def show_res(tracker_name, color_res_queue):

    while True:

        if not color_res_queue.empty():
            color_img = color_res_queue.get()
            # cv2.imshow(tracker_name + ' - color', color_img)
            # cv2.waitKey(40)
