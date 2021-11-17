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


def show_res(tracker_name, color_res_queue, ir_res_queue):

    while True:

        if not color_res_queue.empty() and not ir_res_queue.empty():
            cv2.imshow(tracker_name + ' - color', color_res_queue.get())
            cv2.imshow(tracker_name + ' - ir', ir_res_queue.get())
            cv2.waitKey(40)
