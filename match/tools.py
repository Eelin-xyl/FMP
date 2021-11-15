import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2


def is_cross(m, n):
    # A = [
    #         (x1, y1),     LeftTop
    #         (x2, y2)      RightBottom
    #     ]
    # B = [
    #         (a1, b1),     LeftTop
    #         (a2, b2)      RightBottom
    #     ]

    x1 = min(m[0][0], m[1][0])
    y1 = min(m[0][1], m[1][1])
    x2 = max(m[0][0], m[1][0])
    y2 = max(m[0][1], m[1][1])
    a1 = min(n[0][0], n[1][0])
    b1 = min(n[0][1], n[1][1])
    a2 = max(n[0][0], n[1][0])
    b2 = max(n[0][1], n[1][1])

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

            image = color_res_queue.get()
            heatmap = ir_res_queue.get()

            # 灰度化heatmap
            heatmap_g = heatmap.astype(np.uint8)
            # 热力图伪彩色
            heatmap_color = cv2.applyColorMap(heatmap_g, cv2.COLORMAP_JET)

            # color_map = cv2.applyColorMap(image, cv2.COLORMAP_JET)
            color_map = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2GRAY)
            # overlay热力图
            # merge_img = image.copy()
            # heatmap_img = heatmap_color.copy()
            # overlay = image.copy()
            # alpha = 0.25  # 设置覆盖图片的透明度
            # # cv2.rectangle(overlay, (0, 0), (merge_img.shape[1], merge_img.shape[0]), (0, 0, 0), -1) # 设置蓝色为热度图基本色
            # cv2.addWeighted(overlay, alpha, merge_img, 1 - alpha, 0, merge_img)  # 将背景热度图覆盖到原图
            # cv2.addWeighted(heatmap_img, alpha, merge_img, 1 - alpha, 0, merge_img)  # 将热度图覆盖到原图
            #     # return merge_img
            #
            # cv2.imshow(tracker_name + ' - merge', merge_img)
            # cv2.waitKey(20)
            cv2.imshow(tracker_name + ' - heatmap_color', heatmap_color)
            cv2.waitKey(20)
            # cv2.imshow(tracker_name + ' - overlay', overlay)
            # cv2.waitKey(20)
            cv2.imshow('color_map', color_map)
            cv2.waitKey(20)
