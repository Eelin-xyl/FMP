import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
from tools import covert_img


def track_color(tracker_model, color_queue, color_res_queue, tmp_queue):

    color_tracker = tracker_model()
    init_roi = ((0, 0), (0, 0))

    while True:

        if color_queue.empty():
            continue

        else:
            color_image, roi = color_queue.get()
            color_res_image = color_image.copy()

            if roi != init_roi:
                # create tracker
                init_roi = roi
                color_tracker = tracker_model()
                box1 = roi
                color_tracker.init(color_res_image, box1)
                cv2.rectangle(color_res_image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]),
                              (0, 255, 0), thickness=2)
                gt_val = ((int(roi[0]), int(roi[1])), (int(roi[0] + roi[2]), int(roi[1] + roi[3])))

            else:
                hit, box = color_tracker.update(color_res_image)
                cv2.rectangle(color_res_image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])),
                              (255, 0, 0), 2)
                gt_val = ((int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])))

            tmp_image = color_image[gt_val[0][1]:gt_val[1][1], gt_val[0][0]:gt_val[1][0]]
            tmp_image = covert_img(tmp_image)

            # color_image = cv2.cvtColor(color_image, cv2.COLOR_BGR2HSV)
            # color_image = cv2.split(color_image)
            # color_image = color_image[2]

            color_res_queue.put((color_res_image, tmp_image))
            tmp_queue.put((tmp_image, gt_val))
