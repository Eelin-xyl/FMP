import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
from tools import is_cross, cover_img


def track_color(tracker_model, color_queue, color_res_queue, tmp_queue):

    color_tracker = tracker_model()

    while True:

        if color_queue.empty():
            continue
        else:
            color_image, gt_val = color_queue.get()
            color_res_image = color_image.copy()
            hit, box = color_tracker.update(color_res_image)
            m = ((gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]))
            n = ((box[0], box[1]), (box[0] + box[2], box[1] + box[3]))

            if hit and is_cross(m, n):

                cv2.rectangle(color_res_image, (gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]),
                              (0, 0, 255), 2)
                cv2.rectangle(color_res_image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])),
                              (255, 0, 0), 2)
                gt_val = ((int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])))

            else:
                # Not yet init or Track failed
                color_tracker = tracker_model()
                box1 = (gt_val[0][0], gt_val[0][1], gt_val[1][0] - gt_val[0][0], gt_val[1][1] - gt_val[0][1])
                color_tracker.init(color_res_image, box1)
                cv2.rectangle(color_res_image, (gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]),
                              (0, 255, 0), thickness=2)

            tmp_image = color_image[gt_val[0][1]:gt_val[1][1], gt_val[0][0]:gt_val[1][0]]
            tmp_image = cover_img(tmp_image)

            color_res_queue.put(color_res_image)
            tmp_queue.put((tmp_image, gt_val))
