import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
from tools import is_cross


def track_color(tracker_model, color_queue, color_res_queue, label_queue):

    color_tracker = tracker_model()

    while True:

        if color_queue.empty():
            continue
        else:
            color_image, gt_val = color_queue.get()
            hit, box = color_tracker.update(color_image)
            m = ((gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]))
            n = ((box[0], box[1]), (box[0] + box[2], box[1] + box[3]))

            if hit and is_cross(m, n):

                cv2.rectangle(color_image, (gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]),
                              (0, 0, 255), 2)
                cv2.rectangle(color_image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])),
                              (255, 0, 0), 2)

            else:
                # Not yet init or Track failed
                color_tracker = tracker_model()
                box1 = (gt_val[0][0], gt_val[0][1], gt_val[1][0] - gt_val[0][0], gt_val[1][1] - gt_val[0][1])
                color_tracker.init(color_image, box1)
                cv2.rectangle(color_image, (gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]),
                              (0, 255, 0), thickness=2)

            color_res_queue.put(color_image)
            label_queue.put(gt_val)