import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
from tools import is_cross


def track_color(tracker_model, color_queue, color_res_queue):

    color_tracker = tracker_model()
    init_scene = ''

    while True:

        if color_queue.empty():
            continue
        else:
            color_image, gt_val, scene = color_queue.get()
            hit, bbox = color_tracker.update(color_image)
            box1 = ((gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]))
            box2 = ((bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]))

            if hit and is_cross(box1, box2) and scene == init_scene:

                cv2.rectangle(color_image, (gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]),
                              (0, 0, 255), 2)
                cv2.rectangle(color_image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              (255, 0, 0), 2)

            else:
                # Not yet init or Track failed or Scene changed
                init_scene = scene
                color_tracker = tracker_model()
                bbox1 = (gt_val[0][0], gt_val[0][1], gt_val[1][0] - gt_val[0][0], gt_val[1][1] - gt_val[0][1])
                color_tracker.init(color_image, bbox1)
                cv2.rectangle(color_image, (gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]),
                              (0, 255, 0), thickness=2)

            color_res_queue.put(color_image)
