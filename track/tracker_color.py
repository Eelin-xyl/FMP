import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
from tools import is_cross, cal_iou


def track_color(tracker_model, color_queue, color_res_queue):

    color_tracker = tracker_model()
    init_scene = ''
    sum_iou = 0
    count_iou = 0
    mean_iou = 0
    miss_hit = 0
    miss = 0

    while True:

        if color_queue.empty():
            continue

        else:

            color_image, gt_val, scene = color_queue.get()

            if scene == init_scene:

                hit, bbox = color_tracker.update(color_image)

                if not hit:
                    miss_hit += 1

                # limit the area
                bbox = (max(0, bbox[0]), max(0, bbox[1]), min(color_image.shape[1], bbox[0] + bbox[2]) - bbox[0],
                        min(color_image.shape[0], bbox[1] + bbox[3]) - bbox[1])

                box1 = ((gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]))
                box2 = ((bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]))

                if hit and is_cross(box1, box2):

                    cv2.rectangle(color_image, (gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]),
                                  (0, 0, 255), 2)
                    cv2.rectangle(color_image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                                  (255, 0, 0), 2)

                    track_val = ((int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])))
                    sum_iou += cal_iou(track_val, gt_val)
                    count_iou += 1
                    mean_iou = sum_iou / count_iou

                else:
                    if hit:
                        miss += 1

                    # Track failed
                    color_tracker = tracker_model()
                    bbox1 = (gt_val[0][0], gt_val[0][1], gt_val[1][0] - gt_val[0][0], gt_val[1][1] - gt_val[0][1])
                    color_tracker.init(color_image, bbox1)
                    cv2.rectangle(color_image, (gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]),
                                  (0, 255, 0), thickness=2)

            else:
                # Scene changed
                init_scene = scene
                color_tracker = tracker_model()
                bbox1 = (gt_val[0][0], gt_val[0][1], gt_val[1][0] - gt_val[0][0], gt_val[1][1] - gt_val[0][1])
                color_tracker.init(color_image, bbox1)
                cv2.rectangle(color_image, (gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]),
                              (0, 255, 0), thickness=2)

            print(mean_iou)
            print(miss_hit)
            print(miss)

            color_res_queue.put(color_image)
