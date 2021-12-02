import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
from tools import is_cross


def track_ir(tracker_model, ir_queue, ir_res_queue):

    ir_tracker = tracker_model()
    init_scene = ''

    while True:

        if ir_queue.empty():
            continue
        else:
            ir_image, gt_val, scene = ir_queue.get()
            hit, bbox = ir_tracker.update(ir_image)
            box1 = ((gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]))
            box2 = ((bbox[0], bbox[1]), (bbox[0] + bbox[2], bbox[1] + bbox[3]))

            if hit and is_cross(box1, box2) and scene == init_scene:

                cv2.rectangle(ir_image, (gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]),
                              (0, 0, 255), 2)
                cv2.rectangle(ir_image, (int(bbox[0]), int(bbox[1])), (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3])),
                              (255, 0, 0), 2)

            else:
                # Not yet init or Track failed or Scene changed
                init_scene = scene
                ir_tracker = tracker_model()
                bbox1 = (gt_val[0][0], gt_val[0][1], gt_val[1][0] - gt_val[0][0], gt_val[1][1] - gt_val[0][1])
                ir_tracker.init(ir_image, bbox1)
                cv2.rectangle(ir_image, (gt_val[0][0], gt_val[0][1]), (gt_val[1][0], gt_val[1][1]),
                              (0, 255, 0), thickness=2)

            ir_res_queue.put(ir_image)
