import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2


def track_color(tracker_model, color_queue, color_res_queue):

    color_tracker = tracker_model()
    init_roi = ((0, 0), (0, 0))

    while True:

        if color_queue.empty():
            continue

        else:

            color_image, roi = color_queue.get()

            if roi != init_roi:

                init_roi = roi
                color_tracker = tracker_model()
                box1 = roi
                color_tracker.init(color_image, box1)
                cv2.rectangle(color_image, (roi[0], roi[1]), (roi[0] + roi[2], roi[1] + roi[3]),
                              (0, 255, 0), thickness=2)

            else:

                hit, box = color_tracker.update(color_image)
                cv2.rectangle(color_image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])),
                              (255, 0, 0), 2)

            color_res_queue.put(color_image)
