import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
from tools import cover_img, sensor_miss_area


def track_ir(tracker_model, ir_queue, ir_res_queue, label_queue):

    ir_tracker = tracker_model()

    while True:

        if not ir_queue.empty() and not label_queue.empty():

            ir_image = ir_queue.get()
            ir__res_image = ir_image.copy()
            tmp_image, gt_val = label_queue.get()

            ir_image = cover_img(ir_image)
            target_area = sensor_miss_area(ir_image, gt_val)

            # cv2.rectangle(ir_image, (label_val[0][0], label_val[0][1]), (label_val[1][0], label_val[1][1]),
            #               (0, 255, 0), thickness=2)

            ir_res_queue.put(tmp_image)
