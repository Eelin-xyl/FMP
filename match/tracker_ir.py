import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2


def track_ir(tracker_model, ir_queue, ir_res_queue, label_queue):

    ir_tracker = tracker_model()

    while True:

        if not ir_queue.empty() and not label_queue.empty():

            ir_image = ir_queue.get()
            label_val = label_queue.get()



            ir_res_queue.put(ir_image)
