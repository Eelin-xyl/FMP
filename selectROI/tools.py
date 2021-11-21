import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2


def show_res(tracker_name, color_res_queue):

    while True:

        if not color_res_queue.empty():
            cv2.imshow(tracker_name + ' - color', color_res_queue.get())
            cv2.waitKey(40)
