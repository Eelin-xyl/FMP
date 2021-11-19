import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
import platform
from read_port import read_data
from tracker_color import track_color
from tracker_ir import track_ir
from tools import show_res


def track(tracker_model, tracker_name):

    sys = platform.system()

    path = ''
    if sys == "Windows":
        path = 'D:/Workspace/VOT2019-rgbtir'
    if sys == "Linux":
        path = '/home/eelin/Desktop/VOT2019-rgbtir'

    file_list = []
    for root, dirs, files in os.walk(path):
        file_list = dirs
        break
    file_list.sort()

    len_queue = 10

    color_queue = Queue(len_queue)
    ir_queue = Queue(len_queue)
    color_res_queue = Queue(len_queue)
    ir_res_queue = Queue(len_queue)

    read_process = Process(target=read_data, args=(file_list, path, color_queue, ir_queue))
    read_process.start()

    color_process = Process(target=track_color, args=(tracker_model, color_queue, color_res_queue))
    color_process.start()

    ir_process = Process(target=track_ir, args=(tracker_model, ir_queue, ir_res_queue))
    ir_process.start()

    show_process = Process(target=show_res, args=(tracker_name, color_res_queue, ir_res_queue))
    show_process.start()


if __name__ == "__main__":
    # track(cv2.TrackerBoosting_create, 'BOOSTING')    # 0
    # track(cv2.TrackerMIL_create, 'MIL')    # 0
    # track(cv2.TrackerKCF_create, 'KCF')    # 1074
    # track(cv2.TrackerTLD_create, 'TLD')    # 34
    # track(cv2.TrackerMedianFlow_create, 'MEDIANFLOW')    # 113
    track(cv2.TrackerCSRT_create, 'CSRT')  # 16
