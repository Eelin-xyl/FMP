import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
from read_port import read_data
from tracker_color import track_color
from tracker_ir import track_ir
from tools import show_res


def track(tracker_model, tracker_name):

    filelist = []
    for root, dirs, files in os.walk('D:/Workspace/VOT2019-rgbtir'):
        filelist = dirs
        break

    color_queue = Queue()
    ir_queue = Queue()
    color_res_queue = Queue()
    ir_res_queue = Queue()

    read_process = Process(target=read_data, args=(filelist, color_queue, ir_queue))
    read_process.start()

    color_process = Process(target=track_color, args=(tracker_model, color_queue, color_res_queue))
    color_process.start()

    ir_process = Process(target=track_ir, args=(tracker_model, ir_queue, ir_res_queue))
    ir_process.start()

    show_process = Process(target=show_res, args=(tracker_name, color_res_queue, ir_res_queue))
    show_process.start()

    # print(tracker_name)
    # print('Miss_color: ', miss_color)
    # print('Accuracy_color: ', accuracy_color)
    # print()
    # cv2.destroyWindow(tracker_name)


if __name__ == "__main__":
    # track(cv2.TrackerBoosting_create, 'BOOSTING')    # 0
    # track(cv2.TrackerMIL_create, 'MIL')    # 0
    # track(cv2.TrackerKCF_create, 'KCF')    # 1074
    # track(cv2.TrackerTLD_create, 'TLD')    # 34
    # track(cv2.TrackerMedianFlow_create, 'MEDIANFLOW')    # 113
    track(cv2.TrackerCSRT_create, 'CSRT')  # 16
