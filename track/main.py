import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
import platform
from read_port import read_data
from tracker_color import track_color
from tools import show_res


def detection(track_method):

    tracker_model, tracker_name = track_method

    sys = platform.system()

    path = ''
    if sys == "Windows":
        path = 'D:/Workspace/VOT2019-rgbtir'

        if not os.path.exists(path):
            path = 'C:/Users/xyl-e/Desktop/Workspace/VOT2019-rgbtir'

    if sys == "Linux":
        path = '/home/eelin/Desktop/VOT2019-rgbtir'

    file_list = []
    for root, dirs, files in os.walk(path):
        file_list = dirs
        break
    file_list.sort()

    len_queue = 5

    color_queue = Queue(len_queue)
    color_res_queue = Queue(len_queue)

    read_process = Process(target=read_data, args=(file_list, path, color_queue))
    read_process.start()

    color_process = Process(target=track_color, args=(tracker_model, color_queue, color_res_queue))
    color_process.start()

    show_process = Process(target=show_res, args=(tracker_name, color_res_queue))
    show_process.start()


if __name__ == "__main__":

    # track_algo = (cv2.TrackerMOSSE_create, 'MOSSE')              # miou -                 miss_hit 4000+ miss -
    # track_algo = (cv2.TrackerMedianFlow_create, 'MEDIANFLOW')    # miou 0.487118788918623 miss_hit 90    miss 231
    # track_algo = (cv2.TrackerTLD_create, 'TLD')                  # miou 0.501171326097705 miss_hit 18    miss 1841
    # track_algo = (cv2.TrackerMIL_create, 'MIL')                  # miou 0.481109237642200 miss_hit 0     miss 223
    # track_algo = (cv2.TrackerBoosting_create, 'BOOSTING')        # miou 0.511657367669579 miss_hit 0     miss 182
    # track_algo = (cv2.TrackerKCF_create, 'KCF')                  # miou 0.637434512129973 miss_hit 1263  miss 84
    track_algo = (cv2.TrackerCSRT_create, 'CSRT')                  # miou 0.542870060186356 miss_hit 9     miss 116

    detection(track_algo)

