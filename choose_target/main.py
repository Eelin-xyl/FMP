import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np
import cv2
import platform
from read_port import read_data
from tracker_color import track_color
from matcher_ir import matcher_ir
from tools import show_res


def detection(track_method, match_method):

    tracker_model, tracker_name = track_method
    matcher_model, matcher_name = match_method

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

    len_queue = 3

    color_queue = Queue(len_queue)
    ir_queue = Queue(len_queue)
    color_res_queue = Queue(len_queue)
    ir_res_queue = Queue(len_queue)
    tmp_queue = Queue(len_queue)

    read_process = Process(target=read_data, args=(file_list, path, color_queue, ir_queue))
    read_process.start()

    color_process = Process(target=track_color, args=(tracker_model, color_queue, color_res_queue, tmp_queue))
    color_process.start()

    ir_process = Process(target=matcher_ir, args=(matcher_model, ir_queue, ir_res_queue, tmp_queue))
    ir_process.start()

    show_process = Process(target=show_res, args=(tracker_name, matcher_name, color_res_queue, ir_res_queue))
    show_process.start()


if __name__ == "__main__":

    # track_algo = (cv2.TrackerMOSSE_create, 'MOSSE')              # miou -                 miss_hit 4000+ miss -
    # track_algo = (cv2.TrackerMedianFlow_create, 'MEDIANFLOW')    # miou 0.487118788918623 miss_hit 90    miss 231
    # track_algo = (cv2.TrackerTLD_create, 'TLD')                  # miou 0.501171326097705 miss_hit 18    miss 1841
    # track_algo = (cv2.TrackerMIL_create, 'MIL')                  # miou 0.481109237642200 miss_hit 0     miss 223
    track_algo = (cv2.TrackerBoosting_create, 'BOOSTING')          # miou 0.511657367669579 miss_hit 0     miss 182
    # track_algo = (cv2.TrackerKCF_create, 'KCF')                  # miou 0.637434512129973 miss_hit 1263  miss 84
    # track_algo = (cv2.TrackerCSRT_create, 'CSRT')                # miou 0.542870060186356 miss_hit 9     miss 116

    # match_algo = (cv2.TM_SQDIFF_NORMED, 'SQDIFF_NORMED')         # miou 0.142879529796725 (gray and canny)
    # match_algo = (cv2.TM_SQDIFF, 'SQDIFF')                       # miou 0.401018869909011 (gray and canny)
    # match_algo = (cv2.TM_CCORR, 'CCORR')                         # miou 0.514883207994326 (gray and canny)
    # match_algo = (cv2.TM_CCOEFF_NORMED, 'CCOEFF_NORMED')         # miou 0.536817262918913 (gray and canny)
    # match_algo = (cv2.TM_CCORR_NORMED, 'CCORR_NORMED')           # miou 0.551935810965585 (gray and canny)
    match_algo = (cv2.TM_CCOEFF, 'CCOEFF')                         # miou 0.556620972473444 (gray and canny)

    detection(track_algo, match_algo)

