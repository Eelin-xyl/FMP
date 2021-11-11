import os
import time
import multiprocessing
from multiprocessing import Pipe, Process, shared_memory, Queue
import numpy as np

import cv2

filelist = []
for root, dirs, files in os.walk('D:/Workspace/VOT2019-rgbtir'):
    filelist = dirs
    break


def iscross(m, n):
    # A = [
    #         (x1, y1),     LeftTop
    #         (x2, y2)      RightBottom
    #     ]
    # B = [
    #         (a1, b1),     LeftTop
    #         (a2, b2)      RightBottom
    #     ]

    x1 = min(m[0][0], m[1][0])
    y1 = min(m[0][1], m[1][1])
    x2 = max(m[0][0], m[1][0])
    y2 = max(m[0][1], m[1][1])
    a1 = min(n[0][0], n[1][0])
    b1 = min(n[0][1], n[1][1])
    a2 = max(n[0][0], n[1][0])
    b2 = max(n[0][1], n[1][1])

    if x2 < a1 or a2 < x1:
        return False

    if y2 < b1 or b2 < y1:
        return False

    return True

    # w = [m[0][0], m[1][0], n[0][0], n[1][0]]
    # w.sort()
    # h = [m[0][1], m[1][1], n[0][1], n[1][1]]
    # h.sort()
    #
    # return abs(w[1] - w[2]) * abs(h[1] - h[2])


def track(tracker_model, tracker_name):

    color_queue = Queue()
    ir_queue = Queue()

    for target in filelist:

        # if target != 'car10':
        #     continue

        path = 'D:/Workspace/VOT2019-rgbtir/' + target

        # color_img info
        color_path = '/'.join([path, 'color'])
        color_list = os.listdir(color_path)

        # length of each target_img
        img_num = len(color_list)

        # ir_img info
        ir_path = '/'.join([path, 'ir'])
        ir_list = os.listdir(ir_path)

        gt_path = '/'.join([path, 'groundtruth.txt'])

        # color_tracker = tracker_model()
        # ir_tracker = tracker_model()

        # import groundtruth data
        with open(gt_path, "r") as f:

            gt_file = f.read()  # Read file
            gt_val_list = gt_file.split('\n')
            # print(data)

        for idx in range(img_num):
            gt_val = gt_val_list[idx].split(',')
            gt_val = [int(float(i)) for i in gt_val]

            # get color and ir picture by cv2
            color_img = os.path.join(color_path, color_list[idx])
            color_image = cv2.imread(color_img)

            ir_img = os.path.join(ir_path, ir_list[idx])
            ir_image = cv2.imread(ir_img)

            shape = np.shape(color_image)
            dtype = color_image.dtype
            # images = multiprocessing.Manager().dict()
            shm = shared_memory.SharedMemory(
                create=True, size=color_image.nbytes)
            image_in = np.ndarray(shape, dtype=dtype, buffer=shm.buf)
            image_in[:] = color_image.copy()

            # color_SendPort.send(color_image)
            color_queue.put(color_image)
            # color_SendPort.send([image_in, gt_val])
            # color_image, gt_val,

            color_process = Process(target=track_color, args=(
                tracker_model, tracker_name, color_RecvPort, color_queue), daemon=True)
            color_process.start()
            # ir_process = Process(target=track_ir, args=(tracker_model, ir_tracker, ir_image, gt_val), daemon=True)
            # color_tracker, color_image = track_color(
            #     tracker_model, color_tracker, color_image, gt_val)
            # ir_tracker, ir_image = track_ir(
            #     tracker_model, ir_tracker, ir_image, gt_val)

            # # Displaying the image
            # cv2.imshow(tracker_name + ' - color', color_image)
            # cv2.imshow(tracker_name + ' - ir', ir_image)
            # cv2.waitKey(20)
        # print(target)

    # print(tracker_name)
    # print('Miss_color: ', miss_color)
    # print('Accuracy_color: ', accuracy_color)
    # print()
    cv2.destroyWindow(tracker_name)


def track_color(tracker_model, tracker_name, color_RecvPort, color_queue):

    color_tracker = tracker_model()
    # color_image, gt_val = color_RecvPort.recv()

    while True:

        if not color_queue.empty():
            color_image = color_queue.get()
            cv2.imshow(tracker_name + ' - color', color_image)
            cv2.waitKey(20)
        else:
            continue
        color_image, gt_val = color_RecvPort.recv()
        hit, box = color_tracker.update(color_image)
        m = [(gt_val[0], gt_val[1]), (gt_val[4], gt_val[5])]
        n = [(box[0], box[1]), (box[0] + box[2], box[1] + box[3])]

        if hit and iscross(m, n):

            cv2.rectangle(color_image, (gt_val[0], gt_val[1]), (gt_val[4], gt_val[5]),
                          (0, 0, 255), 2)
            cv2.rectangle(color_image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])),
                          (255, 0, 0), 2)

        else:
            # Not yet init or Track failed
            color_tracker = tracker_model()
            box1 = (min(gt_val[0], gt_val[4]), min(gt_val[1], gt_val[5]),
                    abs(gt_val[4] - gt_val[0]), abs(gt_val[5] - gt_val[1]))
            color_tracker.init(color_image, box1)

            cv2.rectangle(color_image, (gt_val[0], gt_val[1]), (gt_val[4], gt_val[5]),
                          (0, 255, 0), thickness=2)

        cv2.imshow(tracker_name + ' - color', color_image)
        cv2.waitKey(20)


def track_ir(tracker_model, ir_tracker, ir_image, gt_val):

    hit, box = ir_tracker.update(ir_image)
    m = [(gt_val[2], gt_val[3]), (gt_val[6], gt_val[7])]
    n = [(box[0], box[1]), (box[0] + box[2], box[1] + box[3])]

    if hit and iscross(m, n):

        cv2.rectangle(ir_image, (gt_val[2], gt_val[3]), (gt_val[6], gt_val[7]),
                      (0, 0, 255), 2)
        cv2.rectangle(ir_image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])),
                      (255, 0, 0), 2)

    else:
        # Not yet init or Track failed
        ir_tracker = tracker_model()
        box1 = (min(gt_val[2], gt_val[6]), min(gt_val[3], gt_val[6]),
                abs(gt_val[6] - gt_val[2]), abs(gt_val[7] - gt_val[3]))
        ir_tracker.init(ir_image, box1)

        cv2.rectangle(ir_image, (gt_val[2], gt_val[3]), (gt_val[6], gt_val[7]),
                      (0, 255, 0), thickness=2)

    return ir_tracker, ir_image


if __name__ == "__main__":
    # track(cv2.TrackerBoosting_create, 'BOOSTING')    # 0
    # track(cv2.TrackerMIL_create, 'MIL')    # 0
    # track(cv2.TrackerKCF_create, 'KCF')    # 1074
    # track(cv2.TrackerTLD_create, 'TLD')    # 34
    # track(cv2.TrackerMedianFlow_create, 'MEDIANFLOW')    # 113
    track(cv2.TrackerCSRT_create, 'CSRT')  # 16
