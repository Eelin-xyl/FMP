import os
import time

import cv2

filelist = []
for root,dirs,files in os.walk('D:/Workspace/VOT2019-rgbtir'):
    filelist = dirs
    break

def track(tracker, tracker_name):

    num = 0
    hit = 0
    accuracy = 0

    for target in filelist:
        if target != 'car20':
            continue

        path = 'D:/Workspace/VOT2019-rgbtir/' + target
        video_path = '/'.join([path, 'color'])
        video_name_list = os.listdir(video_path)
        GT_path = '/'.join([path, 'groundtruth.txt'])

        init_once = False

        with open(GT_path, "r") as f:
            GT_file = f.read()  # 读取文件
            GT_val_list = GT_file.split('\n')
            # print(data)

        for idx in range(len(video_name_list)):

            video_name_path = os.path.join(video_path, video_name_list[idx])
            GT_val = GT_val_list[idx].split(',')
            GT_val = [int(float(i)) for i in GT_val]

            image = cv2.imread(video_name_path)
            # image = cv2.imread('E:/VOT2019—rgbtir/afterrain/color/00001v.jpg')

            # time.sleep(10)
            if not init_once:

                cv2.rectangle(image, (GT_val[0], GT_val[1]), (GT_val[4], GT_val[5]), (0, 0, 255), thickness=2)
                bbox1 = (min(GT_val[0], GT_val[4]), min(GT_val[1], GT_val[5]), abs(GT_val[4]-GT_val[0]), abs(GT_val[5]-GT_val[1]))
                tracker.init(image, bbox1)
                # ok = tracker.add(cv2.TrackerKCF_create(), image, bbox1)
                cv2.imshow(target, image)
                cv2.waitKey(20)
                # time.sleep(10)
                init_once = True
            else:

                ok, boxes = tracker.update(image)
               
                cv2.rectangle(image, (int(boxes[0]), int(boxes[1])), (int(boxes[0]+boxes[2]), int(boxes[1]+boxes[3])), (255, 0, 0), 2)
                cv2.rectangle(image, (GT_val[0], GT_val[1]), (GT_val[4], GT_val[5]), (0, 0, 255), thickness=2)

                num += 1
                if sum(boxes) != 0:
                    hit += 1
                    box1 = (min(int(boxes[0]), int(boxes[0]+boxes[2])), min(int(boxes[1]), int(boxes[1]+boxes[3])))
                    gt1 = (min(GT_val[0], GT_val[4]), min(GT_val[1], GT_val[5]))
                    diff1 = ((box1[0] - gt1[0])**2 + (box1[1] - gt1[1])**2)**0.5
                    box2 = (max(int(boxes[0]), int(boxes[0]+boxes[2])), max(int(boxes[1]), int(boxes[1]+boxes[3])))
                    gt2 = (max(GT_val[0], GT_val[4]), max(GT_val[1], GT_val[5]))
                    diff2 = ((box2[0] - gt2[0])**2 + (box2[1] - gt2[1])**2)**0.5
                    accuracy += diff1 + diff2

                # Displaying the image
                cv2.imshow(target, image)
                if cv2.waitKey(20) == 27:
                    break
        # print(target)
        cv2.destroyAllWindows()

    print(tracker_name)
    print('Hit: ', hit / num)
    print('Accuracy: ', accuracy)
    print()

        # cv2.waitKey(0)

tracker = cv2.TrackerKCF_create()
track(tracker, 'KCF')
tracker = cv2.TrackerBoosting_create()
track(tracker, 'Boosting')
tracker = cv2.TrackerCSRT_create()
track(tracker, 'CSRT')
tracker = cv2.TrackerMedianFlow_create()
track(tracker, 'MedianFlow')
tracker = cv2.TrackerMIL_create()
track(tracker, 'MIL')
tracker = cv2.TrackerMOSSE_create()
track(tracker, 'MOSSE')
tracker = cv2.TrackerTLD_create()
track(tracker, 'TLD')