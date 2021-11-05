import os
import time
from typing import NewType

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
    miss = 0
    # accuracy = 0

    for target in filelist:
        # if target != 'car10':
        #     continue

        path = 'D:/Workspace/VOT2019-rgbtir/' + target
        img_path = '/'.join([path, 'color'])
        img_list = os.listdir(img_path)
        GT_path = '/'.join([path, 'groundtruth.txt'])

        init_once = False
        tracker = tracker_model()

        with open(GT_path, "r") as f:
            GT_file = f.read()  # 读取文件
            GT_val_list = GT_file.split('\n')
            # print(data)

        for idx in range(len(img_list)):

            img = os.path.join(img_path, img_list[idx])
            GT_val = GT_val_list[idx].split(',')
            GT_val = [int(float(i)) for i in GT_val]

            image = cv2.imread(img)
            # image = cv2.imread('E:/VOT2019—rgbtir/afterrain/color/00001v.jpg')

            # time.sleep(10)
            if not init_once:

                cv2.rectangle(
                    image, (GT_val[0], GT_val[1]), (GT_val[4], GT_val[5]), (255, 0, 0), thickness=2)
                box1 = (min(GT_val[0], GT_val[4]), min(GT_val[1], GT_val[5]), abs(GT_val[4] - GT_val[0]),
                        abs(GT_val[5] - GT_val[1]))
                tracker.init(image, box1)
                # hit = tracker.add(cv2.TrackerKCF_create(), image, box1)
                cv2.imshow(tracker_name, image)
                cv2.waitKey(20)
                # time.sleep(10)
                init_once = True
            else:

                hit, box = tracker.update(image)

                m = [(GT_val[0], GT_val[1]), (GT_val[4], GT_val[5])]
                n = [(box[0], box[1]), (box[0] + box[2], box[1] + box[3])]

                if hit and iscross(m, n):
                    # if hit:

                    # box1 = (min(int(box[0]), int(box[0] + box[2])), min(int(box[1]), int(box[1] + box[3])))
                    # gt1 = (min(GT_val[0], GT_val[4]), min(GT_val[1], GT_val[5]))
                    # diff1 = ((box1[0] - gt1[0]) ** 2 + (box1[1] - gt1[1]) ** 2) ** 0.5
                    # box2 = (max(int(box[0]), int(box[0] + box[2])), max(int(box[1]), int(box[1] + box[3])))
                    # gt2 = (max(GT_val[0], GT_val[4]), max(GT_val[1], GT_val[5]))
                    # diff2 = ((box2[0] - gt2[0]) ** 2 + (box2[1] - gt2[1]) ** 2) ** 0.5
                    # accuracy += diff1 + diff2

                    cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[0] + box[2]), int(box[1] + box[3])),
                                  (255, 0, 0), 2)
                    cv2.rectangle(
                        image, (GT_val[0], GT_val[1]), (GT_val[4], GT_val[5]), (0, 0, 255), 2)

                else:
                    miss += 1
                    tracker = tracker_model()
                    box1 = (min(GT_val[0], GT_val[4]), min(GT_val[1], GT_val[5]), abs(GT_val[4] - GT_val[0]),
                            abs(GT_val[5] - GT_val[1]))
                    tracker.init(image, box1)
                    cv2.rectangle(
                        image, (GT_val[0], GT_val[1]), (GT_val[4], GT_val[5]), (0, 255, 0), 2)

                # Displaying the image
                cv2.imshow(tracker_name, image)
                cv2.waitKey(20)
                # if cv2.waitKey(20) == 27:
                #     break
        # print(target)
        # cv2.destroyAllWindows()

    print(tracker_name)
    print('Miss: ', miss)
    # print('Accuracy: ', accuracy)
    print()

    # cv2.waitKey(0)


track(cv2.TrackerBoosting_create, 'BOOSTING')    # 0
# track(cv2.TrackerMIL_create, 'MIL')    # 0
# track(cv2.TrackerKCF_create, 'KCF')    # 1074
# track(cv2.TrackerTLD_create, 'TLD')    # 34
# track(cv2.TrackerMedianFlow_create, 'MEDIANFLOW')    # 113
track(cv2.TrackerCSRT_create, 'CSRT')  # 16
