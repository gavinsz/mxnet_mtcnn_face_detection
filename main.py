# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
import cv2
import os
import time
import re

re_poseIllum = re.compile('_\d{3}_\d{2}')
out_put_path = 'E:\Multi-PIE\session01\cropped'

def is_normal_illumination(fullpath):
    #print('fullpath=', fullpath)
    span = re_poseIllum.search(fullpath).span()
    num = fullpath[span[0]+5:span[1]]

    #print('span[1]=', fullpath[span[0]+5:span[1]])
    if num == '12':
        return True
    else:
        return False    

def get_file_list(path):
    g = os.walk(path)
    s = []
    for path, dir_list, file_list in g:
        for file_name in file_list:
            if True == is_normal_illumination(file_name):
                tmp = os.path.join(path, file_name)
                #print('tmp=', tmp)
                s.append(tmp)

    return s

def get_cropped_img_name(img_path):
    pos = img_path.rfind('\\')
    cropped_name = ''
    if -1 != pos:
        name = img_path[pos+1:]
        #print('name=', name)
        cropped_name = name[:-4] + '_cropped.png'
    print('cropped_name=', cropped_name)
    return cropped_name

def crop_img(detector, img_path):
    img = cv2.imread(img_path)
    if img is None:
        return
        
    #print('img_path=', img_path)
    #img = cv2.imread('oscar1.jpg')
    # run detector
    results = detector.detect_face(img)
    if results is not None:
        total_boxes = results[0]
        points = results[1]
        
        # extract aligned face chips
        chips = detector.extract_image_chips(img, points, 128, 0.37)
        for i, chip in enumerate(chips):
            #cv2.imshow('chip_'+str(i), chip)
            #cv2.imwrite('chip_'+str(i)+'.png', chip)
            dst_name = get_cropped_img_name(img_path)
            cropped_img_path = out_put_path + '\\' + dst_name
            cv2.imwrite(cropped_img_path, chip)
        '''
        draw = img.copy()
        for b in total_boxes:
            cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

        for p in points:
            for i in range(5):
                cv2.circle(draw, (p[i], p[i + 5]), 1, (0, 0, 255), 2)

        cv2.imshow("detection result", draw)
        cv2.waitKey(0)
        '''
if __name__ == '__main__':
    path = 'E:\Multi-PIE\session01\multiview'
    s = get_file_list(path)
    #print(s)

    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 1 , accurate_landmark = False)
    for img_path in s:
        crop_img(detector, img_path)

    
# --------------
# test on camera
# --------------
'''
camera = cv2.VideoCapture(0)
while True:
    grab, frame = camera.read()
    img = cv2.resize(frame, (320,180))

    t1 = time.time()
    results = detector.detect_face(img)
    print 'time: ',time.time() - t1

    if results is None:
        continue

    total_boxes = results[0]
    points = results[1]

    draw = img.copy()
    for b in total_boxes:
        cv2.rectangle(draw, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (255, 255, 255))

    for p in points:
        for i in range(5):
            cv2.circle(draw, (p[i], p[i + 5]), 1, (255, 0, 0), 2)
    cv2.imshow("detection result", draw)
    cv2.waitKey(30)
'''
