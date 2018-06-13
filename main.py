# coding: utf-8
import mxnet as mx
from mtcnn_detector import MtcnnDetector
from cv2_detector import Cv2FaceDetector
import cv2
import os
import time
import re
import numpy as np
import logging

# 通过下面的方式进行简单配置输出方式与日志级别
#logging.basicConfig(filename='logger.log', level=logging.INFO)
logging.basicConfig(filename='logger.log', 
                    format='%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s', 
                    level = logging.INFO,
                    filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')

re_poseIllum = re.compile('_\d{3}_\d{2}')
src_path = 'E:\Multi-PIE\session04\multiview'
out_put_path = 'E:\Multi-PIE\session04\cropped_align'

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
        if path[-7:-5] != '01':
            continue

        if path[-4:] == '19_1':
            continue

        if path[-4:] == '08_1':
            continue

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
    #print('cropped_name=', cropped_name)
    return cropped_name

def gen_5pt(img, five_pt_text_file):
    results = detector.detect_face(img)
    if results is not None:
        total_boxes = results[0]
        points = results[1]

        #print landmake
        for p in points:
            '''
            print('point:')
            for i in range(5):
                print(p[i], p[i+5])
            '''
            #y0, x0
            #five_pt_array = np.array([(p[5], p[0]), (p[6], p[1]), (p[7], p[2]), (p[8], p[3]), (p[9], p[4])])
            #x0, y0, x1, y1....
            five_pt_array = np.array([(p[0], p[5]), (p[1], p[6]), (p[2], p[7]), (p[3], p[8]), (p[4], p[9])])
            np.savetxt(five_pt_text_file, five_pt_array, fmt='%d', newline='\n')

def conver_point(p):
    key_points = []
    for k in range(len(p)//2):
        key_points.append(p[k])
        key_points.append(p[k+5])
    
    return key_points

def save_5pt(p, save_name):
    #x0, y0, x1, y1....
    five_pt_array = np.array([(p[0], p[1]), (p[2], p[3]), (p[4], p[5]), (p[6], p[7]), (p[8], p[9])])
    np.savetxt(save_name, five_pt_array, fmt='%d', newline='\n')

def crop_img(detector, cv2_face_detector, img_path):
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
        
        #print('points=', points)
        for p in points:
            key_points = conver_point(p)
            chip, five_points = detector.align(img, key_points, 128, 48, 40)
            five_points = np.resize(five_points, 10)

            dst_name = get_cropped_img_name(img_path)
            cropped_img_path = out_put_path + '\\' + dst_name
            #if cv2_face_detector.face_detected(chip) is False:
            #    logging.error('face detected %s failed'%(img_path))
            #else:
            cv2.imwrite(cropped_img_path, chip)
            logging.info('saved cropped %s succ'%(cropped_img_path))

            five_pt_text_file = cropped_img_path[:-4] + '.5pt'
            save_5pt(five_points, five_pt_text_file)
            logging.info('save five pt %s succ'%(five_pt_text_file))

        '''
        # extract aligned face chips
        chips = detector.extract_image_chips(img, points, 128, 0.37)
        for i, chip in enumerate(chips):
            #cv2.imshow('chip_'+str(i), chip)
            #cv2.imwrite('chip_'+str(i)+'.png', chip)
            dst_name = get_cropped_img_name(img_path)
            cropped_img_path = out_put_path + '\\' + dst_name
            
            if cv2_face_detector.face_detected(chip) is False:
                logging.error('face detected %s failed'%(img_path))
            else:
                cv2.imwrite(cropped_img_path, chip)
                logging.info('saved cropped %s succ'%(cropped_img_path))

                five_pt_text_file = cropped_img_path[:-4] + '.5pt'
                gen_5pt(chip, five_pt_text_file)
                logging.info('save five pt %s succ'%(five_pt_text_file))
        '''


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
    #else:
    #    print('face detected failed')

if __name__ == '__main__':    
    s = get_file_list(src_path)
    count = 0;
    cv2_face_detector = Cv2FaceDetector()
    detector = MtcnnDetector(model_folder='model', ctx=mx.cpu(0), num_worker = 1 , accurate_landmark = False)
    for img_path in s:
        count += 1
        cropped_file = get_cropped_img_name(img_path)
        cropped_file = os.path.join(out_put_path, cropped_file)
        #print('exits', cropped_file, os.path.exists(cropped_file))
        if False == os.path.exists(cropped_file):
            crop_img(detector, cv2_face_detector, img_path)
        
        print('cropping %d/%d' %(count, len(s)), end='\r')
    print('cropped completed!')
    exit(1)

    
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
