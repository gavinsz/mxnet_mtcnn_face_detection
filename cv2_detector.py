import cv2
import sys
import os
import logging

class Cv2FaceDetector(object):
    #CASCADE_PATH = "data/haarcascades/haarcascade_frontalface_default.xml"
    CASCADE_PATH = 'E:\Anaconda3\pkgs\opencv-3.4.1-py36_200\Library\etc\haarcascades'
    front_face_cascades_path = CASCADE_PATH + '\haarcascade_frontalface_default.xml'
    profile_face_cascades_path = CASCADE_PATH + '\haarcascade_profileface.xml'
    def __init__(self):
        self.front_face_cascade = cv2.CascadeClassifier(self.front_face_cascades_path)
        self.profile_face_cascade = cv2.CascadeClassifier(self.profile_face_cascades_path)
        self.i = 0
    
    def face_detected(self, img):
        if (img is None):
            return False

        #front face detect
        faces = self.front_face_cascade.detectMultiScale(img, 1.1, 3, flags=2, minSize=(70, 70))
        if (faces is None):
            logging.error('detect front face failed')
            return False
        
        facecnt = len(faces)
        if 0 == facecnt:
            #profile face detect
            faces = self.profile_face_cascade.detectMultiScale(img, 1.1, 3, minSize=(60, 60))
            if (faces is None):
                logging.error('detect profile face failed')
                return False

            facecnt = len(faces)
            if 0 == facecnt:
                return False
            else:
                return True
        else:
            return True
        
    def generate(self, image_path, show_result):
        img = cv2.imread(image_path)
        if (img is None):
            print("Can't open image file")
            return 0

        print ('face detect for ', image_path)
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        #front face detect
        #faces = self.front_face_cascade.detectMultiScale(img, 1.1, 5, minSize=(128, 128))
        faces = self.front_face_cascade.detectMultiScale(img, 1.1, 3, flags=2, minSize=(70, 70))
        if (faces is None):
            print('Failed to detect face')
            return 0

        if (show_result):
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x,y), (x+w, y+h), (255,0,0), 2)
            cv2.imshow('img', img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

        facecnt = len(faces)
        print("Detected front faces: %d" % facecnt)

        if facecnt == 0:
            #profile face detect
            faces = self.profile_face_cascade.detectMultiScale(img, 1.1, 3, minSize=(60, 60))
            if (faces is None):
                print('Failed to detect face')
                return 0

            facecnt = len(faces)
            print("Detected profile faces: %d" % facecnt)
