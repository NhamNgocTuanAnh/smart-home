
import numpy
import cv2,os
from decimal import Decimal
import time
import pytz
UTC = pytz.timezone('UTC')  # utc
from datetime import datetime as dt
now = dt.now(pytz.timezone('Asia/Ho_Chi_Minh'))
from libc.stdio cimport *
cimport numpy, cython
from libc.stdint cimport (
  uint8_t, uint16_t, uint32_t, uint64_t,
  int8_t, int16_t, int32_t, int64_t,
  uintptr_t
)
#import pyshine as ps
import logging

cimport cqueue

from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import threading,queue
import multiprocessing

from multiprocessing import Process, Manager, cpu_count, set_start_method

input_image_time_buffer = queue.Queue(550)


#EMOTIONS = numpy.array(["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral", "background"])     
        
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef smooth_emotions():
    #global EMOTIONS
    
    # load model facial_expression
    model_facial_expression = model_from_json(open("model/fer.json", "r").read())
    # load weights facial_expression
    model_facial_expression.load_weights('model/fer.h5')
    
    emotions = numpy.array(['Angry', 'Background', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
    
    cdef :
        numpy.ndarray roi, gray_temp
        str path_face_save
        int label_temp = 0
    
    while True:
        try:
            (gray_temp, current_time) = input_image_time_buffer.get(timeout=1)
            roi = numpy.expand_dims(img_to_array(gray_temp.astype("float") / 255.0), axis=0)
     
            preds = model_facial_expression.predict(roi)[0]
            if preds.argmax() < 7:
                label_temp = int(preds.argmax())
                #path_face_save = ' ' .join(["face_database/", str(EMOTIONS[label_temp]), "/", str(int(current_time)), '_' ,".jpg"])
                #str("face_database/"+str(EMOTIONS[label_temp])+"/"+str(EMOTIONS[label_temp])+'_'+ str(int(current_time))+"_"+str(number_face) + ".jpg")
                #cv2.imwrite(path_face_save, gray_temp)

                predicted_emotion = str(emotions[label_temp])
                d =  Video_Process()
                d.smo_emotions()
                #print(predicted_emotion)
		
        except queue.Empty:
            logging.warning("Empty memory!")
            pass

def get_best_images(input_frames:[])-> numpy.array([]):
    blur_threshold=int(100)
    input_frames = sorted(input_frames, key=lambda img : cv2.Laplacian(img[0], cv2.CV_64F).var(), reverse=True)
    # Lay khung hinh co muc do focus tot nhat
    best_image = input_frames[0]
    return best_image    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef show():
    cdef :
        bint ret = True
        int i
        int count = 0
        numpy.uint32_t x, y, w, h
        numpy.ndarray frame, gray, roi
        float time_start
        bint time_checker = True

    cap = cv2.VideoCapture("videos/democlassroom.mp4")
    print("Width: %d, Height: %d, FPS: %d" % (cap.get(3), cap.get(4), cap.get(5)))
    input_frames = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")    
    while True :
        try:
            ret, frame = cap.read()

            if (not (ret is  True)):                
                logging.warning("Something wrong!")
                break

            count += 1

            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            input_frames.append(frame_gray)
            #print(type(input_frames))
            if count%5==0:
               gray_img = get_best_images(input_frames)
               faces = face_cascade.detectMultiScale(gray_img, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)
               i = int(0)
               for i in range(0,len(faces)):
                   x = faces[i][0]
                   y = faces[i][1]
                   w = faces[i][2]
                   h = faces[i][3]

                   roi = gray_img[y:y + h, x:x + w]
                   roi = cv2.resize(roi, (48, 48))
                   input_image_time_buffer.put((roi,time_start), timeout=1)
                   
               input_frames.clear()   
               count = int(0) 
           
        except queue.Full:
            logging.warning("full memory!")
            pass
 
    cap.release()
    cv2.destroyAllWindows()

@cython.boundscheck(False)
@cython.wraparound(False)
def Main():
    print("Số lượng cpu : ", multiprocessing.cpu_count())
    
    tReadFile = threading.Thread(target=show)
    tProcessingFile = threading.Thread(target=smooth_emotions)

    tReadFile.start()
    tProcessingFile.start()

    tProcessingFile.join()
    tReadFile.join()
    print("Bye !!!")
