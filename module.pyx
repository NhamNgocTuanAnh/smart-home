
import numpy
import cv2,os
from decimal import Decimal
import time
import pytz
UTC = pytz.timezone('UTC')  # utc
from datetime import datetime as dt
now = dt.now(pytz.timezone('Asia/Ho_Chi_Minh'))
from libc.stdio cimport *
cimport numpy
cimport cython
from libc.stdint cimport (
  uint8_t, uint16_t, uint32_t, uint64_t,
  int8_t, int16_t, int32_t, int64_t,
  uintptr_t
)
#import pyshine as ps
import logging


from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import threading,queue
import multiprocessing

from multiprocessing import Process, Manager, cpu_count, set_start_method

input_image_time_buffer = queue.Queue(550)

# check time 
cdef :
    float last_time_face = -1
    float current_time_face = 0
    int number = 100000
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
# load model facial_expression
model_facial_expression = model_from_json(open("model/fer.json", "r").read())
# load weights facial_expression
model_facial_expression.load_weights('model/fer.h5')


EMOTIONS = numpy.array(["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral", "background"])
cdef :
    str log_system = 'Error: '
    bint paused = False
    bint finished = False
    numpy.ndarray frame_temp, gray_temp
total_preds = numpy.array([])

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef cls():
    os.system('cls' if os.name=='nt' else 'clear')

cdef:
    str path = ''
    str video_path = 'videos/7.mp4'


 
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef smooth_emotions():
    global EMOTIONS, last_time_face, current_time_face
    emotions = numpy.array(['Angry', 'Background', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])
    total_temp = numpy.array([])
    cdef :
        float number_face = 0
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
                #path_face_save = ' ' .join(["face_database/", str(EMOTIONS[label_temp]), "/", str(int(current_time)), '_', str(number_face) ,".jpg"])
                #str("face_database/"+str(EMOTIONS[label_temp])+"/"+str(EMOTIONS[label_temp])+'_'+ str(int(current_time))+"_"+str(number_face) + ".jpg")
                #cv2.imwrite(path_face_save, gray_temp)

                predicted_emotion = str(emotions[label_temp])
                print(predicted_emotion)

        except queue.Empty:
            logging.warning("Empty memory!")

cdef : 
    float x 
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float_to_int(x):
    if x == float('inf') or x == float('-inf'):
        return 0.00 # or a large value you choose
    return x
 
cdef : 
    float time_to_wait,temp
    int time_start    

def get_best_images(input_frames):
    blur_threshold=100
    input_frames = sorted(input_frames, key=lambda img : cv2.Laplacian(img[0], cv2.CV_64F).var(), reverse=True)
    # Lay khung hinh co muc do focus tot nhat
    best_image = input_frames[0]
    return best_image    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef show():
    global paused, EMOTIONS, log_system, video_path
    cdef :
        str text = ''
        bint ret = True
        int i, high_level_emotion
        int count = 0
        int max_emotion_position = 0 
        numpy.uint32_t x, y, w, h
        float end_time, start_time
        numpy.ndarray frame, gray, roi

    cap = cv2.VideoCapture("videos/democlassroom.mp4")
    print("Width: %d, Height: %d, FPS: %d" % (cap.get(3), cap.get(4), cap.get(5)))

    cdef :
        int fps_of_video =  0
        int time_to_wait = 0
        int frame_counter = 0;
        int tick = 0;
        int fps = fps_of_video;
        float time_start
        bint time_checker = True
    fps_of_video = int(cap.get(cv2.CAP_PROP_FPS));
    input_frames = []
    
    while True :
        try:
            ret, frame = cap.read()

            if (cv2.waitKey(1) & 0xFF == ord('q') )or(not (ret is  True)):
                break
            frame_counter += 1
            count += 1


            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            input_frames.append(frame_gray)

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


    #out.release()    
    cap.release()
    cv2.destroyAllWindows()

@cython.boundscheck(False)
@cython.wraparound(False)
def Main():
    global EMOTIONS
    
    print("Số lượng cpu : ", multiprocessing.cpu_count())
    tReadFile = threading.Thread(target=show)
    tProcessingFile = threading.Thread(target=smooth_emotions)

    tReadFile.start()
    tProcessingFile.start()

    tProcessingFile.join()
    tReadFile.join()
    print("Bye !!!")
