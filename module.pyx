
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

input_image_time_buffer = queue.Queue(50)

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
    global EMOTIONS, total_preds, finished,last_time_face, current_time_face
    total_temp = numpy.array([])
    cdef :
        float number_face = 0
        numpy.ndarray roi, gray_temp
        str path_face_save
        int label_temp = 0
    while not finished:
        try:
            (gray_temp, current_time) = input_image_time_buffer.get(timeout=1)

            roi = numpy.expand_dims(img_to_array(gray_temp.astype("float") / 255.0), axis=0)
     
            preds = model_facial_expression.predict(roi)[0]
            if preds.argmax() < 7:
                number_face += 1
                label_temp = int(preds.argmax())
                #path_face_save = ' ' .join(["face_database/", str(EMOTIONS[label_temp]), "/", str(int(current_time)), '_', str(number_face) ,".jpg"])
                #str("face_database/"+str(EMOTIONS[label_temp])+"/"+str(EMOTIONS[label_temp])+'_'+ str(int(current_time))+"_"+str(number_face) + ".jpg")
                #cv2.imwrite(path_face_save, gray_temp)

                if last_time_face == -1:
                    last_time_face = current_time
                    total_temp = preds
                    
                if current_time != last_time_face:
                    last_time_face = current_time
                    if (number_face != 0) and not(total_temp is None):
                        total_preds = total_temp /(number_face-1)
 
                    number_face = 0
                    total_temp = preds
                else:
                    if not(total_temp is None):
                        total_temp += preds

        except queue.Empty:
            logging.warning("Empty memory!")


# Có một số yếu tố khiến mã chậm hơn như đã thảo luận trong tài liệu Cython đó là:
# Kiểm tra giới hạn để đảm bảo các chỉ số nằm trong phạm vi của mảng.
# Sử dụng các chỉ số âm để truy cập các phần tử mảng.
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

    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef show():
    # detection_buffer = Queue()
    global paused, finished, EMOTIONS, total_preds, log_system, video_path
    cdef :
        str text = ''
        bint ret = True
        int i, high_level_emotion
        int count = 0
        int max_emotion_position = 0 
        numpy.uint32_t x, y, w, h
        float end_time, start_time
        numpy.ndarray frame, gray, roi


    # initialize video source, default 0 (webcam)

    cap = cv2.VideoCapture("videos/democlassroom.mp4")
    #cap = cv2.VideoCapture(0)
    print("Width: %d, Height: %d, FPS: %d" % (cap.get(3), cap.get(4), cap.get(5)))
    print("Số lượng cpu : ", multiprocessing.cpu_count())
    #fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    #out = cv2.VideoWriter('%s_output_opencv_dnn.mp4' % (video_path.split('.')[0]), fourcc, cap.get(cv2.CAP_PROP_FPS), (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))))

    cdef :
        int fps_of_video =  0
        int time_to_wait = 0
        int frame_counter = 0;
        int tick = 0;
        int fps = fps_of_video;
        float time_start
        bint time_checker = True
    fps_of_video = int(cap.get(cv2.CAP_PROP_FPS));
    time_to_wait = int(1000 / fps_of_video);
    time_begin = time.time()
    while True :
        try:
            ret, frame = cap.read()
            # Kieru gì <ret 'bool'>
            # Kieru gì <frame 'numpy.ndarray'>

            if (cv2.waitKey(1) & 0xFF == ord('q') )or(not (ret is  True)):
                finished = True
                break
            frame_counter += 1
            count += 1
            time_now = time.time() - time_begin
            time_start = float(cv2.getTickCount())
            if time_now - tick > 0 :
                tick += 1
                fps = frame_counter
                frame_counter = 0
            

            if count % 5 ==0:
                # Our operations on the frame come here

                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30),
                                                        flags=cv2.CASCADE_SCALE_IMAGE)
                i = int(0)
                for i in range(0,len(faces)):
          
                    x = faces[i][0]
                    y = faces[i][1]
                    w = faces[i][2]
                    h = faces[i][3]

                    #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
                    #cv2.imshow('frame', frame)
                    
                    roi = gray[y:y + h, x:x + w]
                    roi = cv2.resize(roi, (48, 48))
                    input_image_time_buffer.put((roi,time_start), timeout=1)
                print(count)   
        
        except queue.Full:
            logging.warning("full memory!")
            pass


    #out.release()    
    cap.release()
    cv2.destroyAllWindows()

@cython.boundscheck(False)
@cython.wraparound(False)
# Using cython compiler directives to remove some of the checks that numpy usually has to make
# Use typed memoryviews so that I can specify memory layout (and sometimes they are faster in general compared to the older buffer interface)
# Unrolled the loops so that we don't use numpy's slice machinary:
def Main():
    global gray_temp, EMOTIONS
    tReadFile = threading.Thread(target=show)
    tProcessingFile = threading.Thread(target=smooth_emotions)

    tReadFile.start()
    tProcessingFile.start()

    tProcessingFile.join()
    tReadFile.join()
    print("Bye !!!")
