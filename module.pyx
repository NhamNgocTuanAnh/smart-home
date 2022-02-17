# https://askubuntu.com/questions/1056314/uploading-code-to-arduino-gives-me-the-error-avrdude-ser-open-cant-open-d
# https://cachdung.com/blog/tu-lam-mot-robot-de-thuong.html
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
import logging
import threading, queue, multiprocessing
import pyfirmata

from multiprocessing import Process, Manager, cpu_count, set_start_method

input_image_time_buffer = queue.Queue()


#EMOTIONS = numpy.array(["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral", "background"])     
        
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int smooth_emotions(path:str):
    #global EMOTIONS
    
    # load model facial_expression
    #model_facial_expression = model_from_json(open("model/fer.json", "r").read())
    # load weights facial_expression
    #model_facial_expression.load_weights('model/fer.h5')
    
    #emotions = numpy.array(['Angry', 'Background', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise'])

    cdef :
        numpy.ndarray roi, gray_temp,frame1,frame2
        str path_face_save
        int label_temp = 0
        int count = 0
    
    while True:
        try:
            (gray_temp, current_time) = input_image_time_buffer.get()
            blink_led()

            #roi = numpy.expand_dims(img_to_array(gray_temp.astype("float") / 255.0), axis=0)

            #preds = model_facial_expression.predict(roi)[0]
            #if preds.argmax() < 7:
                #label_temp = int(preds.argmax())
                #path_face_save = ' ' .join(["face_database/", str(EMOTIONS[label_temp]), "/", str(int(current_time)), '_' ,".jpg"])
                #str("face_database/"+str(EMOTIONS[label_temp])+"/"+str(EMOTIONS[label_temp])+'_'+ str(int(current_time))+"_"+str(number_face) + ".jpg")
            
		
        except queue.Empty:
            logging.warning("Empty memory!")
            pass
    return 1
            
def blink_led()->int:
    #print("find face")
    try:
        board = pyfirmata.Arduino('/dev/ttyACM0')
        board.digital[13].write(1)
        time.sleep(0.1)
        board.digital[13].write(0)
        time.sleep(0.1)
    except OSError as e:
        raise Exception("Arduino not found")

    return 1;
    
def get_best_images(input_frames:[])-> numpy.array([]):
    blur_threshold=int(100)
    input_frames = sorted(input_frames, key=lambda img : cv2.Laplacian(img[0], cv2.CV_64F).var(), reverse=True)
    # Lay khung hinh co muc do focus tot nhat
    best_image = input_frames[0]
    return best_image    
    
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef int show(path:str):
    cdef :
        bint ret = True
        int i
        int count = 0
        int countdown = 300
        numpy.uint32_t x, y, w, h
        numpy.ndarray frame, gray, roi, frame1, frame2
        float time_start = 0
        bint time_checker = True
        bint moving = False

    cap = cv2.VideoCapture(0)
    _, frame1 = cap.read()
    _, frame2 = cap.read()
    
    print("Width: %d, Height: %d, FPS: %d" % (cap.get(3), cap.get(4), cap.get(5)))
    input_frames = []
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")    
    while True :
        try:
            ret, frame = cap.read()
            
            if (moving == False) and (ret is True):
                difference = cv2.absdiff(frame1,frame2)
                gray_image_diff = cv2.cvtColor(difference,cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray_image_diff,(25,25),0)
                ret,thresh = cv2.threshold(blur,18,255,cv2.THRESH_BINARY)
                dilated = cv2.dilate(thresh,None,iterations=3)
                contours, _ = cv2.findContours(dilated,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
                if contours is None:
                    print('No human found!')
                else:
                    for contour in contours:
                        if cv2.contourArea(contour)>200:
                            print("Found the human!")
                            countdown = int(300)
                            moving = True
                            break
                        
                #cv2.imshow("Detect",frame1)
                frame1=frame2
                _,frame2 = cap.read()
            elif (ret is True) and (moving == True):
                #print('Turn on detect Face!')				
                count += 1
                countdown -= 1

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
                    
                    #cv2.imshow("Image", frame)
                    #if cv2.waitKey(1) & 0xFF == ord('q'):
                        #break
                if countdown == 0:
                    moving = False
            elif (not (ret is  True)):
                logging.warning("Missing video!")
                pass

        except queue.Full:
            logging.warning("full memory!")
            pass
 
    cap.release()
    cv2.destroyAllWindows()
    return 1

@cython.boundscheck(False)
@cython.wraparound(False)
def Main():
    print("Số lượng cpu : ", multiprocessing.cpu_count())
    tReadFile = multiprocessing.Process(target=show, args=("videos/test.mp4",))
    tProcessingFile = multiprocessing.Process(target=smooth_emotions, args=("videos/test.mp4",))
    
    #tReadFile = threading.Thread(target=show)
    #tProcessingFile = threading.Thread(target=smooth_emotions)

    tReadFile.start()
    tProcessingFile.start()

    tProcessingFile.join()
    tReadFile.join()
    print("Bye !!!")
