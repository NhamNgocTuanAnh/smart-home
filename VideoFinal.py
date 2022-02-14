import os, sys
import cv2
import time
import numpy 
import pytz
UTC = pytz.timezone('UTC') # utc
from datetime import datetime as dt
from csv import DictWriter
from keras.models import model_from_json
from keras.preprocessing.image import img_to_array
import logging

import threading
from multiprocessing import Process, Manager, cpu_count, set_start_method

import psutil
num_cpus = psutil.cpu_count(logical=False)


from quick_queue import QQueue
class Video_Process:
    def __init__(self):

        self.input_image_time_buffer = QQueue()
        self.save_image_buffer = QQueue()

        self.field_names = ["Timestamp", 'Angry', 'Background','Disgusted','Fearful','Happy','Neutral','Sad','Surprised']
        self.EMOTIONS = numpy.array(["angry", "background", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"])  
        
        self.total_preds = numpy.array([])
        self.stop = False
        self.face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
        
        self.emotions = ('Angry', 'Background','Disgusted','Fearful','Happy','Neutral','Sad','Surprised')
        # model_facial_expression = load_model('model/_mini_XCEPTION.106-0.65.hdf5')

        # load model
        self.model_facial_expression = model_from_json(open("model/model_fernet.json", "r").read())
        #load weights
        self.model_facial_expression.load_weights('model/model_fernet.h5')

    # @njit(fastmath=True, cache=True)
    def detectEmotions(self,img_pixels : numpy.array([])) -> numpy.array([]):

        img_pixels = numpy.expand_dims(img_pixels, axis = 0)
        img_pixels =numpy.true_divide(img_pixels ,255)
        
        return img_pixels


    # @njit(fastmath=True, cache=True)
    def calculatorPred(self,total_temp: numpy.array([]), number_face: float())-> numpy.array([]):
        total_preds = numpy.true_divide(total_temp ,(number_face-1))
        return total_preds

    # @njit(fastmath=True, cache=True)
    def calculatorBe(self,total_temp: numpy.array([]), preds: numpy.array([]))-> numpy.array([]): 
        if total_temp is not None:
            return numpy.add(total_temp, preds)
        return preds

    def findAndSave(self):
        p1 = Process(target=self.saveImage)
        
        p2 = Process(target=self.smoothEmotions)

        p2.start()
        p1.start()
        p2.join()
        p1.join()
               # predicted_emotion = str(self.EMOTIONS[label_temp])
               #
               #  now = dt.now(pytz.timezone('Asia/Ho_Chi_Minh'))
               #  time_now = now.strftime("%T")
               #  date_now = now.strftime("%d %B %Y")
               #  dict_temp = {}
               #  # self.field_names = ["Timestamp", 'Angry', 'Background','Disgust','Fear','Happy','Neutral','Sad','Surprise']
               #
               #  dict = {'Timestamp': str(time_now), 'Angry': str(preds[0]),'Background': str(preds[1]), 'Disgusted': str(preds[2]),
               #          'Fearful': str(preds[3]), 'Happy': str(preds[4]),
               #          'Neutral': str(preds[5]), 'Sad': str(preds[6]), 'Surprised': str(preds[7])}
               #  with open('data.csv', 'a') as f_object:
               #      dictwriter_object = DictWriter(f_object, fieldnames=self.field_names)
               #      dictwriter_object.writerow(dict)
               #      f_object.close()
    def saveImage(self):
        number_face = int(0)
        img_copy, preds = numpy.array([]), numpy.array([])
        while 1 and not self.stop:
            try:
                (img_copy, preds) = self.save_image_buffer.get()
                number_face += 1
                current_time = time.time()
                label_temp = int(preds.argmax())


                path_face_save = str("face_database/"+str(self.EMOTIONS[label_temp])
                                     +"/"+str(self.EMOTIONS[label_temp])+'_'
                                     +str(int(current_time))+"_"
                                     +str(number_face) + ".jpg")
                cv2.imwrite(path_face_save, img_copy)

            except Exception as e: 
                print(e)
                print("Có ngoại lệ ",sys.exc_info()[0]," xảy ra.")
                pass

    def smoothEmotions(self):
        total_temp, img_pixels, roi, img_copy = numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
        last_time_face = float(-1)
        number_face = float(0)

        while 1 and not self.stop:
        # cls()

            try:  
                (img_pixels, current_time) = self.input_image_time_buffer.get()
                img_copy = img_pixels.copy()

                img_pixels = img_to_array(img_pixels)
                
                img_pixels = numpy.expand_dims(img_pixels, axis = 0)
                roi = numpy.true_divide(img_pixels ,255)

                preds = self.model_facial_expression.predict(roi)[0]
                label_temp = int(preds.argmax())

                # self.saveImage(img_copy, preds)
                self.save_image_buffer.put((img_copy, preds))
                self.total_preds = preds

                # if label_temp < 7:
                #     number_face += 1
                #     if current_time != last_time_face:
                #         if not(total_temp is None) and (number_face>1):
                #             self.total_preds = None

                #             self.total_preds = numpy.array(self.calculatorPred(total_temp, number_face))
                            
                #             print(str(int(self.total_preds.argmax())))
                #         number_face = float(0)
                #         last_time_face = current_time
                #         total_temp = None
                #     else: 
                        
                #         # total_temp = numpy.add(total_temp, preds)
                #         total_temp = numpy.array(self.calculatorBe(total_temp, preds))


            except Exception as e: 
                print(e)
                print("Có ngoại lệ ",sys.exc_info()[0]," xảy ra.")
                pass
    def floatToInt(self,x) ->float:
        if x == float('inf') or x == float('-inf'):
            return 0.00 # or a large value you choose
        return x



    def calculateFrameDelay(self ,time_to_wait , time_start)->bool: 
        # Calculate frame delay, in order to make the video look smoother.
        # When fps is higher, should use a smaller ratio, or fps will be limited in a lower value.
        # Larger ratio can make the video look smoother, but fps will hard to become higher.
        # Smaller ratio can make fps higher, but the video looks not too smoother.
        # The ratios below are tested many times.
        temp = float(0)
        temp = (float(cv2.getTickCount()) - time_start) / float(cv2.getTickFrequency()) * 1000
        if time_to_wait > temp:
            return True   
        return False
    def face_detect(self, frame_gray):
        faces_detected = self.face_haar_cascade.detectMultiScale(frame_gray, 1.32, 5)
        return faces_detected
    def focus_measure_caculator(self,gray):
        return cv2.Laplacian(gray, cv2.CV_64F).var()
    def get_best_images(self,input_frames):
        # Sap xep cac input_frames theo muc do Focus giam dan
        # Dinh nghia blur threshold
        blur_threshold=100

        # result = map(lambda frame: self.focus_measure_caculator(frame), input_frames)
        # result = {}
        # for frame_temp in input_frames:
        #     result[frame_temp] = self.focus_measure_caculator(frame_temp)
        input_frames = sorted(input_frames, key=lambda img : cv2.Laplacian(img[0], cv2.CV_64F).var(), reverse=True)
        # Lay khung hinh co muc do focus tot nhat
        best_image = input_frames[0]
        return best_image
    def show(self,path_video:str()):
        x, y, w, h = numpy.int32 ,numpy.int32,numpy.int32,numpy.int32
        max_emotion_position = numpy.int64
        count, i, high_level_emotion = numpy.int32 ,numpy.int32,numpy.int32
        frame, gray, faces, roi, total_preds_fake = numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([]), numpy.array([])
        time_checker = True
        input_frames = []
        count, i, high_level_emotion, frame_counter, tick = int(0), int(0), int(0), int(0), int(0)
        
        # # model_facial_expression = load_model('model/model_v6_23.hdf5')
        # model_facial_expression = model_from_json(open("model/model_fernet.json", "r").read())
        # #load weights
        # model_facial_expression.load_weights('model/model_fernet.h5')

        cap = cv2.VideoCapture(path_video)
        # out = cv2.VideoWriter('output_'+time.strftime("%d-%m-%Y-%H-%M-%S")+'.avi', cv2.VideoWriter_fourcc(*'XVID'), 20.0, (640,480))

        fps_of_video = int(cap.get(cv2.CAP_PROP_FPS));
        fps = fps_of_video;
        if fps_of_video < 1:
            fps_of_video = 1
        time_to_wait = int(1000 / fps_of_video);
        time_begin = time.time()

        tmp_time = time.time()
        fps_list = list()
            # Number of workers (subprocess use to process frames)
        if cpu_count() > 2:
            worker_num = cpu_count() - 1  # 1 for capturing frames
        else:
            worker_num = 2

        while not self.stop:
            try:
                ret, frame = cap.read()# captures frame and returns boolean value and captured image
                if (ret is False) or (frame is None) or (cv2.waitKey(1) & 0xFF == ord('q') ):
                    self.stop = True
                    cap.release()
                    # out.release()
                    cv2.destroyAllWindows
                    break
                if not(self.total_preds is None): 
                    
                    total_preds_fake = self.total_preds

                # frame = cv2.resize(frame, (480, 320))

                # Calculate fps
                delay = time.time() - tmp_time
                tmp_time = time.time()
                fps_list.append(delay)
                if len(fps_list) > 5 * worker_num:
                    fps_list.pop(0)
                fps = len(fps_list) / numpy.sum(fps_list)
                print("fps: %.2f" % fps)


                time_now = time.time() - time_begin
                time_start = float(cv2.getTickCount())
                if time_now - tick > 0 :
                    tick += 1
                    fps = frame_counter
                    frame_counter = int(0)
                frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                input_frames.append(frame_gray)

                if count%5==0:
                    gray_img = self.get_best_images(input_frames)
                    input_frames.clear()
                    faces_detected = self.face_detect(gray_img)
                    # print(str(type(faces_detected)))
                    current_time = time.time()
                    
                    for (x,y,w,h) in faces_detected:
                        roi = gray_img[y:y + h, x:x + w]
                        roi = cv2.resize(roi, (48, 48)) 
                        self.input_image_time_buffer.put((roi,current_time))

                        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)

                if not(total_preds_fake is None) and numpy.size(total_preds_fake)>0:
                
                    max_emotion_position = numpy.int64(total_preds_fake.argmax())
                    # EMOTIONS = numpy.array(["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral", "background"])  
                    for (i, (emotion, prob)) in enumerate(zip(self.EMOTIONS, total_preds_fake)):

                        #construct the label text
                        text = "{}: {:.2f}%".format(emotion, prob * 100)
                
                        color = (255, 0, 0)          
                        if max_emotion_position == i:
                            color = (225,225,225)
                        high_level_emotion = int(self.floatToInt(prob) * 300)
                        cv2.rectangle(frame, (7, (i * 35) + 5),
                                    (high_level_emotion, (i * 35) + 35), (0, 0, 255), -1)
                        cv2.putText(frame, text, (10, (i * 35) + 23), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
                
                time_checker = self.calculateFrameDelay(time_to_wait, time_start)

                while time_checker:
                    #http://stackoverflow.com/questions/1157209/is-there-an-alternative-sleep-function-in-c-to-milliseconds
                    time_checker = self.calculateFrameDelay(time_to_wait, time_start)
                    time.sleep(0.00005);
            
                # resized_img = cv2.resize(frame, (1000, 700))
                frame = cv2.resize(frame, (480, 320))
                cv2.imshow('Facial emotion analysis ',frame)
                count += 1
                # out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            except Exception as e: 
                print(e)
                print("Có ngoại lệ ",sys.exc_info()[0]," xảy ra.")
                pass
        cap.release()
        # out.release()

        cv2.destroyAllWindows
    def run(self,path_video:str()):
        tShowVideo = threading.Thread(target=self.show, args=(path_video,))
        tProcessingEmotion = threading.Thread(target=self.smoothEmotions)
        tsaveImage = threading.Thread(target=self.saveImage)

        tShowVideo.start()
        tProcessingEmotion.start()
        tsaveImage.start()

        tProcessingEmotion.join()
        tShowVideo.join()
        tsaveImage.join()
def assure_path_exists(path):
    dir = os.path.dirname(path)
    if not os.path.exists(dir):
        os.makedirs(dir)
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.info("Main    : before creating thread")
    # raw_data = {}
    # # makes the passed rows header
    # df = pd.DataFrame(raw_data, columns =["Timestamp", 'Angry', 'Background','Disgust','Fear','Happy','Neutral','Sad','Surprise'])
    # df.to_csv('data.csv', index=False)

    EMOTIONS = numpy.array( ["angry", "background", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"])
    assure_path_exists("face_database/")
    for emotion_index in range(0, len(EMOTIONS)):
        assure_path_exists("face_database/"+EMOTIONS[emotion_index]+"/")

    # for r1, d1, f1 in os.walk("downloads/"):
    #     for file_v1 in f1:
    #         # print("so anh tim duoc: " +str(count_image))
    #         if '.mp4' in file_v1:
    #             print("processing video ")
                
    # path_img_1 = str(os.path.join(r1, file_v1))
    # play_video(path_img_1)
    d = Video_Process()
    logging.info("Main    : before running thread")
    d.run('http://192.168.0.101:2222/video')

    logging.info("Main    : wait for the thread to finish")
    logging.info("Main    : all done")

