from multiprocessing import Pool, Queue
import time
import cv2,numpy
import sys
import urllib.request
if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlopen
from keras.models import model_from_json
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
# intialize global variables for the pool processes:
def init_pool(d_b):
    global detection_buffer
    detection_buffer = d_b
count = int(0)
cap = cv2.VideoCapture("democlassroom.mp4")
fps_of_video = int(cap.get(cv2.CAP_PROP_FPS));
if fps_of_video < 1:
    fps_of_video = 1
time_to_wait = int(1000 / fps_of_video);
def detect_object(frame):
    #ime.sleep(1)
    global count,time_to_wait
    # if count%5==0 or True:
    gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +'haarcascade_frontalface_default.xml')
    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)
    for (x,y,w,h) in faces_detected:
        # cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
        # roi = gray_img[y:y + h, x:x + w]
        # roi = cv2.resize(roi, (48, 48))
        # # self.input_image_time_buffer.put((roi,current_time))
        # img_pixels = image.img_to_array(roi)
        # img_pixels = numpy.expand_dims(img_pixels, axis = 0)
        # img_pixels = numpy.true_divide(img_pixels ,255)
        # # load model
        # model_facial_expression = model_from_json(open("model/model_fernet.json", "r").read())
        # #load weights
        # model_facial_expression.load_weights('model/model_fernet.h5')
        # # if count%5==0:
        # predictions = model_facial_expression.predict(img_pixels)
        #
        # # find max indexed array
        # max_index = numpy.argmax(predictions[0])
        # emotions = ('Angry', 'Background','Disgust','Fear','Happy','Neutral','Sad','Surprise')
        #
        # predicted_emotion = str(emotions[max_index])
        # cv2.putText(frame, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),thickness=7)
    # time_start = float(cv2.getTickCount())
    # time_checker = calculateFrameDelay(time_to_wait, time_start)
    # while time_checker:
    #     #http://stackoverflow.com/questions/1157209/is-there-an-alternative-sleep-function-in-c-to-milliseconds
    #     time_checker = calculateFrameDelay(time_to_wait, time_start)
    #     time.sleep(0.00005);
    detection_buffer.put(frame)
    count += 1
    print(count)
def calculateFrameDelay(time_to_wait , time_start)->bool:
    # Calculate frame delay, in order to make the video look smoother.
    # When fps is higher, should use a smaller ratio, or fps will be limited in a lower value.
    # Larger ratio can make the video look smoother, but fps will hard to become higher.
    # Smaller ratio can make fps higher, but the video looks not too smoother.
    # The ratios below are tested many times.
    temp = (float(cv2.getTickCount()) - time_start) / float(cv2.getTickFrequency()) * 1000
    if time_to_wait > temp:
        return True
    return False
def show():
    global fps_of_video, time_to_wait
    while True:

        frame = detection_buffer.get()
        if frame is not None:
            time_start = float(cv2.getTickCount())
            # time_checker = calculateFrameDelay(time_to_wait, time_start)
            # while time_checker:
            #     #http://stackoverflow.com/questions/1157209/is-there-an-alternative-sleep-function-in-c-to-milliseconds
            #     time_checker = calculateFrameDelay(time_to_wait, time_start)
            #     time.sleep(0.00005);
            cv2.imshow("Video", frame)
        else:
            break
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    return


# required for Windows:
if __name__ == "__main__":

    detection_buffer = Queue()
    # 6 workers: 1 for the show task and 5 to process frames:
    pool = Pool(6, initializer=init_pool, initargs=(detection_buffer,))
    # run the "show" task:
    show_future = pool.apply_async(show)
    futures = []
    url='http://192.168.1.59:2222/shot.jpg'
# cap = cv2.VideoCapture('http://192.168.0.101:2222/video')
    cap = cv2.VideoCapture('9.mp4')
    while True:
            # Use urllib to get the image from the IP camera
        # imgResp = urllib.request.urlopen(url)
        #
        # # Numpy to convert into a array
        # imgNp = numpy.array(bytearray(imgResp.read()),dtype=numpy.uint8)
        #
        # # Finally decode the array to OpenCV usable format ;)
        # frame = cv2.imdecode(imgNp,-1)
        #
        # f = pool.apply_async(detect_object, args=(frame,))
        # futures.append(f)
        ret, frame = cap.read()
        if ret:
            # time_start = float(cv2.getTickCount())
            # time_checker = calculateFrameDelay(time_to_wait, time_start)
            # while time_checker:
            #     #http://stackoverflow.com/questions/1157209/is-there-an-alternative-sleep-function-in-c-to-milliseconds
            #     time_checker = calculateFrameDelay(time_to_wait, time_start)
            #     time.sleep(0.00005);
            f = pool.apply_async(detect_object, args=(frame,))
            futures.append(f)
            #time.sleep(0.025)
        else:
            break
    # wait for all the frame-putting tasks to complete:
    for f in futures:
        f.get()
    # signal the "show" task to end by placing None in the queue
    detection_buffer.put(None)
    show_future.get()
    print("program ended")
