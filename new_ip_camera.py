# Using Android IP Webcam video .jpg stream (tested) in Python2 OpenCV3

import urllib
import cv2
import numpy as np
import time
import sys
import urllib.request
if sys.version_info[0] == 3:
    from urllib.request import urlopen
else:
    # Not Python 3 - today, it is most likely to be Python 2
    # But note that this might need an update when Python 4
    # might be around one day
    from urllib import urlopen
# Replace the URL with your own IPwebcam shot.jpg IP:port
url='http://192.168.1.59:8080/shot.jpg'


while True:
    # Use urllib to get the image from the IP camera
    imgResp = urllib.request.urlopen(url)

    # Numpy to convert into a array
    imgNp = np.array(bytearray(imgResp.read()),dtype=np.uint8)

    # Finally decode the array to OpenCV usable format ;)
    img = cv2.imdecode(imgNp,-1)


	# put the image on screen
    cv2.imshow('IPWebcam',img)

    #To give the processor some less stress
    #time.sleep(0.1)

    # Quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# import cv2
#
# cap = cv2.VideoCapture('http://192.168.1.59:8080/shot.jpg')
#
# while(True):
#     ret, frame = cap.read()
#     if ret:
#         cv2.imshow('frame',frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             cv2.destroyAllWindows()
#             break
