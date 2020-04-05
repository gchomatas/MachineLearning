import time
import cv2
from imutils.video import VideoStream
from imutils.video import FPS
import imutils
import numpy as np

WIDTH  = 1280
HEIGHT = 720
#WIDTH = 640
#HEIGHT = 480
FRAMERATE = 120
FLIP_METHOD = 2

def gstreamer_pipeline (capture_width=WIDTH, capture_height=HEIGHT, display_width=WIDTH, display_height=HEIGHT, framerate=120, flip_method=FLIP_METHOD):   
    return ('nvarguscamerasrc ! ' 
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! '
    'videoconvert ! '
    'video/x-raw, format=(string)BGR ! appsink'  % (capture_width,capture_height,framerate,flip_method,display_width,display_height))

def gstreamer_alt_pipeline (capture_width=WIDTH, capture_height=HEIGHT, framerate=120, flip_method=FLIP_METHOD): 
    return ('nvarguscamerasrc ! '
    'video/x-raw(memory:NVMM), '
    'width=(int)%d, height=(int)%d, '
    'format=(string)NV12, framerate=(fraction)%d/1 ! '
    'nvvidconv flip-method=%d ! '
    'video/x-raw, format=(string)BGRx ! '
    'videoconvert ! '
    'appsink'  % (capture_width,capture_height,framerate,flip_method))


# create neural net from the serialized model definition and the model weights
def get_dnn():
    return cv2.dnn.readNetFromCaffe('model_definition.json', 'res10_300x300_ssd_iter_140000.caffemodel')


def detect_faces(frame):
    (h, w) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net = get_dnn()
    net.setInput(blob)
    detections = net.forward()


    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with the
        # prediction
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence < 0.5:
            continue

        # compute the (x, y)-coordinates of the bounding box for the
        # object
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # draw the bounding box of the face along with the associated
        # probability
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.rectangle(frame, (startX, startY), (endX, endY),
            (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)


videocapture = cv2.VideoCapture(gstreamer_alt_pipeline(), cv2.CAP_GSTREAMER)

time.sleep(2.0)

fps = FPS().start()

if videocapture.isOpened():

    window_handle = cv2.namedWindow('Face Detection', cv2.WINDOW_AUTOSIZE)

    while cv2.getWindowProperty('Face Detection',0) >= 0:
            ret, frame = videocapture.read()

            #gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            #frame = imutils.resize(frame, width=400)

            detect_faces(frame)

            cv2.imshow('Face Detection', frame)
            fps.update()

            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quitting...")
                break
    fps.stop()
    print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    videocapture.release()
    cv2.destroyAllWindows()

else:
    print("[FATAL]Unable to open camera\nQuitting now ...")

