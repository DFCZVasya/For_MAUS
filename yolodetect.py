from PIL import Image
from yolo import YOLO
from imutils.video import VideoStream
import cv2
import time
import imutils
from resizevideo import take_and_resize
import numpy as np

class Conductor():
    def __init__(self):
        self.check_dict = {
            'clock': False, 
            'cup' : False, 
            'bottle' : False, 
            'apple' : False, 
            'banana' : False, 
            'orange' : False, 
            'person': False, 
            'cell phone' : False, 
            'book' : False, 
            'keyboard' : False, 
            'mouse' : False, 
            'knife' : False
        }

    def update_dict(self, class_name, status):
        self.check_dict[class_name] = status

    def make_all_false(self):
        for i in self.check_dict:
            self.check_dict[i] = False

    def get_dict(self):
        return self.check_dict
    


def YoloThread(output_obj):
    for_fing = ['clock', 'cup', 'bottle', 'apple', 'banana', 'orange', 'person', 'cell phone', 'book', 'keyboard', 'mouse', 'knife']
    vs = VideoStream(src=0).start() #or 1
    time.sleep(2.0)
    yolo = YOLO('model_data/yolo-tiny.h5', 'model_data/tiny_yolo_anchors.txt', 'model_data/coco_classes.txt')

    while True:
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 314 pixels
        frame = vs.read()
        #frame = imutils.resize(frame, width=314)
        
        #take frame and return square(for wide angle cameras) in the desired resolution default is (314, 314)
        frame = take_and_resize(frame) #here you can also choose your own ouput resolution like take_and_resize(frame, your resolution) 
        
        # loop over the detections
        outBoxes = yolo.detect_image(frame) #Here you can make whatever you want (return[top left x, top left y, bottom right x, bottom right y, class name])
        output_classes = []
        frame = np.asarray(frame)
        if len(outBoxes) > 0:
            for box in outBoxes:
                # extract the bounding box coordinates
                (x, y) = (int(box[0]), int(box[1]))
                (w, h) = (int(box[2]), int(box[3]))
                bbox = [x, y, w, h, box[4]]
                output_classes.append(bbox[4])
                cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0,255,0), 2)
                text = 'classID = {}'.format(box[4])
                cv2.putText(frame, text, (box[0], box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2)
                if bbox[4] in for_fing:
                    output_obj.update_dict(bbox[4], True)
            
            check_dict = output_obj.get_dict()
            for i in check_dict:
                if i not in output_classes:
                    output_obj.update_dict(i, False)

        else:
            output_obj.make_all_false()




        # show the output frame
        #frame = imutils.resize(frame, width=500)
        cv2.imshow("Frame", frame)


        key = cv2.waitKey(1) & 0xFF

        # if the `q` key was pressed, break from the loop
        if key == ord("q"):
            break

    # do a bit of cleanup
    cv2.destroyAllWindows()
    vs.stop()
    yolo.close_session()



