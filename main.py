# import tensorflow.keras as keras
import sys
import cv2
from facenet_pytorch import MTCNN
from PIL import Image, ImageDraw , ImageFont
from tensorflow.keras.models import model_from_json
import numpy as np

categories = ['with mask' , 'without mask']
#loading the json_file

def load_model():
    print('[INFO] loading the model')
    load_json = open('mask_detection.json' , 'r')
    load_facemask_detection_model = load_json.read()
    facemask_detection_model = model_from_json(load_facemask_detection_model)
    facemask_detection_model.load_weights('mask_detection.h5')
    print('[INFO] model loaded')
    return facemask_detection_model

def webcam():
    #load the face mask model
    facemask_detection_model = load_model()

    #load the face detection model
    print('[INFO] loading the face detectiom model')
    mtcnn = MTCNN( keep_all = True , post_process = False , image_size=224)
    print('[INFO] loading face detectiom model')


    cap = cv2.VideoCapture(0)
 
    while cap:
        ret , frame = cap.read()
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        try:
            boxes, _ = mtcnn.detect(frame_rgb)#get bounding boxes boxes cordinates
            
            # fnt = ImageFont.truetype('../../../../../../../../../../usr/share/fonts/truetype/Sarai/Sarai.ttf', 25)
            frame_draw = Image.fromarray(frame_rgb)
            draw = ImageDraw.Draw(frame_draw)
            #detect the faces
            faces = mtcnn(frame_rgb)
            for i in range(len(faces)):
                detected_face = faces[i].permute(1,2,0).int().numpy()
                output = facemask_detection_model(detected_face.reshape(1 , 224 , 224 , 3))
                if categories[np.argmax(output)] == 'without mask':
                    frame = cv2.rectangle(frame , (boxes[i][0] , boxes[i][1]) , (boxes[i][2],boxes[i][3]) , (0, 255, 0) , 2)
                    frame = cv2.putText(frame, categories[np.argmax(output)] , (int(boxes[i][0]+15) , int(boxes[i][1]-10))  , cv2.FONT_HERSHEY_SIMPLEX  , 1 , (255, 0, 0) , 2)
                else:
                    frame = cv2.rectangle(frame , (boxes[i][0] , boxes[i][1]) , (boxes[i][2],boxes[i][3]) , (0, 0, 255) , 2)
                    frame = cv2.putText(frame, categories[np.argmax(output)] , (int(boxes[i][0]+15) , int(boxes[i][1]-10)) , cv2.FONT_HERSHEY_SIMPLEX  , 1 , (255, 0, 0) , 2)
                # draw.text((boxes[i][0]-20,boxes[i][1]-20), "Distracted", fill=(0,255,0))
                cv2.imshow('frame' , frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
        except:
            print(f'{sys.exc_info()[0]}')
            continue
        # cap.release()
        # cv2.destroyAllWindows()

webcam()
