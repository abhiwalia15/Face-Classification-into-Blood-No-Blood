from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from PIL import Image,ImageEnhance
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
import numpy as np
import argparse
import imutils
import time
import glob
import random
import cv2
import os


st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(suppress_st_warning=True)
@st.cache(persist=True)

# helper function to predict images
def imagepreds(image):
    
    # load our serialized face detector model from disk
    prototxtPath = os.path.sep.join(['Model', "deploy.prototxt"])
    weightsPath = os.path.sep.join(['Model', "res10_300x300_ssd_iter_140000.caffemodel"])
    net = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    model = load_model('blood_noblood_classifier.model')
        
    orig = image.copy()
    (h, w) = image.shape[:2]

    # construct a blob from the image
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    net.setInput(blob)
    detections = net.forward()

    # loop over the detections
    for i in range(0, detections.shape[2]):
        
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]
        
        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        
        if confidence > 0.6:
        
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            
            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = image[startY:endY, startX:endX]
            #face = cv2.cvtColor(face, 1)
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)
            
            # pass the face through the model to determine if the face
            # has a mask or not
            (blood, noblood) = model.predict(face)[0]


            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Blood" if blood > noblood else "No Blood"
            color = (0, 0, 255) if label == "Blood" else (0, 255, 0)
            
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(blood, noblood) * 100)
            
            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(image, label, (startX, startY - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(image, (startX, startY), (endX, endY), color, 2)
        
    # show the output image
    st.image(image, caption='Predictions', width=720)
    # cv2.imshow("Output", image)
    # cv2.waitKey(0)


# helper function to predict in real-time
def videopreds():
    # define a helper function to detected face and bounding box for each image 
    # in a live video frame
    def detect_and_predict_blood(frame, faceNet, bloodNet):
        
        # grab the dimensions of the frame and then construct a blob
        # from it
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        
        # pass the blob through the network and obtain the face detections
        faceNet.setInput(blob)
        detections = faceNet.forward()
        
        # initialize our list of faces, their corresponding locations,
        # and the list of predictions from our face mask network
        faces = []
        locs = []
        preds = []

        # loop over the detections
        for i in range(0, detections.shape[2]):
            
            # extract the confidence (i.e., probability) associated with
            # the detection
            confidence = detections[0, 0, i, 2]
            
            # filter out weak detections by ensuring the confidence is
            # greater than the minimum confidence
            if confidence > 0.5:
                
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # ensure the bounding boxes fall within the dimensions of
                # the frame
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # extract the face ROI, convert it from BGR to RGB channel
                # ordering, resize it to 224x224, and preprocess it
                face = frame[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)
                
                # add the face and bounding boxes to their respective
                # lists
                faces.append(face)
                locs.append((startX, startY, endX, endY))

                # only make a predictions if at least one face was detected
        if len(faces) > 0:
            
            # for faster inference we'll make batch predictions on *all*
            # faces at the same time rather than one-by-one predictions
            # in the above `for` loop
            faces = np.array(faces, dtype="float32")
            preds = bloodNet.predict(faces, batch_size=32)
            
        # return a 2-tuple of the face locations and their corresponding
        # locations
        return (locs, preds)

    # load our serialized face detector model from disk
    prototxtPath = os.path.sep.join(['Model', "deploy.prototxt"])
    weightsPath = os.path.sep.join(['Model', "res10_300x300_ssd_iter_140000.caffemodel"])
    faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

    # load the face mask detector model from disk
    bloodNet = load_model('blood_noblood_classifier.model')

    # initialize the video stream and allow the camera sensor to warm up
    # vs = VideoStream(src=0).start()
    
    # time.sleep(2.0)
    @st.cache(allow_output_mutation=True)
    def get_cap():
        return cv2.VideoCapture(0)

    cap = get_cap()

    frameST = st.empty()
    #param=st.sidebar.slider('chose your value')

    # loop over the frames from the video stream
    while True:
        
        # grab the frame from the threaded video stream and resize it
        # to have a maximum width of 400 pixels
        # frame = vs.read()
        # frame = imutils.resize(frame, width=400)
        ret, frame = cap.read()
        # detect faces in the frame and determine if they are wearing a
        # face mask or not
        (locs, preds) = detect_and_predict_blood(frame, faceNet, bloodNet)

        # loop over the detected face locations and their corresponding
        # locations
        for (box, pred) in zip(locs, preds):
            
            # unpack the bounding box and predictions
            (startX, startY, endX, endY) = box
            (blood, noblood) = pred
            
            # determine the class label and color we'll use to draw
            # the bounding box and text
            label = "Blood" if blood > noblood else "No Blood"
            color = (0, 0, 255) if label == "Blood" else (0, 255, 0)
            
            # include the probability in the label
            label = "{}: {:.2f}%".format(label, max(blood, noblood) * 100)
            
            # display the label and bounding box rectangle on the output
            # frame
            cv2.putText(frame, label, (startX, startY - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        # show the output frame
        # cv2.imshow("Frame", frame)
        # key = cv2.waitKey(1) & 0xFF
        frameST.image(frame, channels="BGR")
        
        # if the `q` key was pressed, break from the loop
        # if key == ord("q"):
        #     break
            
    # do a bit of cleanup
    cv2.destroyAllWindows()
    #vs.stop()

@st.cache(persist=True)
def load_image(img):
    im = Image.open(img)
    return im


ig = Image.open('sample.jpg')
st.image(ig, width=920)

st.title('Blood Detection Classifier')

st.text('Build with Streamlit,Tensorflow, Keras and OpenCV By Mrinal Walia')

st.header("Select the options from sidebar: ")

st.subheader("Image Detection: For uploading an Image")

st.subheader("Video Detection: For opening the webcam and checking the results")

st.subheader("Performance Metrics: To check various performance metrices")

st.markdown("THANKS FOLKS!!")

st.subheader("Happy Learning")

st.subheader("Creator: @MRINAL WALIA")

st.markdown('Source Code Link: https://github.com/abhiwalia15/Face-Classification-into-Blood-No-Blood')

def main():

	menu = ['Image Detection', 'Video Detection', 'Perofrmance Metrics']
	choice = st.sidebar.selectbox('Menu',menu)

	if choice == 'Image Detection':
		st.subheader('**Blood Detection**')
		img_file_buffer = st.file_uploader("Upload an image")
		
		if img_file_buffer is not None:
			image = Image.open(img_file_buffer)
			img_array = np.array(image)
			imagepreds(img_array)

	elif choice == 'Video Detection':
		st.subheader('**Blood Detection**')
		videopreds()

	elif choice == 'Perofrmance Metrics':

		st.subheader('About Performance Metrics')

		st.info("CLASSIFICATION REPORT")
		cr = Image.open('Results/classification_report.png')
		st.image(cr, width=320)

		st.info("CONFUSION MATRIX")
		cm = Image.open('Results/confusion_matrix.png')
		st.image(cm, width=320)

		st.info("ACCURACY SCORE")
		acs = Image.open('Results/accuracy_score.png')
		st.image(acs, width=320)


# driver code
if __name__ == '__main__':
    main()
