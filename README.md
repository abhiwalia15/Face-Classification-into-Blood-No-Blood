# Face-Classification-into-Blood-No-Blood

# Blood face detection

## Steps to run:
* You must have streamlit in order to run the web application: pip install streamlit
* then open command terminal and type: streamlit run app.py
* This will open up the application in a web browser and you can run the app.

## Problem Statement:
 To train a model that classifies images based on the two labels i.e blood face and no-blood face.

## Steps: 

1. Preprocessing of data:
‘data_prep.py’ contains the script for preprocessing performed on the initial dataset. Each image in the 'blood' and 'no blood' folder was cropped just to the faces in each image and saved in the respective folders.

2. Training of Model: 
	* 'model_train.ipynb' file contains the code to the jupyter notebook for building the model and training the model using Mobilenet V2 architecture [pretrained on imagenet weights] and transfer learning. The model was trained on 30 epochs with a batch size of 32 initially and the accuracy of ___ was achieved.
	* For performance evaluation classification report, accuracy score and confusion matrix is printed. Also, the training and validation [ accuracy and loss] plot is also plotted for better visualization of the model training process. The plot is in the 'Result' folder.

3. Making Predictions:
 'app.py' file contains the code for the web application for evaluating the results in real-time using webcam and also by uploading an image.

## Tools Used: 
* Python
* Keras and Tensorflow
* Streamlit 
* OpenCV

## Approaches tried: 
* I first tried to crop faces for the dataset and convert them to feature vectors and using these features to train a model using machine learning algorithms like KNN, SVM, etc.
* Tried to crop the face in an image then try to detect the amount of redness in each face using OpenCV. If the amount of red is greater then threshold then we say that the Image has blood

## Future Enhancements: 
Collect more datasets and try to increase accuracy. Also to deploy the model on the web for real-time use cases using tools like Heroku, Netlify or AWS.
