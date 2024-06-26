# Colab Source Code:
https://colab.research.google.com/drive/1dT4vLbaJQLjCfniH04ra-mnM8G-1ESYW#scrollTo=CfTrN5OHKgc7





# Attention Detection CNN Model
Repository for attention detection model aimed at predicting the state of a driver's attentiveness as either attentive or inattentive. Our Implementation is heavily inspired by the GoogeLeNet architecture for CNN's. link to that paper: https://arxiv.org/pdf/1409.4842.pdf

Dataset found on Kaggle. Link to page with dataset:https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data

Required: 
Python 3.11.6, 
TensorFlow 2.14

1. Install required Libraires:
- Shown below
![image](https://github.com/Jborch1/FinalCapstoneDS/assets/122740699/0d87c245-2315-43c6-abdd-1546028893ac)


2. resize your images that you want to be tested to 224X224 if needed (program should take care of it if you read in a "dataset of images". In order to do this set data_directory= to the file path for your images).
3. run the model in terminal.

# TRAINED MODEL DATA and Results
Shown below is the trained model accuracy and loss:
![image](https://github.com/Jborch1/FinalCapstoneDS/assets/122740699/f5abaf6c-aedc-41b5-8b17-e35534f332b1)




This is our loss over epochs graph for the trained data and some of the results we have collected for our own model.
![image](https://github.com/Jborch1/FinalCapstoneDS/assets/122740699/bf6fc486-7d27-40c7-ba43-81942171da52)

Here are some of the results generated by the model, this viewer was attentive:

![image](https://github.com/Jborch1/FinalCapstoneDS/assets/122740699/c23d9289-037c-4133-927e-fecff3f53fbe)

Here is someone inattentive:
![image](https://github.com/Jborch1/FinalCapstoneDS/assets/122740699/a7c917c0-9552-43a5-9964-1b83b4b9698e)




