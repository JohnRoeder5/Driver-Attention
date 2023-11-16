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

TRAINED MODEL DATA + DIAGNOSTICS:
Shown below is an image of the model training with 10 epochs. 
![image](https://github.com/Jborch1/FinalCapstoneDS/assets/122740699/0f7f8fce-ecf8-404b-81d0-ed0a32afe0e0)  



This is our loss over epochs graph for the trained data and some of the results we have collected for our own model.
![image](https://github.com/Jborch1/FinalCapstoneDS/assets/122740699/931fe390-cb62-46c5-8d33-76b45e3452c1)


With this model you are eligible to tune the stats in the googArchitecture file to fit what you are trying to do for your own project.

For the case of our mini project we are doing predicitvie analysis in images so anyone wanting to do the same project in this field could use this model and tune the stats if need be.
