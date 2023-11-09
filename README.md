# FinalCapstoneDS
This is a repository designed to run a model that shows if someone is attentive or not using labels and CNN's


Head over to this link and download the zip file: https://github.com/AleDel/deepdreamer-touchdesigner/tree/master/models

From here you want to add in the models folder into this zip file directory

So step one add: models/bvlc_googlenet.caffemodel and models/deploy.prototxt to this zip files working directory.

Step two: choose what image directory you want to train the model with: so make sure you have a folder with two test sets of data.....pictures--->c0,c1

Step three data is trained (run imagesandModelProcess): now if you want to develop an output based on this you can do so, you can also play around with the settings of the model.

If you want to single handedly see what is most likely attribute in an image... go to googlelet.py select the image you want to run and it should return what is most likely label.

So far we are working on getting the exact output once the model is trained but we have it set up very nice now to train the model


