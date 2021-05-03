# just hosts logic for starting up training of our model, computing loss of an output image compared to the actual
import atexit

import numpy as np

import model
import colorize

# setup for training/evaluation

ourModel = model.SigmoidModel()
# cropping color image to the same 512x512 crop as in colorize
colorImage = (colorize.imageToArray("colorImage.jfif"))[500:500+512,900:900+512,:]
cWidth, cLength, cDepth = np.shape(colorImage)
bwImage = colorize.bwImage(colorImage)
bwWidth, bwLength = np.shape(bwImage)

# asking user what they wish to do
yesNo = int(input("Please enter 0 if you want to train the model and 1 if you want to test the output and compute the loss!"))
if yesNo ==0:
    # train
    atexit.register(model.Model.writeWeightsToFile, ourModel)
    ourModel.loadWeightsFromFile()
    # starting alpha with 0.001
    ourModel.trainModel(bwImage[:,:int(bwLength/2)],colorImage[:,:int(cLength/2),:],0.1)
else:
    # output image and compute loss
    # coloring the entire image and returning the loss
    pass