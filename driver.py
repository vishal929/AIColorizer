# just hosts logic for starting up training of our model, computing loss of an output image compared to the actual
import atexit

import numpy as np

import model
import colorize

# setup for training/evaluation

#can specify different features for different models, driver.py's responsibility to make sure dim matches
def featurePtr(patch):
    patch = np.append(patch,0.1)
    return patch

ourModel = model.SigmoidModel(1, featurePtr, 10) 
# cropping color image to the same 512x512 crop as in colorize
colorImage = (colorize.imageToArray("colorImage.jfif"))[500:500+512,900:900+512,:]
cWidth, cLength, cDepth = np.shape(colorImage)
bwImage = colorize.bwImage(colorImage)
bwWidth, bwLength = np.shape(bwImage)

# asking user what they wish to do
yesNo = int(input("Please enter 0 if you want to train the model and 1 if you want to test the output and compute the loss!"))
if yesNo ==0:
    # train
    #atexit.register(model.Model.writeWeightsToFile, ourModel)
    ourModel.loadWeightsFromFile()
    # starting alpha with 0.001
    ourModel.trainModel(bwImage[:,:int(bwLength/2)],colorImage[:,:int(cLength/2),:],0.01, 0.000001, 0.000001)
else:
    # output image and compute loss
    ourModel.loadWeightsFromFile()
    result = np.zeros((bwWidth,bwLength,3),dtype="uint8")
    for row in range(bwWidth):
        if row==0 or row==bwWidth-1:
            continue
        for col in range(bwLength):
            if col==0 or col==bwLength-1:
                continue
            #coloring the image from a patch
            patch = np.array([bwImage[row - 1][col - 1], #column1
                bwImage[row][col - 1],
                bwImage[row + 1][col - 1],
                bwImage[row - 1][col],#column2
                bwImage[row][col],
                bwImage[row + 1][col],
                bwImage[row - 1][col + 1],#column3
                bwImage[row][col + 1],
                bwImage[row + 1][col + 1]])
            red, green, blue=ourModel.getRGB(patch)
            result[row, col, 0] = red
            result[row, col, 1] = green
            result[row, col, 2] = blue
    # outputting color image
    colorize.arrayToImage(result,"ImprovedResult.jfif")