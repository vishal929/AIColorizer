# just a file to organize our improved model thoughts
import sys

import numpy as np
import random
import atexit
import math
import signal


# PATCH IS ASSUMING THE FOLLOWING FORMAT FOR EACH PATCH (CONSISTENT WITH COLORIZE.py format):
'''
[
blackWhiteTraining[z - 1, y - 1],
blackWhiteTraining[z, y - 1],
blackWhiteTraining[z + 1, y - 1],
blackWhiteTraining[z - 1, y],
blackWhiteTraining[z, y],
blackWhiteTraining[z + 1, y],
blackWhiteTraining[z - 1, y + 1],
blackWhiteTraining[z, y + 1],
blackWhiteTraining[z + 1, y + 1]]
'''


class Model():
    
    '''
    This class is an abstract class for a Model. From here we could specify different models and loss functions
    Features are passed in as a pointer. It is the caller's responsibility to make sure featurePtr and featureDim match. If they don't then there won't be the correct number of weights.
    '''
    def __init__(self, id, featurePtr, featureDim):
        # standard featureDim is 10 --> (1,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9)
        self.features = featurePtr
        self.featureDim = featureDim
        self.redWeights=[]
        self.greenWeights=[]
        self.blueWeights=[]
        self.id = id


    #return r,g,b value model gets at patch
    def evaluateModel(self, patch):
        pass

    def getRGB(self, patch):
        pass

    #evaluate gradient of the loss (which we precalculate) at vector x
    def loss_gradient(self, patch, actualrgb):
        pass

    # by stochastic gradient descent
    # trains the red, green, and blue models
    # assumes blackWhiteTraining and colorTraining are numpy arrays
    # alpha, beta, and delta are our learning rates for our 3 separate models
    def trainModel(self,simpleTraining,actualTraining, alpha, beta, delta):
        #scaling data down
        blackWhiteTraining = np.copy((simpleTraining.astype(np.double)))/2500
        colorTraining = np.copy((actualTraining.astype(np.double)))/255
        
        #if the weights list is empty, initialize random small weights
        alpha = np.double(alpha)
        beta = np.double(beta)
        delta = np.double(delta)
        if len(self.redWeights) < 1:
            self.redWeights = np.random.rand(self.featureDim).astype(np.double)
        if len(self.greenWeights)<1:
            self.greenWeights = np.random.rand(self.featureDim).astype(np.double)
        if len(self.blueWeights)<1:
            self.blueWeights = np.random.rand(self.featureDim).astype(np.double)

        # every 1000 iterations, we will calculate the loss of our training data and stop if we are satisfied
        threshold=100000
        iterCount=0
        learningRateDecay = np.double(1)

        #initial loss
        lastLoss = self.trainingLoss(blackWhiteTraining,colorTraining)
        print("INITIAL LOSS:"+str(lastLoss))
        while(True):
            #in order to check for convergence, maybe theres a better way to do this than to make copies
            old_redWeights = np.copy(self.redWeights)
            old_greenWeights = np.copy(self.greenWeights)
            old_blueWeights = np.copy(self.blueWeights)

            #choose random patch from blackWhiteTraining (excluding border)
            #pixel chosen is at (row,col), then the patch is built around it
            row = random.randint(1, blackWhiteTraining.shape[0] - 2)
            col = random.randint(1, blackWhiteTraining.shape[1] - 2)
            x = np.array([blackWhiteTraining[row - 1][col - 1], #column1
                blackWhiteTraining[row][col - 1],
                blackWhiteTraining[row + 1][col - 1],
                blackWhiteTraining[row - 1][col],#column2
                blackWhiteTraining[row][col],
                blackWhiteTraining[row + 1][col],
                blackWhiteTraining[row - 1][col + 1],#column3
                blackWhiteTraining[row][col + 1],
                blackWhiteTraining[row + 1][col + 1]])

            #the rgb value the model predicts, corresponds to f(x)
            actual_rgb = colorTraining[row][col] #the rgb value from the data, corresponds to y

            # update w_{t+1}=w_t - alpha (GRADIENT(LOSS_i))(w_t)
            grad = self.loss_gradient(x, actual_rgb)
            redModifier = alpha * grad[0]
            greenModifier = beta * grad[1]
            blueModifier = delta * grad[2]
            self.redWeights = old_redWeights - (alpha * grad[0])
            self.greenWeights = old_greenWeights - (beta * grad[1])
            self.blueWeights = old_blueWeights - (delta * grad[2])

            # just printing weights after each adjustment for convenience
            '''
            print("NEW WEIGHTS ----------- RGB ORDER ----------------")
            print("RED WEIGHTS")
            print(self.redWeights)
            print("GREEN WEIGHTS")
            print(self.greenWeights)
            print("BLUE WEIGHTS")
            print(self.blueWeights)
            print("END OF WEIGHTS -----------------------------------")
            '''
            alpha *= learningRateDecay
            beta *= learningRateDecay
            delta *= learningRateDecay
            iterCount +=1
            if iterCount == threshold:
                # computing the loss
                # computing loss and seeing if we are close now (maybe like 1000 ish loss is acceptable? we will see)
                loss = self.trainingLoss(blackWhiteTraining,colorTraining)

                print("GOT LOSS :"+str(loss))
                print("RED WEIGHTS:")
                print(self.redWeights)
                print("GREEN WEIGHTS:")
                print(self.greenWeights)
                print("BLUE WEIGHTS:")
                print(self.blueWeights)


                if loss[0] <lastLoss[0]:
                    # writing red weights
                    self.writeWeightsToFile(0)
                    lastLoss = loss[0], lastLoss[1], lastLoss[2]

                if loss[1]<lastLoss[1]:
                    # writing green weights
                    self.writeWeightsToFile(1)
                    lastLoss = lastLoss[0], loss[1], lastLoss[2]

                if loss[2] < lastLoss[2]:
                    # writing blue weights
                    self.writeWeightsToFile(2)
                    lastLoss = lastLoss[0], lastLoss[1], loss[2]

                # resetting iterCount
                iterCount=0
            
            #check for convergence with the old weights - stop if change in weights < 0.1 (experiment with the number 0.1)
            '''
            if (np.less(np.absolute(old_redWeights - self.redWeights), np.full(self.featureDim, 0.001)) and
                np.less(np.absolute(old_greenWeights - self.greenWeights), np.full(self.featureDim, 0.001)) and
                np.less(np.absolute(old_blueWeights - self.blueWeights), np.full(self.featureDim, 0.001))):
                self.writeWeightsToFile()
                break
            '''



    #generates a color image from bwImage to compare to the original color image
    def testModel(blackWhiteTest,colorTest):
        pass

    # computes loss for the training image
    def trainingLoss(self,blackWhiteTraining,colorImageTraining):
        pass


    # TXT FILE ENCODING:
        # redWeight_1 redWeight_2 ...
        # gWeight_1 gWeight_2 ...
        # bWeight_1 bWeight_2 ...
    def loadWeightsFromFile(self):
        # idea, we split the output into lines, append each number into its respective weights vector
        try:
            file = open("redWeights{id}.txt".format(id = self.id), 'r')
            actualRedWeights=[]
            lines = file.read().splitlines()
            stringRedWeights = lines[0].split()
            for num in stringRedWeights:
                actualRedWeights.append(np.double(num))
            self.redWeights = actualRedWeights
            file.close()
        except :
            pass

        try:
            file = open("greenWeights{id}.txt".format(id = self.id), 'r')
            lines = file.read().splitlines()
            actualGreenWeights = []
            stringGreenWeights = lines[0].split()
            for num in stringGreenWeights:
                actualGreenWeights.append(np.double(num))
            self.greenWeights = actualGreenWeights
            file.close()
        except:
            pass

        try :
            file = open("blueWeights{id}.txt".format(id = self.id), 'r')
            lines = file.read().splitlines()
            actualBlueWeights = []
            stringBlueWeights = lines[0].split()
            for num in stringBlueWeights:
                actualBlueWeights.append(np.double(num))
            self.blueWeights = actualBlueWeights
            file.close()
        except:
            pass

    
    def writeWeightsToFile(self,rgb):
        # first clearing all the weights currently in the txt file
        if rgb==0:
            # write red weights
            file = open("redWeights{id}.txt".format(id = self.id), 'w')
            file.truncate(0)
            for reds in self.redWeights:
                file.write(str(reds) + " ")
            file.write("\n")
            print("WROTE RED WEIGHTS TO redWeights.txt")
        elif rgb==1:
            # write green weights
            file = open("greenWeights{id}.txt".format(id = self.id), 'w')
            file.truncate(0)
            for greens in self.greenWeights:
                file.write(str(greens) + " ")
            file.write("\n")
            print("WROTE GREEN WEIGHTS TO greenWeights.txt")
        else:
            # write blue weights
            file = open("blueWeights{id}.txt".format(id = self.id), 'w')
            file.truncate(0)
            for blues in self.blueWeights:
                file.write(str(blues) + " ")
            file.write("\n")
            print("WROTE BLUE WEIGHTS TO blueWeights.txt")

        file.close()

class SigmoidModel(Model):

    def __init__(self, id, featurePtr, featureDim):
        super().__init__(id, featurePtr, featureDim)

    def sigmoid(self,z):
        res= np.double(1.) / (np.double(1.) + np.double(np.exp(-z)))
        return np.double(res)

    def evaluateModel(self, patch):
        #return r,g,b value model gets at patch
        # sigmoid function is:  sigmoid(z) = 1/(1+e^{-z})
        phi = self.features(patch)
        red_dot = np.dot(self.redWeights, phi)
        green_dot = np.dot(self.greenWeights, phi)
        blue_dot = np.dot(self.blueWeights, phi)
        return self.sigmoid(red_dot) , self.sigmoid(green_dot), self.sigmoid(blue_dot)

    def getRGB(self, patch, scaled = False):
        if not scaled:
            patch /= 2500
        modelR, modelG, modelB = self.evaluateModel(patch)
        return 255*modelR, 255*modelG, 255*modelB

    # computes loss for the training image
    # data should be scaled down before calling this method
    def trainingLoss(self, blackWhiteTraining, colorImageTraining):
        # going through all the patches, computing, and returning the loss
        trainingWidth, trainingLength = np.shape(blackWhiteTraining)
        redLoss = 0
        greenLoss = 0
        blueLoss = 0
        for row in range(trainingWidth):
            if row==0 or row==trainingWidth-1:
                continue
            for col in range(trainingLength):
                if col == 0 or col == trainingLength - 1:
                    continue

                # this is a valid patch, we first compute the model rgb value
                patch = [blackWhiteTraining[row - 1, col - 1],
                         blackWhiteTraining[row, col - 1],
                         blackWhiteTraining[row + 1, col - 1],
                         blackWhiteTraining[row - 1, col],
                         blackWhiteTraining[row, col],
                         blackWhiteTraining[row + 1, col],
                         blackWhiteTraining[row - 1, col + 1],
                         blackWhiteTraining[row, col + 1],
                         blackWhiteTraining[row + 1, col + 1]]
                modelR, modelG, modelB = self.evaluateModel(patch)
                redLoss += (modelR - colorImageTraining[row, col, 0]) **2
                greenLoss += (modelG - colorImageTraining[row, col ,1]) **2
                blueLoss += (modelB - colorImageTraining[row,col,2])**2
        # returning loss for each model
        return (redLoss,greenLoss,blueLoss)

    #evaluate gradient of the loss at vector x for red, green, and blue
    def loss_gradient(self, patch, actualrgb):
        # 510 (255 sigma(w dot features(patch)) - actualrgb[c]) sigma(w dot features(patch))(1 - sigma(w dot features(patch)))
        actualR, actualG, actualB = actualrgb
        red_sigmoid, green_sigmoid, blue_sigmoid = self.evaluateModel(patch)
        phi = self.features(patch)

        redGradient = 2 * ( red_sigmoid - actualR) * red_sigmoid * (1-red_sigmoid) * phi
        greenGradient = 2 * ( green_sigmoid - actualG) * (green_sigmoid * (1-green_sigmoid)) * phi
        blueGradient = 2 * (blue_sigmoid - actualB) * (blue_sigmoid * (1-blue_sigmoid)) * phi

        return redGradient, greenGradient, blueGradient

