# just a file to organize our improved model thoughts
import sys

import numpy as np
import random
import atexit
import math
import signal

class Model():
    # standard featureDim is 10 --> (1,x_1,x_2,x_3,x_4,x_5,x_6,x_7,x_8,x_9)
    featureDim = 10
    redWeights=[]
    greenWeights=[]
    blueWeights=[]

    #for now this is equivalent to saying there are no features
    def features(self,patch):
        np.append(patch,1)
        return patch

    #return r,g,b value model gets at patch
    def evaluateModel(self, patch):
        pass
        
    #evaluate gradient of the loss (which we precalculate) at vector x
    def loss_gradient(self, patch, actualrgb):
        pass

    def loss(self,modelrgb, actualrgb):
        pass

    '''have to standardize what the shape of weights arrays are, I think its (featureDim,)
        also don't know if weights arrays have to numpy arrays for everything below to work'''
    #by stochastic gradient descent
    #would do red, green, blue
    #alpha stepsize
    #assumes blackWhiteTraining and colorTraining are numpy arrays
    def trainModel(self,blackWhiteTraining,colorTraining, alpha):
        #if the weights list is empty, initialize random small weights
        if len(self.redWeights) < 1:

            self.redWeights = np.random.rand(self.featureDim)*0.001
            self.greenWeights = np.random.rand(self.featureDim)*0.001
            self.blueWeights = np.random.rand(self.featureDim)*0.001
            self.redWeights = np.array([-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5,-0.5]).astype(np.double)
            self.greenWeights = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]).astype(np.double)
            self.blueWeights = np.array([-0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5, -0.5]).astype(np.double)
            print("STARTING RED: "+str(self.redWeights))
            print("STARTING GREEN: "+str(self.greenWeights))
            print("STARTING BLUE: "+str(self.blueWeights))

        # every 1000 iterations, we will calculate the loss of our training data and stop if we are satisfied
        threshold=1000
        iterCount=0

        while(True):
            #in order to check for convergence, maybe theres a better way to do this than to make copies
            old_redWeights = np.copy(self.redWeights)
            old_greenWeights = np.copy(self.greenWeights)
            old_blueWeights = np.copy(self.blueWeights)

            #choose random patch from blackWhiteTraining (excluding border)
            #pixel chosen is at (row,col), then the patch is built around it
            row = random.randint(1, blackWhiteTraining.shape[0] - 2)
            col = random.randint(1, blackWhiteTraining.shape[1] - 2)
            '''
            x = np.array([blackWhiteTraining[row-1][col-1], blackWhiteTraining[row-1][col], blackWhiteTraining[row-1][col+1],
                blackWhiteTraining[row][col-1], blackWhiteTraining[row][col], blackWhiteTraining[row][col+1],
                blackWhiteTraining[row+1][col-1], blackWhiteTraining[row+1][col], blackWhiteTraining[row+1][col+1]])
            '''
            x = np.array([blackWhiteTraining[row - 1][col - 1], #column1
                blackWhiteTraining[row][col - 1],
                blackWhiteTraining[row + 1][col - 1],
                blackWhiteTraining[row - 1][col],#column2
                blackWhiteTraining[row][col],
                blackWhiteTraining[row + 1][col],
                blackWhiteTraining[row - 1][col + 1],#column3
                blackWhiteTraining[row][col + 1],
                blackWhiteTraining[row + 1][col + 1]])

            # (Soumya: I just realized we don't need the value for the loss gradient if we plug x and y into the loss function we have)
             #the rgb value the model predicts, corresponds to f(x)
            actual_rgb = colorTraining[row][col] #the rgb value from the data, corresponds to y

            # update w_{t+1}=w_t - alpha (GRADIENT(LOSS_i))(w_t)
            grad = self.loss_gradient(x, actual_rgb)
            print("GRADIENT: "+str(grad))
            redModifier = alpha* grad[0]
            greenModifier = alpha*grad[1]
            blueModifier = alpha*grad[2]
            print("RED MODIFIER: "+str(redModifier))
            print("GREEN MODIFIER: "+str(greenModifier))
            print("BLUE MODIFIER: "+str(blueModifier))
            self.redWeights = self.redWeights - alpha * grad[0]
            self.greenWeights = self.greenWeights - alpha * grad[1]
            self.blueWeights = self.blueWeights - alpha * grad[2]

            # just printing weights after each adjustment for convenience
            print("NEW WEIGHTS ----------- RGB ORDER ----------------")
            print("RED WEIGHTS")
            print(self.redWeights)
            print("GREEN WEIGHTS")
            print(self.greenWeights)
            print("BLUE WEIGHTS")
            print(self.blueWeights)
            print("END OF WEIGHTS -----------------------------------")

            iterCount +=1
            if iterCount == threshold:
                # computing the loss
                # computing loss and seeing if we are close now (maybe like 100 ish loss is acceptable? we will see)
                loss = self.trainingLoss(blackWhiteTraining,colorTraining)
                print("GOT LOSS :"+str(loss))
                if loss[0] <= 100 and loss[1]<=100 and loss[2]<=100:
                    # this is close enough, for now
                    print("DONE TRAINING")
                    self.writeWeightsToFile()
                    break
                # resetting iterCount
                iterCount=0
            #check for keyword "stop" from user, if this happens then call writetofile
            '''looking to atexit.register() to write data when press ctrl-c, the only issue is it needs a function that does not have arguments
                if loadWeightsFromFile were changed so that it just writes to a set file, then it would work
                the issue with checking for stop is that the program would have to prompt user every x iterations or so about whether they want to stop'''
            
            #check for convergence with the old weights - stop if change in weights < 0.1 (experiment with the number 0.1)
            '''
            if (np.less(np.absolute(old_redWeights - self.redWeights), np.full(self.featureDim, 0.1)) and
                np.less(np.absolute(old_greenWeights - self.greenWeights), np.full(self.featureDim, 0.1)) and
                np.less(np.absolute(old_blueWeights - self.blueWeights), np.full(self.featureDim, 0.1))):
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
    #returns the weight vectors as a tuple (redWeights, greenWeights, blueWeights)
    def loadWeightsFromFile(self):
        # idea, we split the output into lines, append each number into its respective weights vector
        try:
            file = open("weights.txt", 'r')
        except :
            return False
        lines = file.read().splitlines()
        stringRedWeights = lines[0].split()
        #print(stringRedWeights)
        actualRedWeights=[]
        stringGreenWeights = lines[1].split()
        #print(stringGreenWeights)
        actualGreenWeights=[]
        stringBlueWeights=lines[2].split()
        #print(stringBlueWeights)
        actualBlueWeights=[]
        for i in range(len(stringRedWeights)):
            actualRedWeights.append(np.double(stringRedWeights[i]))
            actualGreenWeights.append(np.double(stringGreenWeights[i]))
            actualBlueWeights.append(np.double(stringBlueWeights[i]))
        self.redWeights=actualRedWeights
        self.greenWeights=actualGreenWeights
        self.blueWeights=actualBlueWeights
        file.close()
        return True

    
    def writeWeightsToFile(self):
        # first clearing all the weights currently in the txt file
        file = open("weights.txt", 'w')
        file.truncate(0)
        # now writing our numbers to the file with newlines separating
        for reds in self.redWeights:
            file.write(str(reds)+" ")
        file.write("\n")
        for greens in self.greenWeights:
            file.write(str(greens)+" ")
        file.write("\n")
        for blues in self.blueWeights:
            file.write(str(blues)+" ")
        file.write("\n")
        file.close()
        print("WROTE WEIGHTS TO WEIGHTS.txt")


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
class SigmoidModel(Model):

    def sigmoid(self,z):
        return 1 / (1 + np.exp(-z))

    def evaluateModel(self, patch):
        #return r,g,b value model gets at patch
        # sigmoid function is:  sigmoid(z) = 1/(1+e^{-z})
        phi = self.features(patch) #should probably return a numpy array
        red_dot = np.dot(self.redWeights, phi)
        green_dot = np.dot(self.greenWeights, phi)
        blue_dot = np.dot(self.blueWeights, phi)
        return 255 * self.sigmoid(red_dot) , 255 * self.sigmoid(green_dot), 255 * self.sigmoid(blue_dot)

    # computes loss for the training image
    def trainingLoss(self, blackWhiteTraining, colorImageTraining):
        # going through all the patches, computing, and returning the loss
        trainingWidth, trainingLength = np.shape(blackWhiteTraining)
        redLoss = 0
        greenLoss = 0
        blueLoss = 0
        for row in range(trainingWidth):
            if row == 0 or row == trainingWidth - 1:
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
                phi = self.features(patch)
                # getting the pixel value
                modelR= 255 * self.sigmoid(np.dot(self.redWeights,phi))
                modelG = 255 * self.sigmoid(np.dot(self.greenWeights, phi))
                modelB = 255 * self.sigmoid(np.dot(self.blueWeights, phi))
                redLoss += (modelR - colorImageTraining[row, col, 0]) **2
                greenLoss += (modelG - colorImageTraining[row, col ,1]) **2
                blueLoss += (modelB - colorImageTraining[row,col,2])**2
        # returning loss for each model
        return (redLoss,greenLoss,blueLoss)

    #evaluate gradient of the loss at vector x for red, green, and blue
        # phi is the feature included data vector - Soumya: we already have the self.features(patch)
    def loss_gradient(self, patch, actualrgb):
        # 510 (255 sigma(w dot features(patch)) - actualrgb[c]) sigma(w dot features(patch))(1 - sigma(w dot features(patch)))
        #modelR, modelG, modelB = modelrgb #Soumya: each is R(x), G(x), and B(x) resp, but loss is in terms of x and y
        actualR, actualG, actualB = actualrgb

        phi = self.features(patch) #should probably return a numpy array
        #each sigmoid term is sigma(w dot features(patch))
        red_sigmoid = self.sigmoid(np.dot(self.redWeights, phi))
        green_sigmoid = self.sigmoid(np.dot(self.redWeights, phi))
        blue_sigmoid = self.sigmoid(np.dot(self.redWeights, phi))

        print("MODEL RED: "+str(255*red_sigmoid))
        print("MODEL GREEN: "+str(255*green_sigmoid))
        print("MODEL BLUE: "+str(255*blue_sigmoid))

        redGradient = 510 * (255 * red_sigmoid - actualR) * red_sigmoid * (1-red_sigmoid) * phi
        greenGradient = 510 * (255 * green_sigmoid - actualG) * green_sigmoid * (1-green_sigmoid) * phi
        blueGradient = 510 * (255 * blue_sigmoid - actualB) * blue_sigmoid * (1-blue_sigmoid) * phi

        return redGradient, greenGradient, blueGradient

    # we can hardcode what features we want for now
    def features(self,patch):
        # standard features
            # appending 1 for the w_0 weight
        np.append(patch,1)
        features=[1]
        for value in patch:
            features.append(value)

        return np.array(features)
        # x^2 features
        '''
        phi =[1]
        for greyValue in patch:
            phi.append(greyValue)
            phi.append(greValue**2)
        return np.array(phi)
        '''
        #my idea of the middle component mattering the most, then 1 level out mattering less, last level mattering the least)

        pass

    # just squared loss between returned rgb value from model and the actual rgb value
    def loss(self,modelrgb, actualrgb):
        modelR, modelG, modelB = modelrgb
        actualR, actualG, actualB = actualrgb
        return (modelR - actualR)**2, (modelG - actualG)**2, (modelB - actualB)**2


    


# testing getting weights and writing weights to a file
'''
testModel = Model()
testModel.redWeights = [3.65,4.6,0.003,9.87654]
testModel.greenWeights = [4.34, 0.0002, 5.453, 7.0003]
testModel.blueWeights = [7.984, 0.0543, 0.0000003, 0.3]

#testing atexit functionality
atexit.register(Model.writeWeightsToFile, testModel)
while True:
    pass

testModel.writeWeightsToFile("weights.txt")

otherTestModel = Model()

# loading weights into otherTestModel and seeing if they are preserved
otherTestModel.loadWeightsFromFile("weights.txt")
print(otherTestModel.redWeights)
print(otherTestModel.greenWeights)
print(otherTestModel.blueWeights)
'''

'''
input X --> y



class Model
    weights for red
    weights for green
    weights for blue

    evaluate()
    getDerivative()


'''

'''
features(patch1) dot weights

'''

