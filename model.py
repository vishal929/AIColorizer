# just a file to organize our improved model thoughts
import numpy as np
import random
import atexit
import math

class Model():
    featureDim = 9
    redWeights=[]
    greenWeights=[]
    blueWeights=[]

    #for now this is equivalent to saying there are no features
    def features(self,patch):
        return patch

    '''maybe this method isn't needed and can just be part of training
    def initializeWeights(self):
        pass
    '''

    def evaluateModel(self, patch):
        #return r,g,b value model gets at patch
        # sigmoid function is:  sigmoid(z) = 1/(1+e^{-z})
        pass

    ''' maybe simpler to not have these
    #helper to evaluateModel and can be called separately
    def red(self,patch):
        pass

    def green(self,patch):
        pass

    def blue(self,patch):
        pass
    '''
        
    #evaluate gradient of the loss (which we precalculate) at vector x
    def loss_gradient(self,modelrgb, actualrgb):
        # (-y_i +sigmoid(w DOT x_i)) (jth component of x_i) #actually since this would return a vector, I think its just x_i
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
            self.redWeights = np.random.rand(featureDim) * 0.001
            self.greenWeights = np.random.rand(featureDim) * 0.001
            self.blueWeights = np.random.rand(featureDim) * 0.001

        while(True):
            #in order to check for convergence, maybe theres a better way to do this than to make copies
            old_redWeights = np.copy(self.redWeights)
            old_greenWeights = np.copy(self.greenWeights)
            old_blueWeights = np.copy(self.blueWeights)

            #choose random patch from blackWhiteTraining (excluding border)
            #pixel chosen is at (row,col), then the patch is built around it
            row = random.randint(1, blackWhiteTraining.shape[0] - 2)
            col = random.randint(1, blackWhiteTraining.shape[1] - 2)
            x = np.array([blackWhiteTraining[row-1][col-1], blackWhiteTraining[row-1][col], blackWhiteTraining[row-1][col+1],
                blackWhiteTraining[row][col-1], blackWhiteTraining[row][col], blackWhiteTraining[row][col+1],
                blackWhiteTraining[row+1][col-1], blackWhiteTraining[row+1][col], blackWhiteTraining[row+1][col+1]])

            model_rgb = self.evaluateModel(x) #the rgb value the model predicts, corresponds to f(x)
            actual_rgb = colorTraining[row][col] #the rgb value from the data, corresponds to y

            # update w_{t+1}=w_t - alpha (GRADIENT(LOSS_i))(w_t)
            grad = self.loss_gradient(model_rgb, actual_rgb)
            self.redWeights = self.redWeights - alpha * grad[0]
            self.greenWeights = self.greenWeights - alpha * grad[1]
            self.blueWeights = self.blueWeights - alpha * grad[2]

            #check for keyword "stop" from user, if this happens then call writetofile
            '''looking to atexit.register() to write data when press ctrl-c, the only issue is it needs a function that does not have arguments
                if loadWeightsFromFile were changed so that it just writes to a set file, then it would work
                the issue with checking for stop is that the program would have to prompt user every x iterations or so about whether they want to stop'''
            
            #check for convergence with the old weights (experiment with the number 0.1)
            if (np.less(np.absolute(old_redWeights - self.redWeights), np.full(featureDim, 0.1)) and
                np.less(np.absolute(old_greenWeights - self.greenWeights), np.full(featureDim, 0.1)) and
                np.less(np.absolute(old_blueWeights - self.blueWeights), np.full(featureDim, 0.1))):
                self.writeWeightsToFile()
                break



    #generates a color image from blackWhiteTest and returns the observed loss when compared to colorTest
    def testModel(blackWhiteTest,colorTest):
        pass

    # TXT FILE ENCODING:
        # redWeight_1 redWeight_2 ...
        # gWeight_1 gWeight_2 ...
        # bWeight_1 bWeight_2 ...
    #returns the weight vectors as a tuple (redWeights, greenWeights, blueWeights)
    def loadWeightsFromFile(self,fileName):
        # idea, we split the output into lines, append each number into its respective weights vector
        file = open(fileName, 'r')
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
            actualRedWeights.append(float(stringRedWeights[i]))
            actualGreenWeights.append(float(stringGreenWeights[i]))
            actualBlueWeights.append(float(stringBlueWeights[i]))
        self.redWeights=actualRedWeights
        self.greenWeights=actualGreenWeights
        self.blueWeights=actualBlueWeights
        file.close()

    
    def writeWeightsToFile(self,filename):
        # first clearing all the weights currently in the txt file
        file = open(filename, 'w')
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

class SigmoidModel(Model):

    def sigmoid(z):
        return 1 / (1 + math.exp(-z))

    def evaluateModel(self, patch):
        #return r,g,b value model gets at patch
        # sigmoid function is:  sigmoid(z) = 1/(1+e^{-z})
        phi = features(patch) #should probably return a numpy array
        red_dot = np.dot(self.redWeights, phi)
        green_dot = np.dot(self.greenWeights, phi)
        blue_dot = np.dot(self.blueWeights, phi)
        return 255 * self.sigmoid(red_dot) , 255 * self.sigmoid(green_dot), 255 * self.sigmoid(blue_dot)

    #evaluate gradient of the loss (which we precalculate) at vector x
    def loss_gradient(self,modelrgb, actualrgb):
        # (-y_i +sigmoid(w DOT x_i)) (jth component of x_i) #actually since this would return a vector, I think its just x_i
        pass

    def features(self,patch):
        pass

    def loss(self,modelrgb, actualrgb):
        pass
    


# testing getting weights and writing weights to a file
testModel = Model()
testModel.redWeights = [3.65,4.6,0.003,9.87654]
testModel.greenWeights = [4.34, 0.0002, 5.453, 7.0003]
testModel.blueWeights = [7.984, 0.0543, 0.0000003, 0.3]

testModel.writeWeightsToFile("weights.txt")

otherTestModel = Model()

# loading weights into otherTestModel and seeing if they are preserved
otherTestModel.loadWeightsFromFile("weights.txt")
print(otherTestModel.redWeights)
print(otherTestModel.greenWeights)
print(otherTestModel.blueWeights)

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

