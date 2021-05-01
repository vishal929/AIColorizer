# just a file to organize our improved model thoughts


class Model():
    redWeights=[]
    greenWeights=[]
    blueWeights=[]

    def initializeWeights(self):
        pass

    def evaluateModel(self, patch):
        #return r,g,b value model gets at patch
        # sigmoid function is:  sigmoid(z) = 1/(1+e^{-z})
        pass

    #helper to evaluateModel and can be called separately
    def red(self,patch):
        pass

    def green(self,patch):
        pass

    def blue(self,patch):
        pass
        
    #evaluate gradient of the loss (which we precalculate) at vector x
    def gradient(patch, actualrgb):
        # (-y_i +sigmoid(w DOT x_i)) (jth component of x_i)
        pass

    def features(patch):
        pass

    def loss(modelrgb, actualrgb):
        pass

    #by stochastic gradient descent
    #would do red, green, blue
    #alpha stepsize
    def trainModel(blackWhiteTraining,colorTraining, alpha):
        #choose random patch from blackWhiteTraining
        #get model (r,g,b)
        #get the actual (r,g,b) from colorTraining

        #calculate loss
        # update w_{t+1}=w_t - alpha (GRADIENT(LOSS_i))(w_t)

        #update redModel weights
        #update greenModel weights
        #update blueModel weights
        pass

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

class sigmoidModel(Model):
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

