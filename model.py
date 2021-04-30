# just a file to organize our improved model thoughts
class sigmoidModel(Model):
    pass

class Model():
    redWeights=[]
    greenWeights=[]
    blueWeights=[]

    def initializeWeights():
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
    def loadWeightsFromFile(fileName):
        pass
    
    def writeWeightsToFile(filename):
        pass

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