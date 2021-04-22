# I tried to keep same format as the picture you sent

# WE ARE USING SCIKIT ONLY FOR IMAGE REPRESENTATION HERE! NO BUILT IN METHODS ARE USED IN OUR IMPLEMENTATION
from random import randint

from skimage import io
import numpy
# takes a colored image file as input, writes to this directory the black and white output
def bwImage(image):
    #pixels is a np array with rgb "layers" i.e mxn pixels in 3 dimensions for rgb
    pixels = io.imread(image)
    #greyscale function is replacing (r,g,b) with 0.21 r + 0.72 g + 0.07b
    r,g,b = pixels[:,:,0], pixels[:,:,1], pixels[:,:,2]
    greyValues = (numpy.rint(numpy.add((0.21 * r),(0.72 * g),(0.07*b)))).astype(numpy.uint8)
    # writing greyscale image to project directory
    io.imsave("GreyScaleImage.jfif",greyValues)

# runs k means on the image, (num means is the # of means to use)
    # for the basic writeup, numMeans will be 5
    # for the bonus, we will have to find a good numMeans
def knn(colorImage, blackWhiteImage,numMeans):
    # first running k-means on the left half of color picture to get representative colors
    colorImagePixels = io.imread(colorImage)
    shapes = numpy.shape(colorImagePixels)
    # cropping matrix to only the left half
    colorImagePixels = colorImagePixels[:shapes[0],:shapes[1]/2,:]
        # picking numMeans number of initial means from the left hand side of colorImage
    # picking length,width of pixel
        # pixel is a tuple of rgb
    meansChosen = 0
    means=numpy.array([])
    while (meansChosen!=numMeans):
        randRow = randint(0, shapes[0] - 1)
        randCol = randint(0, int(shapes[1] / 2) - 1)
        meanChosen = colorImagePixels[randRow, randCol, 0], \
                     colorImagePixels[randRow, randCol, 1], \
                     colorImagePixels[randRow, randCol, 2]
        if (meanChosen in means):
            # then we have a repeat value, which is bad
            continue
        # empty list is for later mapping of pixels
        numpy.append(means,(meanChosen,numpy.array([])))
        meansChosen +=1
    # doing initial mapping of means
    for i in range(numpy.shape(colorImagePixels)[0]):
        for j in range(numpy.shape(colorImagePixels)[1]):
            # getting distance from each mean and then assigning a mapping
            # closestMean is index of closest mean
            closestMean = None
            closestMeanCost = None
            for z in range(len(means)):
                # by "closest" we take (r_1-r_2)^2 + (g_1-g_2)^2 + (b_1-b_2)^2 and minimize this
                cost = (colorImagePixels[i, j, 0] - means[z][0]) ^ 2 + \
                       (colorImagePixels[i, j, 1] - means[z][1]) ^ 2 + \
                       (colorImagePixels[i, j, 2] - means[z][2]) ^ 2
                if ((closestMean is None) or (cost < closestMeanCost)):
                    closestMean = z
                    closestMeanCost = cost
            # now we know the closest mean index
            numpy.append(means[closestMean][1],
                         (colorImagePixels[i, j, 0], colorImagePixels[i, j, 1], colorImagePixels[i, j, 2]))

    # running actual k means algorithm now
        #converting to numpy array
    oldMeans = means
    while True:
        means = numpy.array([])
        # firstly getting new means
        for i in range(len(oldMeans)):
            numpy.append(means,(numpy.average(oldMeans[i][1]),numpy.array([])))
        # adding new neighbors to the respective lists of associated pixels
        for i in range(numpy.shape(colorImagePixels)[0]):
            for j in range(numpy.shape(colorImagePixels)[1]):
                # getting distance from each mean and then assigning a mapping
                    #closestMean is index of closest mean
                closestMean= None
                closestMeanCost = None
                for z in range(len(means)):
                    # by "closest" we take (r_1-r_2)^2 + (g_1-g_2)^2 + (b_1-b_2)^2 and minimize this
                    cost = (colorImagePixels[i,j,0]-means[z][0])^2 + \
                           (colorImagePixels[i,j,1]-means[z][1])^2 +\
                           (colorImagePixels[i,j,2]-means[z][2])^2
                    if ((closestMean is None) or (cost<closestMeanCost) ):
                        closestMean = z
                        closestMeanCost = cost
                # now we know the closest mean index
                numpy.append(means[closestMean][1],
                             (colorImagePixels[i, j, 0], colorImagePixels[i, j, 1], colorImagePixels[i, j, 2]))
        # checking if any changes are made to assignments between means and oldmeans
        changes = False
        for i in range(len(means)):
            if set(means[i][1]) != set(oldMeans[i][1]):
                changes=True
                break;

        if not changes:
            # then we hit convergence
            for i in range(len(means)):
                # converting to set, so we can easily see the pixels to recolor
                means[i][1]=set(means[i][1])
            break
        else:
            oldMeans=means
    # now we have our k means representative colors, we can start to "recolor" our left half of the colored image
    for i in range(numpy.shape(colorImagePixels)[0]):
        for j in range(numpy.shape(colorImagePixels)[1]):
            rgb = colorImagePixels[i,j,0],colorImagePixels[i,j,1],colorImagePixels[i,j,2]
            for i in range(len(means)):
                if rgb in means[i][1]:
                    # then this is the corresponding representative rgb value
                    colorImagePixels[i,j,0]=means[i][0][0]
                    colorImagePixels[i,j,1]=means[i][0][1]
                    colorImagePixels[i,j,2] = means[i][0][2]
                    break
    # now the left half of the color image is "recolored"

    # getting greyscale values of the black and white image
    blackWhiteValues = io.imread(blackWhiteImage)
    # getting left hand side of image and right hand side of image
    blackWhiteTraining,blackWhiteTest = numpy.hsplit(blackWhiteValues,numpy.shape(colorImagePixels)[1])
    # need to go through all 3x3 patches on left hand side of image and associate them with a value
        # and representative color
    trainingShape = numpy.shape(blackWhiteTraining)
    testShape = numpy.shape(blackWhiteTest)
    # training data holds data in the form of ((v_1,v_2,...,v_9) , (r,g,b))
        # where the first element are the grey values of every square in the patch
        # the second element is the rgb color associated with the middle pixel from k-means
    trainingData = numpy.array([])
    for i in range(trainingShape[0]):
        # need to see if surrounding squares are valid
        # if not, then this cannot be a middle square for a 3x3 patch
        if i + 1 >= trainingShape[0]:
            # invalid patch
            continue
        if i - 1 < 0:
            # invalid patch
            continue
        for j in range(trainingShape[1]):
            if j+1>= trainingShape[1]:
                # invalid patch
                continue
            if j-1 <0 :
                # invalid patch
                continue
            # if we reached here, we have a valid patch









# trains a model
def trainModel(image,model):
    pass

# outputs a color version of a black and white image based on a model
def improved(image,model):
    pass


#test
bwImage("colorImage.jfif")