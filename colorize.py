# I tried to keep same format as the picture you sent

# WE ARE USING SCIKIT ONLY FOR IMAGE REPRESENTATION HERE! NO BUILT IN METHODS ARE USED IN OUR IMPLEMENTATION
import time
import threading
from collections import deque
from random import randint, sample

from skimage import io
import numpy

# converts numpy array to image with specified name
def arrayToImage(array,imageName):
    io.imsave(imageName,array)

# converts image to numpyArray for manipulation
def imageToArray(imageName):
    return io.imread(imageName)

# image is a numpy array of 3 dimensions representing a color image
    # returns the numpy array representing a black and white image
def bwImage(colorImage):
    #colorImage is a np array with rgb "layers" i.e mxn pixels in 3 dimensions for rgb
    #greyscale function is replacing (r,g,b) with 0.21 r + 0.72 g + 0.07b
    r,g,b = colorImage[:,:,0], colorImage[:,:,1], colorImage[:,:,2]
    greyValues = (numpy.rint(numpy.add((0.21 * r),(0.72 * g),(0.07*b)))).astype(numpy.uint8)
    # returning greyscale numpy array
    return greyValues

# general function that takes a numpy data array representing the entire color image
    # runs numMeans-clustering on the image
    # then recolors the image that was input as an argument
    # returns the numpy array representing the recolored left half of the image
def kmeans(colorImage,numMeans):
    ogWidth,ogLength,ogDim= numpy.shape(colorImage)
    print("width, length, and dim of colored image: " + str(ogWidth) + ", " + str(ogLength) + ", " + str(ogDim))
    colorLeftHalf = colorImage[:,:int(ogLength/2),:]
    width,length,dim = numpy.shape(colorLeftHalf)
    print("width, length, and dim of cropped left half of colored image: "+str(width)+", "+str(length)+", "+str(dim))
    meansChosen = 0
    means=[]
    #means = numpy.array([])
    while (meansChosen != numMeans):
        randRow = randint(0, width-1)
        randCol = randint(0, length-1)
        meanChosen = (colorLeftHalf[randRow, randCol, 0], \
                     colorLeftHalf[randRow, randCol, 1], \
                     colorLeftHalf[randRow, randCol, 2])
        if ((meanChosen,[]) in means):
            # then we have a repeat value, which is bad
            continue
        # empty list is for later mapping of pixels
        means.append((meanChosen,{}))
        #means.append((meanChosen,[]))
        #numpy.append(means, (meanChosen, numpy.array([])))
        meansChosen += 1
    print("CHOSE INITIAL "+str(meansChosen)+ " MEANS!")
    # doing initial mapping of means
    for i in range(width):
        for j in range(length):
            # getting distance from each mean and then assigning a mapping
            # closestMean is index of closest mean
            closestMean = None
            closestMeanCost = None
            for z in range(len(means)):
                # by "closest" we take (r_1-r_2)^2 + (g_1-g_2)^2 + (b_1-b_2)^2 and minimize this
                #print(colorLeftHalf[i,j])
                #print(means[z][0])
                cost = ((int(colorLeftHalf[i, j, 0]) - int(means[z][0][0])) ** 2) + \
                       ((int(colorLeftHalf[i, j, 1]) - int(means[z][0][1])) ** 2) + \
                       ((int(colorLeftHalf[i, j, 2]) - int(means[z][0][2])) ** 2)
                if ((closestMean is None) or (cost < closestMeanCost)):
                    closestMean = z
                    closestMeanCost = cost
            # now we know the closest mean index
            colorToAdd = (colorLeftHalf[i, j, 0], colorLeftHalf[i, j, 1], colorLeftHalf[i, j, 2])
            if (colorToAdd in means[closestMean][1]):
                means[closestMean][1][colorToAdd] +=1
            else:
                means[closestMean][1][colorToAdd] =1
            #means[closestMean][1].append((colorLeftHalf[i, j, 0], colorLeftHalf[i, j, 1], colorLeftHalf[i, j, 2]))

            #numpy.append(means[closestMean][1],(colorLeftHalf[i, j, 0], colorLeftHalf[i, j, 1], colorLeftHalf[i, j, 2]))
    print("FINISHED INITIAL MAPPING")

    # running actual k means algorithm now
    # converting to numpy array
    oldMeans = means
    while True:
        #means = numpy.array([])
        means=[]
        # firstly getting new means
        for i in range(len(oldMeans)):
            R=0
            G=0
            B=0
            totalValues=0
            for value in oldMeans[i][1]:
                numOccurences = oldMeans[i][1][value]
                totalValues += numOccurences
                R += value[0]*numOccurences
                G += value[1]*numOccurences
                B += value[2]*numOccurences
            means.append(((int(R/totalValues),int(G/totalValues),int(B/totalValues)), {}))
            #means.append((sum(i) for i in zip(*oldMeans[i][1]))/len(oldMeans[i][1]) , [])
            #means.append( (sum(oldMeans[i][1])/len(oldMeans[i][1]), []))
            #numpy.append(means, (numpy.average(oldMeans[i][1]), numpy.array([])))
        # adding new neighbors to the respective lists of associated pixels
        for i in range(numpy.shape(colorLeftHalf)[0]):
            for j in range(numpy.shape(colorLeftHalf)[1]):
                # getting distance from each mean and then assigning a mapping
                # closestMean is index of closest mean
                closestMean = None
                closestMeanCost = None
                for z in range(len(means)):
                    # by "closest" we take (r_1-r_2)^2 + (g_1-g_2)^2 + (b_1-b_2)^2 and minimize this
                    cost = (int(colorLeftHalf[i, j, 0]) - int(means[z][0][0])) ** 2 + \
                           (int(colorLeftHalf[i, j, 1]) - int(means[z][0][1])) ** 2 + \
                           (int(colorLeftHalf[i, j, 2]) - int(means[z][0][2])) ** 2
                    if ((closestMean is None) or (cost < closestMeanCost)):
                        closestMean = z
                        closestMeanCost = cost
                # now we know the closest mean index
                colorToAdd = (colorLeftHalf[i, j, 0], colorLeftHalf[i, j, 1], colorLeftHalf[i, j, 2])
                if (colorToAdd in means[closestMean][1]):
                    means[closestMean][1][colorToAdd] += 1
                else:
                    means[closestMean][1][colorToAdd] = 1
                #numpy.append(means[closestMean][1],
                             #(colorLeftHalf[i, j, 0], colorLeftHalf[i, j, 1], colorLeftHalf[i, j, 2]))
                #means[closestMean][1].append((colorLeftHalf[i, j, 0], colorLeftHalf[i, j, 1], colorLeftHalf[i, j, 2]))
        # checking if any changes are made to assignments between means and oldmeans
        changes = False
        for i in range(len(means)):
            for key in means[i][1]:
                if key not in oldMeans[i][1] or means[i][1][key] != oldMeans[i][1][key]:
                    changes = True
                    break
            if changes:
                break
        if not changes:
            # then we hit convergence
            break
        else:
            oldMeans = means
    print("REACHED CONVERGENCE! RECOLORING IMAGE NOW")
    '''
    Recolor left half of image
    '''
    # now we have our k means representative colors, we can start to "recolor" our left half of the colored image
    for i in range(width):
        for j in range(length):
            rgb = colorLeftHalf[i, j, 0], colorLeftHalf[i, j, 1], colorLeftHalf[i, j, 2]
            for i in range(len(means)):
                if rgb in means[i][1]:
                    # then this is the corresponding representative rgb value
                    colorLeftHalf[i, j, 0] = means[i][0][0]
                    colorLeftHalf[i, j, 1] = means[i][0][1]
                    colorLeftHalf[i, j, 2] = means[i][0][2]
                    break
    # now the left half of the color image is "recolored"
    # returning the recolored left half of the image
    print("FINISHED RECOLORING IMAGE")
    return colorLeftHalf

'''in the middle of making a separate method out of picking the closest patches, 
one thing needed is the right image parameters'''
#returns the six closest patches to patch, a 9d numpy array
def sixClosest(patch,....), :
    '''
    sixClosest = []
    #there might be a numpy method(s) for doing what is in the distance sum
        #for instance, get a numpy array from subtracting 2 patches (also numpy arrays), then dot that numpy array with itself
    '''


    # if we reached here, we have a valid 3x3 patch
    # now we need to go to the training data and find the 6 closest 3x3 patches
        # form of this is [(distanceValue, (rgb)) , ...]
    sixClosest = []
    for z in range(trainingWidth):
        for y in range(trainingLength):
            # need to see if surrounding squares are valid
            # if not, then this cannot be a middle square for a 3x3 patch
            if z + 1 >= trainingWidth or z - 1 < 0 or y + 1 >= trainingLength or y-1 < 0:
                # invalid patch
                continue

            # if we reached here, we have a valid patch
            # calculating the distance value between patches
            distance = (int(blackWhiteTraining[z+1,y+1]) - int(blackWhiteTest[i+1,j+1]))**2 \
                + (int(blackWhiteTraining[z+1,y])-int(blackWhiteTest[i+1,j])) ** 2 \
                + (int(blackWhiteTraining[z+1,y-1])-int(blackWhiteTest[i+1,j-1])) ** 2 \
                + (int(blackWhiteTraining[z,y+1])-int(blackWhiteTest[i,j+1])) ** 2 \
                + (int(blackWhiteTraining[z,y])-int(blackWhiteTest[i,j])) ** 2 \
                + (int(blackWhiteTraining[z,y-1])-int(blackWhiteTest[i,j-1])) ** 2 \
                + (int(blackWhiteTraining[z-1,y+1])-int(blackWhiteTest[i-1,j+1])) ** 2 \
                + (int(blackWhiteTraining[z-1,y])-int(blackWhiteTest[i-1,j])) ** 2 \
                + (int(blackWhiteTraining[z-1,y-1])-int(blackWhiteTest[i-1,j-1])) ** 2
            rgb = (colorImage[z, y, 0],colorImage[z, y, 1],colorImage[z, y, 2])

            # seeing if we can add this data to the list
            if len(sixClosest)<6:
                # then we can just add it
                #numpy.append(sixClosest,(distance,rgb))
                sixClosest.append((distance, rgb))
            else:
                # then we have to replace this with the greatest value if it is larger
                largest= None
                for closest in sixClosest:
                    if largest is None or closest[0] > largest[0]:
                        largest = closest
                #largest = numpy.argmax(sixClosest)
                if largest[0] > distance:
                    # then we can replace it
                    sixClosest.remove(largest)
                    sixClosest.append((distance, rgb))

# runs knn to recolor right hand side of black and white image
    # returns the combined image as a numpy 3d array
# colorImage is a numpy3d array of pixels representing the recolored left half of the color image
# blackWhite image is the entire black and white image as a 2d numpy array of grey values
def knn(colorImage, blackWhiteImage):
    print("STARTING KNN")
    width,length = numpy.shape(blackWhiteImage)
    print("width, length of bw image: "+str(width)+", "+str(length))
    # getting left hand side of image and right hand side of image
    #blackWhiteTraining, blackWhiteTest = numpy.hsplit(blackWhiteImage, int(length/2))
    blackWhiteTraining = blackWhiteImage[:,:int(length/2)]
    trainingWidth, trainingLength = numpy.shape(blackWhiteTraining)
    print("width, length of bw training: " + str(trainingWidth) + ", " + str(trainingLength))
    blackWhiteTest = blackWhiteImage[:,int(length/2):]
    testWidth,testLength = numpy.shape(blackWhiteTest)
    print("width, length of bw test: " + str(testWidth) + ", " + str(testLength))
    # need to go through all 3x3 patches on left hand side of image and associate them with a value
        # and representative color
    # adding patches to a list
        # then the test data will refer to these patches to get the 6 best matches
    patches=[]
    for z in range(trainingWidth):
        if z==0 or z==trainingWidth-1:
            # this cannot be part of a patch
            continue
        for y in range (trainingLength):
            if y==0 or y==trainingLength-1:
                # this cannot be part of a patch
                continue
            # we have a valid patch here
                # we will append only grey values here in a specific order
                # the locations do not matter
                # order is [top left, middle left, bottom left, middle top, middle ,middle bottom, top right, right middle, bottom left]
            patches.append((blackWhiteTraining[z+1,y-1],
                            blackWhiteTraining[z,y-1],
                            blackWhiteTraining[z-1,y-1],
                            blackWhiteTraining[z+1,y],
                            blackWhiteTraining[z,y],
                            blackWhiteTraining[z-1,y],
                            blackWhiteTraining[z+1,y+1],
                            blackWhiteTraining[z,y+1],
                            blackWhiteTraining[z-1,y+1],
                            (colorImage[z,y,0],colorImage[z,y,1],colorImage[z,y,2])))
    #patches = numpy.array(patches)


    '''
    Run knn with k=6 to get recolored right half
    '''
    resultData = numpy.zeros((testWidth,testLength,3),dtype='uint8')
    #resultData=numpy.zeros((shapes[0], shapes[1] - (shapes[1] / 2), shapes[2]))
    threads=deque()
    for i in range(testWidth):
        if i == 0 or i == testWidth - 1:
            # no contribution
            continue
        for j in range(testLength):
           if j == 0 or j == testLength - 1:
                #no contribution
                continue
           # starting thread to handle this pixel
           thread = threading.Thread(target=thread_knn, args=(i,j,patches,blackWhiteTest,colorImage,resultData))
           thread.start()
           threads.appendleft(thread)
    # waiting on all threads to finish
    while threads:
        t=threads.popleft()
        t.join()
    print("FINISHED COLORING RIGHT SIDE!!!")
    # now resultData holds the recolored right half
    # we combine coloredPixels and resultData and write the output
        #combining both 3d arrays along the horizontal axis (because they have diff # of columns)
    outputImage = numpy.hstack((colorImage, resultData))

    #returning the mashed left and right half
    return outputImage

# idea to use threading to speedup knn
def thread_knn(i,j,patches,blackWhiteTest,colorImage,resultData):
    # if we reached here, we have a valid 3x3 patch
    # now we need to go to the training data and find the 6 closest 3x3 patches
    # form of this is [(distanceValue, (rgb)) , ...]
    # firstly, getting random sample of 1000 patches to choose 6 best from
    thousandPatches = sample(patches,1000)
    sixClosest = []
    for patch in thousandPatches:
        # if we reached here, we have a valid patch
        # calculating the distance value between patches
        distance = (int(patch[0]) - int(blackWhiteTest[i + 1, j - 1])) ** 2 \
                   + (int(patch[1]) - int(blackWhiteTest[i , j-1])) ** 2 \
                   + (int(patch[2]) - int(blackWhiteTest[i - 1, j - 1])) ** 2 \
                   + (int(patch[3]) - int(blackWhiteTest[i+1, j ])) ** 2 \
                   + (int(patch[4]) - int(blackWhiteTest[i, j])) ** 2 \
                   + (int(patch[5]) - int(blackWhiteTest[i-1, j ])) ** 2 \
                   + (int(patch[6]) - int(blackWhiteTest[i+1 , j + 1])) ** 2 \
                   + (int(patch[7]) - int(blackWhiteTest[i + 1, j+1])) ** 2 \
                   + (int(patch[8]) - int(blackWhiteTest[i + 1, j +1])) ** 2
        rgb = patch[9]
        #rgb = (colorImage[z, y, 0], colorImage[z, y, 1], colorImage[z, y, 2])

        # seeing if we can add this data to the list
        if len(sixClosest) < 6:
            # then we can just add it
            # numpy.append(sixClosest,(distance,rgb))
            sixClosest.append((distance, rgb))
        else:
            # then we have to replace this with the greatest value if it is larger
            largest = None
            for closest in sixClosest:
                if largest is None or closest[0] > largest[0]:
                    largest = closest
            # largest = numpy.argmax(sixClosest)
            if largest[0] > distance:
                # then we can replace it
                sixClosest.remove(largest)
                sixClosest.append((distance, rgb))
    # now we have the six closest neighbors of this patch
    # if there is a win in representative colors, we pick that color
    # otherwise pick color with least distance
    # print("GOT CLOSEST NEIGHBORS")
    counter = {}
    for c in range(len(sixClosest)):
        if sixClosest[c][1] in counter:
            counter[sixClosest[c][1]] += 1
        else:
            counter[sixClosest[c][1]] = 0
    allTie = True
    numOccurence = counter[sixClosest[0][1]]
    bestColor = sixClosest[0][1]
    for keys in counter:
        if counter[keys] > numOccurence:
            allTie = False
            bestColor = keys
    if allTie:
        # we have to pick the lowest distance color
        lowestDistance = sixClosest[0][0]
        bestColor = sixClosest[0][1]
        for d in range(len(sixClosest)):
            if sixClosest[d][0] < lowestDistance:
                lowestDistance = sixClosest[d][0]
                bestColor = sixClosest[d][1]
    # we color this rgb
    resultData[i, j, 0] = bestColor[0]
    resultData[i, j, 1] = bestColor[1]
    resultData[i, j, 2] = bestColor[2]
    print("colored pixel "+ str(i)+", "+str(j))


# trains a model
def trainModel(image,model):
    pass

# outputs a color version of a black and white image based on a model
def improved(image,model):
    pass


#test
colorImage = imageToArray("colorImage.jfif")
# making image much smaller for calculation purposes (og image has 1,105,440 pixels to process)
colorWidth,colorLength,colorDim = numpy.shape(colorImage)
#colorImage = colorImage[-int(colorWidth/10):,-int(colorLength/10):,:]
colorImage =colorImage[500:500+512,900:900+512,:]
arrayToImage(colorImage,"croppedImage.jfif")
blackWhiteArray = bwImage(colorImage)
t1=time.time()
recoloredLeftHalf = kmeans(colorImage,5)
t2 = time.time()
print("TOTAL CLUSTERING TIME: "+str(t2-t1))

t3 = time.time()
arrayToImage(recoloredLeftHalf,"recoloredLeftHalf.jfif")
t4 = time.time()
print("RECOLORING LEFT HALF TAKES "+str(t4-t3))

t5 = time.time()
outputBasicAgent = knn(recoloredLeftHalf,blackWhiteArray)
t6 = time.time()
print("KNN TOOK "+str(t6-t5))

arrayToImage(outputBasicAgent,"basicAgentOutput.jfif")

'''
I think it would be more modular if the reading in images was done here, or in a main/test method, and then the methods above used the resulting numpy arrays
Then the methods could return numpy arrays, and down here could save files or compare with the true right half to see how well it did.
'''
