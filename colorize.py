# I tried to keep same format as the picture you sent

# WE ARE USING SCIKIT ONLY FOR IMAGE REPRESENTATION HERE! NO BUILT IN METHODS ARE USED IN OUR IMPLEMENTATION
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
def knn(image, numMeans):
    pass

# trains a model
def trainModel(image,model):
    pass

# outputs a color version of a black and white image based on a model
def improved(image,model):
    pass


#test
bwImage("colorImage.jfif")