"""
Adrian's imuitls library (at least a portion of it) from 'Practical Python With OpenCV + Case Studies'
Started writing code on 10/10/2020
"""

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

#Imports
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.mplot3d import Axes3D

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def translate(image, x, y):
    """Function to translate an image

    Args:
        image (numpy.array): Array representation of an image.
        x (int): How many pixels to translate image left/right. Negative values are left translation, positive values are right translation.
        y (int): How many pixels to translate image up/down. Negative values are downward translation, positive values are upward translation.

    Returns:
        numpy.array: Numpy array representation of the input image translated in the desired manner.
    """
    
    #Construct translation matrix
    M = np.float32([[1,0,x], [0,1,y]])
    
    #Translate the image
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))
    
    #Return the shifted image
    return shifted

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def rotate(image, angle, center = None, scale = 1.0):
    """Function to rotate an image

    Args:
        image (numpy.array): Array representation of an image.
        angle (float): Angle of rotation in degrees. Positve values rotate the image counterclockwise, positive values rotate the image clockwise
        center (tuple, optional): Tuple in shape of (centerX, centerY) = (columnLocation, rowLocation). The tuple represents the coordinates of the point of rotation. Defaults to None -> image center.
        scale (float, optional): Factor to scale rotated image by. Defaults to 1.0.

    Returns:
        numpy.array: Numpy array representation of the input image rotated and scaled in the desired manner
    """
    
    #Grab image dimensions
    h,w = image.shape[:2]
    
    #If the center is None, set the find the center of the image and use this point as point of rotation in rotation matrix
    if center is None:
        center = (w//2,h//2)
    
    #Get the rotation matrix
    M = cv2.getRotationMatrix2D(center, angle, scale)
    
    #Rotate and return the rotated image
    rotated = cv2.warpAffine(image, M, (w,h))
    return rotated

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    """Function to resize an image along a specified axis while maintaining the aspect ratio.

    Args:
        image (numpy.array): Array representation of target image.
        width (int, optional): Desired width to resize image to. Defaults to None.
        height (int, optional): Desired height to resize image to. Defaults to None.
        inter (cv2.INTER, optional): Interpolation method. Defaults to cv2.INTER_AREA.

    Returns:
        numpy.array: Numpy array representation of input image resized along specified axis to desired dimension while maintaining the aspect ration of the input image. If a target dimension is not specified, the orignal image is returned.
    """
    
    #First grab the original height and width of the image
    (h,w) = image.shape[:2]
    
    #If width and height are both none, return the orignal image
    if width is None and height is None:
        return image
    
    #If width is none and height is specified, resize along height dimension
    elif width is None:
        r = height/float(h)
        dimensions = (int(w*r), height)
    
    #If height is none and width is specified, resize along the width dimension
    else:
        r = width/float(w)
        dimensions = (width, int(h*r))
    
    #Return the resized image
    return cv2.resize(image, dimensions, interpolation = inter)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
def gray_histogram(grayscale_image, mask = None, show = False, title = "Grayscale Histogram"):
    """Computes histogram for a grayscale image.

    Args:
        grayscale_image (numpy.array): Array representation of a grayscale image.
        mask (numpy.array, optional): Mask to apply to input image. Histogram only calculated for this region. Defaults to None.
        show (bool, optional): Flag which indicates if histogram should be displayed. Defaults to False.
        title (str, optional): Title of histogram plot. Defaults to "Grayscale Histogram".
    """
    
    #Calculate histogram
    hist = cv2.calcHist([grayscale_image], [0], mask, [256], [0,256])
    
    #Plot histogram
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("bins")
    ax.set_ylabel("Number of Pixels")
    ax.plot(hist)
    ax.set_xlim([0,256])
    if show:
        fig.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def color_histogram(BGR_image, mask = None, show = False, title = "Color Histogram"):
    """Compute color histogram for a BGR image.

    Args:
        BGR_image (numpy.array): Array representation of an BGR image.
        mask (numpy.array, optional): Mask to apply to input image. Histogram only calculated in this region. Defaults to None.
        show (bool, optional): Flag which indicates if histogram should be displayed. Defaults to False.
        title (str, optional): Title of histogram plot. Defaults to "Color Histogram".
    """
    
    #Split image into individual channels
    chans = cv2.split(BGR_image)
    
    #List of colors
    colors = ["Blue", "Green", "Red"]
    
    #List to hold histogram from each color channel
    histograms = []
    
    #Initialize figure and axis for plotting and do some formatting
    fig, ax = plt.subplots()
    ax.set_title(title)
    ax.set_xlabel("Pixel Value")
    ax.set_ylabel("Number of Pixels")
    
    #Zip through channels and colors and plot each one
    for chan, color in zip(chans, colors):
        hist = cv2.calcHist([chan], [0], mask, [256], [0,256])
        histograms.append(hist)
        ax.plot(hist, label = color, color = color)
        
    if show:
        fig.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def two_dimensional_histograms(BGR_image, mask = None, hist_size = [256, 256], figsize = (15,5), show = False):
    """Computes 2D histograms (heatmaps) for each combination of color channels.

    Args:
        BGR_image (numpy.array): Array representation of a BGR image.
        mask (numpy.array, optional): Mask to apply to input image. Histogram only calculated in this region. Defaults to None.
        hist_size (list, optional): Number of bins for histograms representing each respective channel. Defaults to [256, 256].
        figsize (tuple, optional): Size of figure which displays histograms. Defaults to (15,5).
        show (bool, optional): Flag which indicates if histogram should be displayed. Defaults to False.
    """
    
    #Get individual color channels
    chans = cv2.split(BGR_image)
    
    #Initialize figure for plotting
    fig = plt.figure(figsize = figsize)
    
    #Add a subplot to the figure
    #Code below indicates a subplot structure of 1 row and 3 columns. This particular subplot will be added at the first position of the flattened subplot matrix
    ax = fig.add_subplot(131)
    
    #Green - Blue Heatmap (histogram)
    #See page 96 of Practical Python and OpenCV for description of each function input
    hist = cv2.calcHist([chans[1], chans[0]], [0,1], mask, hist_size, [0,256, 0, 256])
    color_heatmap = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("Green and Blue 2D Histogram")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(color_heatmap, cax = cax)
    
    
    #Green - Red Heatmap
    #See page 96 of Practical Python and OpenCV for description of each function input
    ax = fig.add_subplot(132)
    hist = cv2.calcHist([chans[1], chans[2]], [0,1], mask, hist_size, [0,256, 0, 256])
    color_heatmap = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("Green and Red 2D Histogram")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(color_heatmap, cax = cax)
    
    #Blue - Red Heatmap
    #See page 96 of Practical Python and OpenCV for description of each function input
    ax = fig.add_subplot(133)
    hist = cv2.calcHist([chans[0], chans[2]], [0,1], mask, hist_size, [0,256, 0, 256])
    color_heatmap = ax.imshow(hist, interpolation = "nearest")
    ax.set_title("Blue and Red 2D Histogram")
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    plt.colorbar(color_heatmap, cax = cax)
    
    #Title on figure
    fig.suptitle("2D histograms with shapes: {} and with {} values".format(hist.shape, hist.flatten().shape[0]))
    
    #Show if need be
    if show:
        fig.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def three_dimensional_histogram(BGR_image, mask = None, size = 5000, bins = 8, figsize = (15,15), show = False):
    """Fucntion to generate a 3D histogram of a BGR image. See pages 99-102 of Practical Python with OpenCV + Case Studies 4th Edition for more detailed explanation of code.

    Args:
        BGR_image (numpy.array): Array representation of a BGR image.
        mask (numpy.array, optional): Mask to apply to input image. Histogram only calculated within this mask. Defaults to None.
        size (int, optional): Size of largest bin. Defaults to 5000.
        bins (int, optional): Number of bins. Defaults to 8.
        figsize (tuple, optional): Size of the displayed figure. Defaults to (15,15).
        show (bool, optional): Flag which indicates if image should be shown shown. Defaults to False.
    """
    
    #Ensure size is a float
    size = float(size)
    
    #Ensure bins is an integer
    bins = int(bins)
    
    #Calculate 3 channel histogram
    hist = cv2.calcHist([BGR_image], [0,1,2], mask, [bins, bins, bins], [0, 256, 0, 256, 0, 256])
    
    #Create figure
    fig = plt.figure(figsize = figsize)
    ax = fig.add_subplot(111, projection = "3d")
    ratio = size / np.max(hist)
    
    for (x, plane) in enumerate(hist):
        for (y, row) in enumerate(plane):
            for (z, col) in enumerate(row):
                if hist[x][y][z] > 0.0:
                    siz = ratio * hist[x][y][z]
                    rgb = (z/(bins - 1), y/(bins - 1), x/(bins - 1))
                    ax.scatter(x, y, z, s = siz, facecolors = rgb)
    
    #Added a title and show if need be
    ax.set_title("3D histogram with shape: {} and with {} values".format(hist.shape, hist.flatten().shape[0]))
    if show:
        fig.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show(img, title = False):
    """Function to quickly show an RGB image with matplotlib.pyplot.imshow()

    Args:
        img (numpy.array): Array representation of an image.
        title (bool, optional): Title of the displayed image. Defaults to False. -> No title.
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if title:
        plt.title(title)
    plt.axis("off")
    plt.imshow(img)
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def show_gray(img, title = False):
    """Function to quickly show a grayscale image with plt.imshow()

    Args:
        img (numpy.array): Array representation of an image.
        title (bool, optional): Title of the displayed image. Defaults to False -> No title.
    """
    #Show the image with gray colormapping
    plt.imshow(img, cmap = "gray")
    if title:
        plt.title(title)
    plt.axis("off")
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

def grab_contours(contour_output):
    """Fucntion to grab list of contours from cv2.findContours.

    Args:
        contour_output (Tuple): Output from cv2.findContours.

    Raises:
        Exception: If the length of the contours is not 2 or 3, an exception is raised.
    """
    
    if len(contour_output) == 2:
        return contour_output[0]
    
    elif len(contour_output) == 3:
        return contour_output[1]
    
    else:
        raise Exception(("Contours tuple must have a length of 2 or 3, otherwise openCV has changed their cv2.findContours signature. Please refer to OpenCV's documentation"))

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#