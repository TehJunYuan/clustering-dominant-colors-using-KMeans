# ---------------------------------------------------------------------------- 
# import library/packages

import cv2
from sklearn.cluster import KMeans
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np

# example image

picture1 = 'colors_01.jpg'
picture2 = 'colors_02.png'
picture3 = 'colors_03.png'

# ---------------------------------------------------------------------------- 

# ---------------------------------------------------------------------------- 
# 1: show the clusters in diagram 
# you may uncomment code below and run it.
# ---------------------------------------------------------------------------- 

# read image using cv2
# img = cv2.imread(picture3)
# # convert from BGR to RGB
# img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # get rgb values from image to 1D array
# r, g, b = cv2.split(img)
# r = r.flatten()
# g = g.flatten()
# b = b.flatten()
# # show the diagram
# fig = plt.figure()
# ax = fig.add_subplot(projection='3d')
# # plotting the rgb in diagram
# ax.scatter(r,g,b)
# ax.set_title("Cluster in plot")
# plt.show()

# ---------------------------------------------------------------------------- 

# ---------------------------------------------------------------------------- 
# 2: visualize the cluster and show the color palette
#
# DominantColors class
# - attributes: CLUSTERS, IMAGE, COLORS, LABELS
# - constructor: required user input image and the number of clusters
# - method: dominantColors(), rgb_to_hex(), plotClusters(), plotHistogram()
# 
# Method: 
# dominantColors()
# - used to read image path, convert image format, k-means algo process
# - return int 
#
# rgb_to_hex()
# - used to convert the rgb format to hex format
#
# plotClusters()
# - return cluster visualization in diagram
# 
# plotHistogram()
# - return color palette
#
# ---------------------------------------------------------------------------- 

class DominantColors:

    CLUSTERS = None
    IMAGE = None
    COLORS = None
    LABELS = None
    
    def __init__(self, image, clusters):
        self.CLUSTERS = clusters
        self.IMAGE = image
        
    def dominantColors(self):
    
        # read image
        img = cv2.imread(self.IMAGE)
        
        # convert to rgb from bgr
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
        # reshaping to a list of pixels
        img = img.reshape((img.shape[0] * img.shape[1], 3))
        
        # save image after operations
        self.IMAGE = img
        
        # using k-means to cluster pixels
        kmeans = KMeans(n_clusters = self.CLUSTERS)
        kmeans.fit(img)
        
        #the cluster centers are our dominant colors.
        self.COLORS = kmeans.cluster_centers_
        
        #save labels
        self.LABELS = kmeans.labels_
        
        #returning after converting to integer from float
        return self.COLORS.astype(int)

    def rgb_to_hex(self, rgb):
        return '#%02x%02x%02x' % (int(rgb[0]), int(rgb[1]), int(rgb[2]))

    def plotClusters(self):

        #plotting 
        fig = plt.figure()
        ax = fig.add_subplot(projection='3d')       
        for label, pix in zip(self.LABELS, self.IMAGE):
            ax.scatter(pix[0], pix[1], pix[2], color = self.rgb_to_hex(self.COLORS[label]))
        plt.show()

    def plotHistogram(self):

        # labels form 0 to no. of clusters
        numLabels = np.arange(0, self.CLUSTERS+1)

        # create frequency count tables    
        (hist, _) = np.histogram(self.LABELS, bins = numLabels)
        hist = hist.astype("float")
        hist /= hist.sum()

        # appending frequencies to cluster centers
        colors = self.COLORS

        # descending order sorting as per frequency count
        colors = colors[(-hist).argsort()]
        hist = hist[(-hist).argsort()] 

        # creating empty chart
        chart = np.zeros((50, 500, 3), np.uint8)
        start = 0

        # creating color rectangles
        for i in range(self.CLUSTERS):
            end = start + hist[i] * 500

            # getting rgb values
            r = colors[i][0]
            g = colors[i][1]
            b = colors[i][2]

            # using cv2.rectangle to plot colors
            cv2.rectangle(chart, (int(start), 0), (int(end), 50), (r,g,b), -1)
            start = end	

        # display chart
        plt.figure()
        plt.axis("off")
        plt.imshow(chart)
        plt.show()

# ---------------------------------------------------------------------------- 
# case 2: programme start 
# 1. store 'image path' in variable (picture) 
# 2. set clusters
# 3. pass the image and clusters to DominantColors class
# 4. use the dominantColors method to find the k-means
# 5. call the plotClusters method to show the clusters visualization or
#    call the plotHistogram method to show the color palette
# 
# uncomment code below and run it
# ---------------------------------------------------------------------------- 

# numbers of cluster
clusters = 5

# pass it to DominantColors and call the function
dc = DominantColors(picture3, clusters) 
colors = dc.dominantColors()

# # display the cluster in 3D diagram 
# dc.plotClusters()

# display the color palette
dc.plotHistogram()

# ---------------------------------------------------------------------------- 