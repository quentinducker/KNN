import os
from PIL import Image
import pandas as pd
import numpy as np
from matplotlib.image import imread

directory = "/Users/quentinducker/Documents/Spring 2024/ITS365/KNN Project/KNN"
samples = os.listdir(f"{directory}/samples")

'''
print( 
    "\n".join( samples ) 
    )
'''

imgDataFrames = pd.DataFrame()
imgFeaturesDataFrames = pd.DataFrame()

for index, sample in enumerate(samples):

    # open image and make an array
    img = Image.open(f"{directory}/samples/{sample}")
    img = img.convert('RGBA')
    imgMatrix = np.array(img)

    if( len(imgMatrix.shape) != 3):
        raise Exception("img not right size")

    # show the image and ask questions
    img.show()

    numberOfPeaks = int(input("How many peaks (including inner layers) are?"))
    solid = int(input("is the flame solid?"))
    numberOfColors = int(input("How many color are there?"))
    numberOfStrayFlames = int(input("How many stray flames are there?"))

    # fatten img add it to the images dataframe
    # make array from answers and add it to the features dataframe 
    flatImg = imgMatrix.flatten().reshape((1, -1))
    
    imgFeatures = np.array([ [numberOfPeaks, solid, numberOfColors, numberOfStrayFlames] ])
    imgFeatures.flatten().reshape((1, -1))

    imgDataFrame = pd.DataFrame(flatImg)
    imgFeatureDataFrame = pd.DataFrame(imgFeatures)

    imgDataFrames = pd.concat([imgDataFrames, imgDataFrame], axis=0, ignore_index=True)
    imgDataFrames.reset_index()

    imgFeaturesDataFrames = pd.concat([imgFeaturesDataFrames, imgFeatureDataFrame], axis=0, ignore_index=True)
    imgFeatureDataFrame.reset_index()

    print(imgDataFrames)
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print(imgFeaturesDataFrames)
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()
    print()

    img.close()

# make csv's from the dataframes store them in the right directory

allImgData = pd.concat(imgDataFrames, axis=1)
allImgFeatures = pd.concat(imgFeaturesDataFrames, axis=1)

allImgData.to_csv(r'/Users/quentinducker/Documents/Spring 2024/ITS365/KNN Project/KNN/ImageCsv/imgData.csv')
allImgFeatures.to_csv(r'/Users/quentinducker/Documents/Spring 2024/ITS365/KNN Project/KNN/ImageFeatures/imgFeatures.csv')