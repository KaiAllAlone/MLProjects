import sklearn
from sklearn.cluster import KMeans
import pandas as pd
import numpy as np
import cv2 as cv
import sys
import os

def bit_compressor(k ,image):
  flat_img=image.reshape((-1,1))
  vals=np.float32(flat_img)
  kmeans=KMeans(n_clusters=k,random_state=0)
  predicted=kmeans.fit_predict(vals)
  means=[]
  for i in range (0,k):
    condition = (predicted == i)
    c=vals[condition]
    mean_val=np.mean(c,axis=0)
    means.append(mean_val)
  means=np.array(means)
  for i in range (0,k):
    means[i]=np.uint8(means[i])
  for i in range(0,predicted.shape[0]):
    flat_img[i]=means[predicted[i]]

  compressed_img=flat_img.reshape(image.shape)
  return compressed_img


def sep_channel(path):
    image=cv.imread(path)
    blue,green,red=cv.split(image)

    blank = np.zeros(shape=blue.shape, dtype=np.uint8)

    # Now merge the channels correctly for display
    resb=bit_compressor(1,blue)
    resg=bit_compressor(50,green)
    resr=bit_compressor(50,red)
    merged_image=cv.merge((resb,resg,resr))
    # cv.imshow('Result sep',merged_image)
    cv.imwrite('Result.png',merged_image)
    # cv.waitKey(0)
    return


def whole(path,k):
    image=cv.imread(path)
    comp_img=bit_compressor(k,image)
    # cv.imshow('Result whole',comp_img)
    cv.imwrite('Result.png',comp_img)
    # cv.waitKey(0)
    return

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide an image path as argument")
        print("Usage: python image_compressor.py <image_path>")
        sys.exit(1)
        
    path = sys.argv[1]
    print(f"Attempting to read image from: {os.path.abspath(path)}")
    
    orig = cv.imread(path)
    if orig is None:
        print(f"Error: Could not read image at {path}")
        sys.exit(1)
    
    print(f"Successfully read image with shape: {orig.shape}")
    
    # Save and show original
    cv.imwrite('original.png', orig)
    cv.imshow('Original', orig)
    
    print("Processing image...")
    # Using separate channel compression for better quality
    sep_channel(path)
    
    print("Reading compressed result...")
    res = cv.imread('Result.png')
    if res is None:
        print("Error: Could not read compressed result")
        sys.exit(1)
        
    print(f"Compressed image shape: {res.shape}")
    
    # Post-processing
    res = cv.dilate(res, (3,3))
    res = cv.erode(res, (3,3))
    
    # Save and show result
    cv.imwrite('Result.png', res)
    cv.imshow('Result', res)
    print("Press any key to close the windows...")
    cv.waitKey(0)
    cv.destroyAllWindows()

    