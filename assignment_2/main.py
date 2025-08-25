import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import os


image = cv.imread('assignment_2/lena-2.png')

def padding(image, border_width):
    image_with_border = cv.copyMakeBorder(image, border_width, border_width, border_width, border_width, cv.BORDER_REFLECT)
    return image_with_border

def crop(image, x_0, x_1, y_0, y_1):
    cropped = image[y_0:y_1, x_0:x_1]
    return cropped

def resize(image, width, height):
    resized = cv.resize(image, (width, height))
    return resized

def copy(image, emptyPictureArray):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    emptyPictureArray = np.zeros((height, width, channels), dtype=np.uint8)
    emptyPictureArray[:] = image[:]
    return emptyPictureArray

def grayscale(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return gray

def hsv(image):
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV)
    return hsv

def hue_shifted(image, emptyPictureArray, hue):
    height = image.shape[0]
    width = image.shape[1]
    channels = image.shape[2]
    emptyPictureArray = np.zeros((height, width, channels), dtype=np.uint8)
    emptyPictureArray[:] = image[:]
    emptyPictureArray += hue
    return emptyPictureArray

def smoothing(image):
    dst = cv.GaussianBlur(image, (15, 15), 0)
    return dst

def rotation(image, rotation_angle):
    if rotation_angle == 90:
        rotated = cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
    elif rotation_angle == 180:
        rotated = cv.rotate(image, cv.ROTATE_180)
    elif rotation_angle == 270:
        rotated = cv.rotate(image, cv.ROTATE_90_COUNTERCLOCKWISE)
    return rotated

def save_image(image, filename : str, target_dir = "assignment_2/results"):
    os.makedirs(target_dir, exist_ok=True)
    save_path = os.path.join(target_dir, filename)
    cv.imwrite(save_path, image)



#--------------------------------------------------------------------------------------------
#--------------------------------------------------------------------------------------------
#Show all images. Press 0 to close them all
cv.imshow('image', image)
cv.imshow('padded', padding(image, 100))
cv.imshow('cropped', crop(image, 79, 469, 79, 469))
cv.imshow('resized', resize(image, 200, 200))
cv.imshow('copied', copy(image, np.array([])))
cv.imshow('grayscale', grayscale(image))
cv.imshow('hsv', hsv(image))
cv.imshow('hue_shifted', hue_shifted(image, np.array([]), 50))
cv.imshow('smoothing', smoothing(image))
cv.imshow('rotated', rotation(image, 180))
print("Press 0 to close all pictures")

cv.waitKey(0)
cv.destroyAllWindows()

#Save images to results folder
save_image(padding(image, 100), 'padded.png')
save_image(crop(image, 79, 469, 79, 469), 'cropped.png')
save_image(resize(image, 200, 200), 'resized.png')
save_image(copy(image, np.array([])), 'copied.png')
save_image(grayscale(image), 'grayscale.png')
save_image(hsv(image), 'hsv.png')
save_image(hue_shifted(image, np.array([]), 50), 'hue_shifted.png')
save_image(smoothing(image), 'smoothing.png')
save_image(rotation(image, 180), 'rotated_180.png')
save_image(rotation(image, 90), 'rotated_90.png')
