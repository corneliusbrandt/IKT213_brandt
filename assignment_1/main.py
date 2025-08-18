import cv2 as cv
import numpy as np

def print_image_information(image_path):
    # Load the image
    image = cv.imread(image_path)
    print(" The height of the image is: ", image.shape[0])
    print(" The width of the image is: ", image.shape[1])
    print(" The number of channels in the image is: ", image.shape[2])
    print(" The size of the image is: ", image.size)
    print(" The data type of the image is: ", image.dtype)

# Function to create cam.txt file with camera information
def create_cam_txt_file(fps, width, height):
    with open('assignment_1/solutions/cam.txt', 'w') as file:
        file.write(f"FPS: {fps}\n")
        file.write(f"Width: {width}\n")
        file.write(f"Height: {height}\n")


# Exersice V: Camera Information
cam = cv.VideoCapture(0)
frame_width = int(cam.get(cv.CAP_PROP_FRAME_WIDTH))
frame_height = int(cam.get(cv.CAP_PROP_FRAME_HEIGHT))
fps = int(cam.get(cv.CAP_PROP_FPS))

# Print information and create cam.txt file
print_image_information('assignment_1/lena-1.png')
create_cam_txt_file(fps, frame_width, frame_height)
