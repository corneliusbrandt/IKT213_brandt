import cv2
import numpy as np


img = cv2.imread("assignment_3/lambo.png")
shapes = cv2.imread("assignment_3/shapes-1.png")
template = cv2.imread("assignment_3/shapes_template.jpg")
img_blur = cv2.GaussianBlur(img, (3,3), sigmaX=0)

def sobel_edge_detection(image):
    sobelxy = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    cv2.imwrite("assignment_3/results/sobel_edge_detect.jpg", sobelxy)

def canny_edge_detection(image, threshold_1, threshold_2):
    cannyedge = cv2.Canny(image, threshold_1, threshold_2)
    cv2.imwrite("assignment_3/results/canny_edge_detection.jpg", cannyedge)

def template_match(image, template):
    
    shapes_grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    template_grey = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    w, h = template_grey.shape[::-1]

    res = cv2.matchTemplate(shapes_grey, template_grey, cv2.TM_CCOEFF_NORMED)
    threshold = 0.9
    loc = np.where( res >= threshold)
    for pt in zip(*loc[::-1]):
        cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,0,255), 1)

    cv2.imwrite("assignment_3/results/template_matching.png",image)

def resize(image, scale_factor:int, up_or_down:str):
    rows, cols, _channels = map(int, image.shape)
    if up_or_down == "up":
        resized_image = cv2.pyrUp(image, dstsize=(scale_factor* cols, scale_factor*rows))
    elif up_or_down == "down":
        resized_image = cv2.pyrDown(image, dstsize=(cols // scale_factor, rows // scale_factor))

    cv2.imwrite(f"assignment_3/results/resized_image_{up_or_down}.png", resized_image)






sobel_edge_detection(img_blur)
canny_edge_detection(img_blur, 50, 50)
template_match(shapes, template)
#Here I just used the lambo image
resize(img, scale_factor=2, up_or_down="up")
resize(img, scale_factor=2, up_or_down="down")