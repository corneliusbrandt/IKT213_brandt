import cv2


img = cv2.imread("assignment_3/lambo.png")
img_blur = cv2.GaussianBlur(img, (3,3), sigmaX=0)
cv2.imshow('blured', img_blur)

def sobel_edge_detection(image):
    sobelxy = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    cv2.imshow('Sobel XY', sobelxy)
    cv2.waitKey(0)
    cv2.imwrite("assignment_3/results/sobel_edge_detect.jpg", sobelxy)

def canny_edge_detection(image, threshold_1, threshold_2):
    cannyedge = cv2.Canny(image, threshold_1, threshold_2)
    cv2.imshow('Canny Edge Detection', cannyedge)
    cv2.waitKey(0)
    cv2.imwrite("assignment_3/results/canny_edge_detection.jpg", cannyedge)

    


sobel_edge_detection(img_blur)
canny_edge_detection(img_blur, 50, 50)