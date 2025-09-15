import cv2


img = cv2.imread("assignment_3\lambo.png")
img_blur = cv2.GaussianBlur(img, (3,3), sigmaX=0)

def sobel_edge_detection(image):
    sobelxy = cv2.Sobel(src=image, ddepth=cv2.CV_64F, dx=1, dy=1, ksize=1)
    cv2.imshow('Sobel XY', sobelxy)
    cv2.waitKey(0)


sobel_edge_detection(img)