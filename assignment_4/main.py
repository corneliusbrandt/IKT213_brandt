import cv2
import numpy as np
from matplotlib import pyplot as plt

filename_reference = 'assignment_4/reference_img.png'
img_reference = cv2.imread(filename_reference)


def harris_corner_detection(reference_img):
    gray = cv2.cvtColor(reference_img,cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray,2,3,0.04)

    #result is dilated for marking the corners, not important
    dst = cv2.dilate(dst,None)

    # Threshold for an optimal value, it may vary depending on the image.
    reference_img[dst>0.01*dst.max()]=[0,0,255]

    cv2.imshow('dst',reference_img)
    if cv2.waitKey(0) & 0xff == 27:
        cv2.destroyAllWindows()
    cv2.imwrite('assignment_4/results/harris.png', reference_img)


def image_alignment(image_to_align, reference_image, max_features, good_match_percent):
    MIN_MATCH_COUNT = max_features

    # Convert images to grayscale
    gray_image_to_align = cv2.cvtColor(image_to_align, cv2.COLOR_BGR2GRAY)
    gray_reference_image = cv2.cvtColor(reference_image, cv2.COLOR_BGR2GRAY)

    # Initiate SIFT detector
    sift = cv2.SIFT_create()

    # Find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(gray_reference_image, None)
    kp2, des2 = sift.detectAndCompute(gray_image_to_align, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # Store all the good matches as per Lowe's ratio test.
    good_matches = []
    for m, n in matches:
        if m.distance < good_match_percent * n.distance:
            good_matches.append(m)

    if len(good_matches) > MIN_MATCH_COUNT:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Find the homography matrix
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()

        M_warp = np.linalg.inv(M)
        h, w = gray_reference_image.shape
        aligned = cv2.warpPerspective(image_to_align, M_warp, (w, h))
       
    else:
        print("Not enough matches are found - {}/{}".format(len(good_matches), MIN_MATCH_COUNT))
        matchesMask = None
    
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,
                       flags=2)

    img_matches = cv2.drawMatches(gray_reference_image, kp1, gray_image_to_align, kp2, good_matches, None, **draw_params)

    cv2.imwrite('assignment_4/results/aligned.png', aligned)
    cv2.imwrite('assignment_4/results/matches.png', img_matches)
    plt.figure(figsize=(20, 10))
    plt.subplot(1, 2, 1)
    plt.title('Aligned Image')
    plt.imshow(cv2.cvtColor(aligned, cv2.COLOR_BGR2RGB))
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title('Matches')
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.axis('off')
    plt.show()
    

harris_corner_detection(img_reference)

image_to_align = 'assignment_4/align_this.jpg'
reference_image = 'assignment_4/reference_img.png'

img_to_align = cv2.imread(image_to_align)
ref_img = cv2.imread(reference_image)
image_alignment(img_to_align, ref_img, 10, 0.7)
