import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import time


def preprocess_fingerprint(image_path):
    img = cv2.imread(image_path, 0)
    _, img_bin = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return img_bin


def match_fingerprints_orb(img1_path, img2_path):
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)
 
    # Initialize ORB detector
    orb = cv2.ORB_create(nfeatures=1000)
 
    # Find keypoints and descriptors
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, None  # Return 0 matches if no descriptors found
 
    # Use Brute-Force Matcher with Hamming distance
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
 
    # KNN Match
    matches = bf.knnMatch(des1, des2, k=2)
 
    # Apply Lowe's ratio test (keep only good matches)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
 
    # Draw only good matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), match_img


def match_fingerprints_sift(img1_path, img2_path):
    img1 = preprocess_fingerprint(img1_path)
    img2 = preprocess_fingerprint(img2_path)
 
    # Initialize SIFT detector
    sift = cv2.SIFT_create(nfeatures=1000)
 
    # Find keypoints and descriptors
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    if des1 is None or des2 is None:
        return 0, None  # Return 0 matches if no descriptors found
 
    # FLANN parameters (KD-tree for SIFT)
    index_params = dict(algorithm=1, trees=5)  # KD-tree
    search_params = dict(checks=50)  # Number of checks for nearest neighbors
    flann = cv2.FlannBasedMatcher(index_params, search_params)
 
    # KNN Match
    matches = flann.knnMatch(des1, des2, k=2)
 
    # Apply Lowe's ratio test (keep only good matches)
    good_matches = [m for m, n in matches if m.distance < 0.7 * n.distance]
 
    # Draw only good matches
    match_img = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return len(good_matches), match_img


def process_images(image1_path, image2_path, results_folder):
    # Create results folder if it does not exist
    os.makedirs(results_folder, exist_ok=True)

    threshold = 20 

    # Match using ORB
    start = time.perf_counter()
    orb_match_count, orb_match_img = match_fingerprints_orb(image1_path, image2_path)
    end = time.perf_counter()
    print(f"ORB matching took {end - start:.4f} seconds")
    
    # Save match image in the results folder
    if orb_match_img is not None:
            match_img_filename = f"orb_match.png"
            match_img_path = os.path.join(results_folder, match_img_filename)
            cv2.imwrite(match_img_path, orb_match_img)
            print(f"Saved match image at: {match_img_path} with {orb_match_count} good matches")
            if orb_match_count > threshold:
                print("ORB: MATCHED")
            else:
                print("ORB: UNMATCHED")


    # Match using SIFT
    start = time.perf_counter()
    sift_match_count, sift_match_img = match_fingerprints_sift(image1_path, image2_path)
    end = time.perf_counter()
    print(f"SIFT matching took {end - start:.4f} seconds")

    # Save match image in the results folder
    if sift_match_img is not None:
        match_img_filename = f"sift_match.png"
        match_img_path = os.path.join(results_folder, match_img_filename)
        cv2.imwrite(match_img_path, sift_match_img)
        print(f"Saved match image at: {match_img_path} with {sift_match_count} good matches")
        if sift_match_count > threshold:
            print("SIFT: MATCHED")
        else:
            print("SIFT: UNMATCHED")

image1_path = r"figureprint_matching\UiA front1.png"
image2_path = r"figureprint_matching\UiA front3.jpg"
results_folder = r"figureprint_matching\results\uia_match"
process_images(image1_path, image2_path, results_folder)