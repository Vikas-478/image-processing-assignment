# Name: Vikas Gandas
# Roll No: 2301010478
# Assignment 4: Feature-Based Traffic Monitoring System

import cv2
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)

# -------- LOAD IMAGE --------
img = cv2.imread("sample_images/sample.png")

if img is None:
    print("Image not found!")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print("Image loaded")

# -------- EDGE DETECTION --------

# Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)
sobel = cv2.magnitude(sobelx, sobely)
sobel = np.uint8(sobel)
cv2.imwrite("outputs/sobel.png", sobel)

# Canny
canny = cv2.Canny(gray, 100, 200)
cv2.imwrite("outputs/canny.png", canny)

print("Edge detection done")

# -------- CONTOURS --------
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

contour_img = img.copy()

for cnt in contours:
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)

    if area > 500:   # filter small noise
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(contour_img, (x,y), (x+w, y+h), (0,255,0), 2)

        print("Object Area:", area, " Perimeter:", perimeter)

cv2.imwrite("outputs/contours.png", contour_img)

print("Contours detected")

# -------- FEATURE EXTRACTION (ORB) --------
orb = cv2.ORB_create()

kp, des = orb.detectAndCompute(gray, None)

feature_img = cv2.drawKeypoints(img, kp, None, color=(0,255,0))

cv2.imwrite("outputs/features.png", feature_img)

print("Feature extraction done")

# -------- ANALYSIS --------
print("\n--- Analysis ---")
print("Canny gives better edges than Sobel.")
print("Contours help detect objects like vehicles.")
print("ORB detects keypoints useful for tracking.")