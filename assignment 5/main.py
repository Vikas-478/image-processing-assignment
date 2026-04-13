# Name: Vikas Gandas
# Roll No: 2301010478
# Assignment 5: Intelligent Image Enhancement & Analysis System

import cv2
import numpy as np
import os
import math

os.makedirs("outputs", exist_ok=True)

print("Intelligent Image Processing System")

# -------- LOAD IMAGE --------
img = cv2.imread("sample_images/sample.png")

if img is None:
    print("Image not found!")
    exit()

img = cv2.resize(img, (512, 512))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

cv2.imwrite("outputs/gray.png", gray)

# -------- NOISE --------
gaussian = gray + np.random.normal(0, 25, gray.shape)
gaussian = np.clip(gaussian, 0, 255).astype(np.uint8)

sp = gray.copy()
rand = np.random.rand(*gray.shape)
sp[rand < 0.02] = 0
sp[rand > 0.98] = 255

noisy = gaussian
cv2.imwrite("outputs/noisy.png", noisy)

# -------- RESTORATION --------
mean = cv2.blur(noisy, (5,5))
median = cv2.medianBlur(noisy, 5)
gauss = cv2.GaussianBlur(noisy, (5,5), 0)

restored = gauss
cv2.imwrite("outputs/restored.png", restored)

# -------- ENHANCEMENT --------
enhanced = cv2.equalizeHist(restored)
cv2.imwrite("outputs/enhanced.png", enhanced)

# -------- SEGMENTATION --------
_, thresh = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("outputs/segmented.png", thresh)

# -------- MORPHOLOGY --------
kernel = np.ones((5,5), np.uint8)
thresh = cv2.dilate(thresh, kernel, 1)
thresh = cv2.erode(thresh, kernel, 1)

# -------- EDGE DETECTION --------
edges = cv2.Canny(enhanced, 100, 200)
cv2.imwrite("outputs/edges.png", edges)

# -------- FEATURES --------
orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(enhanced, None)

features = cv2.drawKeypoints(img, kp, None, color=(0,255,0))
cv2.imwrite("outputs/features.png", features)

# -------- METRICS --------
def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(m))

print("\n--- Performance ---")
print("MSE:", mse(gray, restored))
print("PSNR:", psnr(gray, restored))

# -------- FINAL --------
print("\nSystem complete")
print("Image enhanced, restored, segmented and analyzed successfully.")