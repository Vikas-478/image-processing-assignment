# Name: Vikas Gandas
# Roll No: 2301010478
# Assignment 2: Image Restoration

import cv2
import numpy as np
import os
import math

os.makedirs("outputs", exist_ok=True)

# Load image
img = cv2.imread("sample_images/sample.png")

if img is None:
    print("Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imwrite("outputs/original.png", gray)

print("Image loaded")

# -------- Noise --------

# Gaussian Noise
gaussian = gray + np.random.normal(0, 25, gray.shape)
gaussian = np.clip(gaussian, 0, 255).astype(np.uint8)
cv2.imwrite("outputs/gaussian_noise.png", gaussian)

# Salt & Pepper Noise
sp = gray.copy()
prob = 0.02
rand = np.random.rand(*gray.shape)

sp[rand < prob] = 0
sp[rand > 1 - prob] = 255
cv2.imwrite("outputs/sp_noise.png", sp)

print("Noise added")

# -------- Filters --------

mean = cv2.blur(gaussian, (5,5))
median = cv2.medianBlur(sp, 5)
gauss = cv2.GaussianBlur(gaussian, (5,5), 0)

cv2.imwrite("outputs/mean.png", mean)
cv2.imwrite("outputs/median.png", median)
cv2.imwrite("outputs/gaussian.png", gauss)

print("Filtering done")

# -------- Metrics --------

def mse(a, b):
    return np.mean((a - b) ** 2)

def psnr(a, b):
    m = mse(a, b)
    if m == 0:
        return 100
    return 20 * math.log10(255.0 / math.sqrt(m))

print("\n--- Results ---")

print("Mean Filter → MSE:", mse(gray, mean), " PSNR:", psnr(gray, mean))
print("Median Filter → MSE:", mse(gray, median), " PSNR:", psnr(gray, median))
print("Gaussian Filter → MSE:", mse(gray, gauss), " PSNR:", psnr(gray, gauss))

# -------- Analysis --------
print("\nBest for Salt & Pepper: Median Filter")
print("Best for Gaussian Noise: Gaussian Filter")