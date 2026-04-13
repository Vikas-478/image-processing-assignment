# Name: Vikas Gandas
# Roll No: 2301010478
# Assignment 3: Medical Image Compression & Segmentation

import cv2
import numpy as np
import os

os.makedirs("outputs", exist_ok=True)

# ---------------- LOAD IMAGE ----------------
img = cv2.imread("sample_images/sample.png", 0)

if img is None:
    print("Image not found!")
    exit()

cv2.imwrite("outputs/original.png", img)
print("Image loaded")

# ---------------- RLE COMPRESSION ----------------
def rle_encode(image):
    pixels = image.flatten()
    encoded = []
    prev = pixels[0]
    count = 1

    for pixel in pixels[1:]:
        if pixel == prev:
            count += 1
        else:
            encoded.append((prev, count))
            prev = pixel
            count = 1

    encoded.append((prev, count))
    return encoded

rle = rle_encode(img)

original_size = img.size
compressed_size = len(rle) * 2

compression_ratio = original_size / compressed_size
savings = (1 - (compressed_size / original_size)) * 100

print("\n--- Compression ---")
print("Compression Ratio:", compression_ratio)
print("Storage Savings (%):", savings)

# ---------------- SEGMENTATION ----------------

# Global Threshold
_, global_thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
cv2.imwrite("outputs/segmented_global.png", global_thresh)

# Otsu Threshold
_, otsu = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
cv2.imwrite("outputs/segmented_otsu.png", otsu)

print("Segmentation done")

# ---------------- MORPHOLOGY ----------------
kernel = np.ones((5,5), np.uint8)

dilation = cv2.dilate(otsu, kernel, iterations=1)
erosion = cv2.erode(otsu, kernel, iterations=1)

cv2.imwrite("outputs/dilation.png", dilation)
cv2.imwrite("outputs/erosion.png", erosion)

print("Morphological processing done")

# ---------------- ANALYSIS ----------------
print("\n--- Analysis ---")
print("Otsu gives better automatic segmentation.")
print("Dilation fills gaps in regions.")
print("Erosion removes noise.")