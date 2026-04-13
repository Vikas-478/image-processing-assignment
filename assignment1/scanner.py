# Name: Vikas Gandas
# Roll No: 2301010478
# Assignment 2: Image Sampling and Quantization


import cv2
import numpy as np
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Load image
img = cv2.imread("sample_images/sample.png")

if img is None:
    print("Image not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# ---------------- Sampling ----------------
high = gray
med = cv2.resize(gray, (256, 256))
low = cv2.resize(gray, (128, 128))

med_up = cv2.resize(med, (512, 512))
low_up = cv2.resize(low, (512, 512))

# ---------------- Quantization ----------------
def quantize(img, levels):
    factor = 256 // levels
    return (img // factor) * factor

q8 = quantize(gray, 256)
q4 = quantize(gray, 16)
q2 = quantize(gray, 4)

# ---------------- Save outputs ----------------
cv2.imwrite("outputs/high.png", high)
cv2.imwrite("outputs/med.png", med_up)
cv2.imwrite("outputs/low.png", low_up)
cv2.imwrite("outputs/q8.png", q8)
cv2.imwrite("outputs/q4.png", q4)
cv2.imwrite("outputs/q2.png", q2)

print("Done! Check outputs folder.")