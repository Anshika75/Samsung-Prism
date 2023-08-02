import cv2
import numpy as np

# Load the two portrait images
image1 = cv2.imread('img1.jpg')
image2 = cv2.imread('img2.jpg')

# Convert the images to grayscale
gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Compute the absolute difference between the two grayscale images
diff = cv2.absdiff(gray_image1, gray_image2)

# Threshold the difference image to highlight the regions with significant differences
threshold = 30  # Adjust this value according to your preference
_, thresholded_diff = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)

# Find contours in the thresholded difference image
contours, _ = cv2.findContours(thresholded_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Draw bounding rectangles around the detected contours
for contour in contours:
    (x, y, w, h) = cv2.boundingRect(contour)
    cv2.rectangle(image1, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.rectangle(image2, (x, y), (x + w, y + h), (0, 0, 255), 2)

# Display the images with highlighted differences
cv2.imshow('Image 1', image1)
cv2.imshow('Image 2', image2)
cv2.imshow('Difference', thresholded_diff)
cv2.waitKey(0)
cv2.destroyAllWindows()