import cv2
from skimage.metrics import structural_similarity as ssim

def compare_images(image1_path, image2_path):
    # Load the images
    image1 = cv2.imread(image1_path)
    image2 = cv2.imread(image2_path)

    # Convert the images to grayscale
    gray_image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray_image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # Compute the Structural Similarity Index (SSIM)
    similarity_score = ssim(gray_image1, gray_image2)

    return similarity_score

# Example usage
image1_path = 'img1.jpg'
image2_path = 'img2.jpg'

similarity_score = compare_images(image1_path, image2_path)
print("Similarity score:", similarity_score)