import cv2
import numpy as np
import os  # Import os to create directories

# Correcting the paths by using raw strings
image1 = cv2.imread(r'C:\Users\Rohit\OneDrive\Desktop\cv-code\assignment-1\part3\motorcycle.jpg')
image2 = cv2.imread(r'C:\Users\Rohit\OneDrive\Desktop\cv-code\assignment-1\part3\bicycle.jpg')
# image1 = cv2.imread(r'C:\Users\Rohit\OneDrive\Desktop\cv-code\assignment-1\part3\cereal.jpg')
# image2 = cv2.imread(r'C:\Users\Rohit\OneDrive\Desktop\cv-code\assignment-1\part3\box.jpg')
# image1 = cv2.imread(r'C:\Users\Rohit\OneDrive\Desktop\cv-code\assignment-1\part3\submarine.jpg')
# image2 = cv2.imread(r'C:\Users\Rohit\OneDrive\Desktop\cv-code\assignment-1\part3\fish.jpg')


h, w, c = image1.shape

# Converting images to grayscale
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

# Converting to float32
gray1 = gray1.astype(np.float32)
gray2F = gray2.astype(np.float32)

# Applying Gaussian blur
g1 = cv2.GaussianBlur(gray1, (5, 5), 3.1)
blurred = cv2.GaussianBlur(gray2F, (5,5), 11)

# Subtracting blurred image from the original
g2 = gray2 - blurred 

# Adding the two processed images
result = g1 + g2
result1 = (result - np.min(result)) / (np.max(result) - np.min(result))

# Resize the result for saving
resized_result = cv2.resize(result1, (result1.shape[1] // 4, result1.shape[0] // 4))

# Create a results directory if it doesn't exist
results_dir = r'C:\Users\Rohit\OneDrive\Desktop\cv-code\assignment-1\part3\results'
os.makedirs(results_dir, exist_ok=True)

# Save images
cv2.imwrite(os.path.join(results_dir, 'g1.jpg'), g1)
cv2.imwrite(os.path.join(results_dir, 'g2.jpg'), g2)
cv2.imwrite(os.path.join(results_dir, 'result.jpg'), (result1 * 255).astype(np.uint8))  # Scale to 0-255 for saving
cv2.imwrite(os.path.join(results_dir, 'resized_result.jpg'), (resized_result * 255).astype(np.uint8))  # Save resized result

# Displaying the result correctly
# cv2.imshow('Result', result1)

# # Wait until a key is pressed and then close the window
# cv2.waitKey(0)
# cv2.destroyAllWindows()
