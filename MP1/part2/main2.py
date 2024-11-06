import cv2
import numpy as np
import os

# Read the image
input_path = r'C:\Users\Rohit\OneDrive\Desktop\cv-code\assignment-1\part2\input\01861a.tif'
img = cv2.imread(input_path)

# Get image dimensions and split into color channels
height, width, channels = img.shape
part_height = height // 3
blue = img[0:part_height, 0:width]
green = img[part_height:2*part_height, 0:width]
red = img[2*part_height:3*part_height, 0:width]

# Crop the images to remove 12% from all sides
h, w, c = blue.shape
crop_amount = int(0.12 * w)
crop_blue = blue[crop_amount:h-crop_amount, crop_amount:w-crop_amount]
crop_green = green[crop_amount:h-crop_amount, crop_amount:w-crop_amount]
crop_red = red[crop_amount:h-crop_amount, crop_amount:w-crop_amount]

# Convert to grayscale
blueb = cv2.cvtColor(crop_blue, cv2.COLOR_BGR2GRAY)
greenb = cv2.cvtColor(crop_green, cv2.COLOR_BGR2GRAY)
redb = cv2.cvtColor(crop_red, cv2.COLOR_BGR2GRAY)

def laplace_of_gaussian(image, sigma=0.01):
    """Apply a Laplacian of Gaussian filter to an image."""
    # First, apply a Gaussian blur
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
    # Then apply the Laplacian filter
    log_image = cv2.Laplacian(blurred, cv2.CV_64F)
    return log_image

def findshift(image1og, image2og):
    """Find the shift between two images."""
    image1 = laplace_of_gaussian(image1og)
    image2 = laplace_of_gaussian(image2og)

    fimage1 = np.fft.fft2(image1)
    fimage2 = np.fft.fft2(image2)
    fcimage2 = np.conjugate(fimage2)

    fc21 = fimage1 * fcimage2
    ifc = np.abs(np.fft.ifft2(fc21))

    max_index = np.unravel_index(np.argmax(ifc), ifc.shape)
    shift = (max_index[0], max_index[1])
    aligned_image2 = np.roll(image2og, shift, axis=(0, 1))
    
    return shift, aligned_image2, fc21

# Get shifts between blue and green, and blue and red
sg, ab, fc21g = findshift(blueb, greenb)
sr, ar, fc21r = findshift(blueb, redb)

# Print the offset values
print(f"Offset between Blue and Green: {sg}")
print(f"Offset between Blue and Red: {sr}")

# Merge aligned images
imout = cv2.merge((blueb, ab, ar))

# Extract the original filename without extension
original_filename = os.path.splitext(os.path.basename(input_path))[0]

# Specify the directory path where you want to save the image
output_directory = r'C:\Users\Rohit\OneDrive\Desktop\cv-code\assignment-1\part2\results'
output_filename = f'Aligned_image_{original_filename}.jpg'
output_path = os.path.join(output_directory, output_filename)

# Save the output image
cv2.imwrite(output_path, imout)

print(f"Output image saved at: {output_path}")


# Show the output image
# cv2.imshow('Merged Image', imout)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

