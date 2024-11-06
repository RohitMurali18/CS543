import cv2
import numpy as np
import os
import time
import csv
# Read the image

# Get image dimensions and split into color channels


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


def visualize_inverse_fourier(fc_image, title):
    """Visualize the Inverse Fourier Transform of the given frequency domain image."""
    ifc_image = np.abs(np.fft.ifft2(fc_image))
    # Normalize the image for better visualization
    ifc_image = cv2.normalize(ifc_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return ifc_image


directory = r"C:\\Users\\Rohit\\OneDrive\\Desktop\\cv-code\\assignment-1\\part2\\input"
filtertypes = ["ncc","ssd"]
stamp = []
resultdir = r"C:\\Users\\Rohit\\OneDrive\\Desktop\\cv-code\\assignment-1\\part2\\results"
os.makedirs(resultdir,exist_ok=True)
for filename in os.listdir(directory):
    # Check if the file is an image
    if filename.lower().endswith(".jpg"):
        image_path = os.path.join(directory, filename)
        start_time = time.time() 
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        

        height, width = img.shape
        part_height = height // 3
        blue = img[0:part_height, 0:width]
        green = img[part_height:2*part_height, 0:width]
        red = img[2*part_height:3*part_height, 0:width]

        # Crop the images to remove 12% from all sides
        h, w = blue.shape
        crop_amount = int(0.12 * w)
        crop_blue = blue[crop_amount:h-crop_amount, crop_amount:w-crop_amount]
        crop_green = green[crop_amount:h-crop_amount, crop_amount:w-crop_amount]
        crop_red = red[crop_amount:h-crop_amount, crop_amount:w-crop_amount]

        # Convert to grayscale
        blueb = cv2.cvtColor(crop_blue, cv2.COLOR_BGR2GRAY)
        greenb = cv2.cvtColor(crop_green, cv2.COLOR_BGR2GRAY)
        redb = cv2.cvtColor(crop_red, cv2.COLOR_BGR2GRAY)
        sg, ab, fc21g = findshift(blueb, greenb)
        sr, ar, fc21r = findshift(blueb, redb)

        # Print the offset values
        print(f"Offset between Blue and Green: {sg}")
        print(f"Offset between Blue and Red: {sr}")

        # Merge aligned images
        imout = cv2.merge((blueb, ab, ar))

        # Visualize Inverse Fourier Transform (without processing)
        bg_pre = visualize_inverse_fourier(np.fft.fft2(blueb) * np.conjugate(np.fft.fft2(greenb)), "IFC Blue to Green - Unprocessed")
        br_pre = visualize_inverse_fourier(np.fft.fft2(blueb) * np.conjugate(np.fft.fft2(redb)), "IFC Blue to Red - Unprocessed")

        # Visualize Inverse Fourier Transform (with processing)
        bg = visualize_inverse_fourier(fc21g, "IFC Blue to Green - Processed")
        br = visualize_inverse_fourier(fc21r, "IFC Blue to Red - Processed")

        end_time = time.time() 
        total_time = end_time - start_time
        output_path = os.path.join(resultdir, filename[:-4])
        stamp.append([filename,sg,sr,total_time])
        os.makedirs(output_path,exist_ok=True)
        cv2.imwrite(output_path+"//bg.png" , bg) 
        cv2.imwrite(output_path+"//br.png" , br) 
        cv2.imwrite(output_path+"//bg_pre.png" , bg_pre) 
        cv2.imwrite(output_path+"//br_pre.png" , br_pre) 
        cv2.imwrite(output_path+"//Aligned_result.png",imout)



csv_file = 'data_part1.csv'
with open(csv_file, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Filename', 'flter_method', 'g_channel_shift (x,y)', 'r_channel_shift (x,y))', 'Total_time'])  # Write header
    writer.writerows(stamp)  # Write data rows
