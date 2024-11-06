import cv2
import numpy as np
import os
import time
import csv

def laplace_of_gaussian(image, sigma=0.01):
    """Apply a Laplacian of Gaussian filter to an image."""
    blurred = cv2.GaussianBlur(image, (5, 5), sigma)
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
    ifc_image = cv2.normalize(ifc_image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return ifc_image

# Define base directory
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define input and output directories
input_dir = os.path.join(base_dir, "input")
output_dir = os.path.join(base_dir, "results")

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

stamp = []

for filename in os.listdir(input_dir):
    if filename.lower().endswith((".jpg", ".jpeg", ".png",".tif",".TIF")):
        image_path = os.path.join(input_dir, filename)
        start_time = time.time() 
        
        try:
            img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                raise ValueError(f"Failed to read image: {filename}")

            height, width = img.shape
            part_height = height // 3
            blue = img[0:part_height, 0:width]
            green = img[part_height:2*part_height, 0:width]
            red = img[2*part_height:3*part_height, 0:width]

            h, w = blue.shape
            crop_amount = int(0.12 * w)
            crop_blue = blue[crop_amount:h-crop_amount, crop_amount:w-crop_amount]
            crop_green = green[crop_amount:h-crop_amount, crop_amount:w-crop_amount]
            crop_red = red[crop_amount:h-crop_amount, crop_amount:w-crop_amount]

            sg, ab, fc21g = findshift(crop_blue, crop_green)
            sr, ar, fc21r = findshift(crop_blue, crop_red)

            print(f"Offset between Blue and Green: {sg}")
            print(f"Offset between Blue and Red: {sr}")

            imout = cv2.merge((crop_blue, ab, ar))

            bg_pre = visualize_inverse_fourier(np.fft.fft2(crop_blue) * np.conjugate(np.fft.fft2(crop_green)), "IFC Blue to Green - Unprocessed")
            br_pre = visualize_inverse_fourier(np.fft.fft2(crop_blue) * np.conjugate(np.fft.fft2(crop_red)), "IFC Blue to Red - Unprocessed")

            bg = visualize_inverse_fourier(fc21g, "IFC Blue to Green - Processed")
            br = visualize_inverse_fourier(fc21r, "IFC Blue to Red - Processed")

            end_time = time.time() 
            total_time = end_time - start_time
            
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0])
            os.makedirs(output_path, exist_ok=True)
            
            cv2.imwrite(os.path.join(output_path, "bg.png"), bg)
            cv2.imwrite(os.path.join(output_path, "br.png"), br)
            cv2.imwrite(os.path.join(output_path, "bg_pre.png"), bg_pre)
            cv2.imwrite(os.path.join(output_path, "br_pre.png"), br_pre)
            cv2.imwrite(os.path.join(output_path, "Aligned_result.png"), imout)
            
            stamp.append([filename, sg, sr, total_time])
            
        except Exception as e:
            print(f"Error processing {filename}: {str(e)}")

csv_file = os.path.join(output_dir, 'data_part1.csv')
try:
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Filename', 'g_channel_shift (x,y)', 'r_channel_shift (x,y)', 'Total_time'])
        writer.writerows(stamp)
    print(f"CSV file saved successfully: {csv_file}")
except Exception as e:
    print(f"Error saving CSV file: {str(e)}")