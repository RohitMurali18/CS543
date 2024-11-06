import cv2
import numpy as np
import os
import time  # Import time module

# Start the timer
start_time = time.time()

img_path = r'C:\Users\Rohit\OneDrive\Desktop\cv-code\assignment-1\part1\00398v.jpg'
img = cv2.imread(img_path)

height, width, channels = img.shape
part_height = height // 3
blue = img[0:part_height, 0:width]
green = img[part_height:2*part_height, 0:width]
red = img[2*part_height:3*part_height, 0:width]

h, w, c = blue.shape
crop_amount = int(0.12 * w)
crop_blue = blue[crop_amount:h - crop_amount, crop_amount:w - crop_amount]
crop_green = green[crop_amount:h - crop_amount, crop_amount:w - crop_amount]
crop_red = red[crop_amount:h - crop_amount, crop_amount:w - crop_amount]

blueb = cv2.cvtColor(crop_blue, cv2.COLOR_BGR2GRAY)
greenb = cv2.cvtColor(crop_green, cv2.COLOR_BGR2GRAY)
redb = cv2.cvtColor(crop_red, cv2.COLOR_BGR2GRAY)

# Function to compute SSD (Sum of Squared Differences)
def compute_ssd(image1, image2):
    return np.sum((image1 - image2) ** 2)

# Function to compute NCC (Normalized Cross-Correlation)
def compute_ncc(image1, image2):
    image1_norm = (image1 - np.mean(image1)) / np.std(image1)
    image2_norm = (image2 - np.mean(image2)) / np.std(image2)
    return np.sum(image1_norm * image2_norm)

# Function for exhaustive search (grid search)
def exhaustive_search(image1, image2, metric, displacement_range):
    best_score = None
    best_displacement = (0, 0)

    for dx in range(-displacement_range, displacement_range + 1):
        for dy in range(-displacement_range, displacement_range + 1):
            shifted_image2 = np.roll(np.roll(image2, dx, axis=1), dy, axis=0)
            if metric == 'ssd':
                score = compute_ssd(image1, shifted_image2)
                if best_score is None or score < best_score:  # Minimizing SSD
                    best_score = score
                    best_displacement = (dx, dy)
            elif metric == 'ncc':
                score = compute_ncc(image1, shifted_image2)
                if best_score is None or score > best_score:  # Maximizing NCC
                    best_score = score
                    best_displacement = (dx, dy)

    return best_displacement, best_score

# Pyramid search function
def pyramid_search(image1, image2, metric, displacement_range, depth):
    if depth == 0 or min(image1.shape) < 100:  # Base case for small images
        return exhaustive_search(image1, image2, metric, displacement_range)

    # Reduce image size by half
    reduced_image1 = cv2.resize(image1, (image1.shape[1] // 2, image1.shape[0] // 2))
    reduced_image2 = cv2.resize(image2, (image2.shape[1] // 2, image2.shape[0] // 2))

    # Recursively call pyramid search on reduced images
    displacement, _ = pyramid_search(reduced_image1, reduced_image2, metric, displacement_range, depth - 1)

    # Upscale the displacement to the current resolution
    displacement = (displacement[0] * 2, displacement[1] * 2)

    # Refine displacement at the current resolution
    shifted_image2 = np.roll(np.roll(image2, displacement[0], axis=1), displacement[1], axis=0)
    refined_displacement, _ = exhaustive_search(image1, shifted_image2, metric, 2)

    # Combine the displacements
    final_displacement = (displacement[0] + refined_displacement[0], displacement[1] + refined_displacement[1])

    return final_displacement, _

# Align color channels using pyramid search
def align_channels_pyramid(redb, blueb, greenb, metric, displacement_range, depth=3):
    # Align Red channel to Blue
    best_displacement_red, _ = pyramid_search(blueb, redb, metric, displacement_range, depth)
    aligned_red = np.roll(np.roll(redb, best_displacement_red[0], axis=1), best_displacement_red[1], axis=0)

    # Align Green channel to Blue
    best_displacement_green, _ = pyramid_search(blueb, greenb, metric, displacement_range, depth)
    aligned_green = np.roll(np.roll(greenb, best_displacement_green[0], axis=1), best_displacement_green[1], axis=0)

    # Reconstruct the aligned image
    aligned_image = np.stack([blueb, aligned_green, aligned_red], axis=-1)

    return aligned_image, best_displacement_red, best_displacement_green

# Example usage with SSD and pyramid search
displacement_range = 20
depth = 3  # Number of pyramid levels

# Align using SSD
aligned_image_ssd, disp_red_ssd, disp_green_ssd = align_channels_pyramid(redb, blueb, greenb, 'ssd', displacement_range, depth)
print(f"Best displacement for Red channel (SSD): {disp_red_ssd}")
print(f"Best displacement for Green channel (SSD): {disp_green_ssd}")

# Align using NCC
aligned_image_ncc, disp_red_ncc, disp_green_ncc = align_channels_pyramid(redb, blueb, greenb, 'ncc', displacement_range, depth)
print(f"Best displacement for Red channel (NCC): {disp_red_ncc}")
print(f"Best displacement for Green channel (NCC): {disp_green_ncc}")

results_dir = r'C:\Users\Rohit\OneDrive\Desktop\cv-code\assignment-1\part1\results'
if not os.path.exists(results_dir):
    os.makedirs(results_dir)
file_name = os.path.splitext(os.path.basename(img_path))[0]
result_filename = os.path.join(results_dir, f"{file_name}_aligned_ncc.jpg")  # Change "_aligned_ncc" as needed
cv2.imwrite(result_filename, aligned_image_ncc)

# End the timer
end_time = time.time()

# Calculate elapsed time
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
