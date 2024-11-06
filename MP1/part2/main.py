import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.ndimage import gaussian_laplace
import os
import cv2

# Load image and divide into three color channels (BGR order)
def load_image(image_path):
    file_ext = os.path.splitext(image_path)[1].lower()
    # Handle .tif images as high-resolution and .jpg as low-resolution
    if file_ext in ['.tif', '.jpg']:
        print(f"Processing {file_ext} image")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Error loading image at {image_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.array(img, dtype=np.float32)  # Use float32 for large images to save memory
    else:
        raise ValueError("Unsupported image format")

    height = img.shape[0] // 3
    if height == 0:
        raise ValueError("Image is too small to split into R, G, B channels.")
        
    B = img[:height]
    G = img[height:2 * height]
    R = img[2 * height:]
    return R, G, B

# Crop the image to avoid border effects during alignment
def crop_image(img, crop_width=25):
    return img[crop_width:-crop_width, crop_width:-crop_width]

# Crop all three channels to the smallest common size
def crop_to_same_size(R, G, B):
    min_height = min(R.shape[0], G.shape[0], B.shape[0])
    min_width = min(R.shape[1], G.shape[1], B.shape[1])
    R_cropped = R[:min_height, :min_width]
    G_cropped = G[:min_height, :min_width]
    B_cropped = B[:min_height, :min_width]
    return R_cropped, G_cropped, B_cropped

# Laplacian of Gaussian for edge enhancement
def laplacian_of_gaussian(channel):
    return gaussian_laplace(channel, sigma=1.0)

# Alignment using FFT cross-correlation
def align_channels_fft(reference, target):
    # FFT based cross-correlation
    f1 = np.fft.fft2(reference)
    f2 = np.fft.fft2(target)
    cross_power_spectrum = f1 * np.conjugate(f2)
    cross_power_spectrum /= np.abs(cross_power_spectrum)
    cross_correlation = np.fft.ifft2(cross_power_spectrum)
    shift = np.unravel_index(np.argmax(np.abs(cross_correlation)), reference.shape)
    return shift, cross_correlation

# Normalize the image for display
def normalize_image(img):
    img_min = img.min()
    img_max = img.max()
    if img_max == img_min:  # Prevent division by zero
        return img  # Return the original image if max equals min
    return (img - img_min) / (img_max - img_min)

# Fourier-based alignment for the entire image
def fourier_alignment(image_path):
    start_time = time.time()
    R, G, B = load_image(image_path)
    B = crop_image(B)
    G = crop_image(G)
    R = crop_image(R)
    R, G, B = crop_to_same_size(R, G, B)

    # Apply Laplacian of Gaussian to enhance edges (preprocessing)
    R_lap = laplacian_of_gaussian(R)
    G_lap = laplacian_of_gaussian(G)
    B_lap = laplacian_of_gaussian(B)

    # Align channels using Fourier-based alignment (with preprocessing)
    G_offset, _ = align_channels_fft(B_lap, G_lap)
    R_offset, _ = align_channels_fft(B_lap, R_lap)
    print(f"G offset: {G_offset}, R offset: {R_offset}")

    # Apply the calculated offsets
    G_aligned = np.roll(np.roll(G, G_offset[0], axis=0), G_offset[1], axis=1)
    R_aligned = np.roll(np.roll(R, R_offset[0], axis=0), R_offset[1], axis=1)

    # Crop channels to the same size after alignment
    R_cropped, G_cropped, B_cropped = crop_to_same_size(R_aligned, G_aligned, B)

    # Normalize each channel before stacking to form the final image
    R_normalized = normalize_image(R_cropped)
    G_normalized = normalize_image(G_cropped)
    B_normalized = normalize_image(B_cropped)

    # Stack the aligned channels to form the final RGB image
    aligned_img = np.stack([R_normalized, G_normalized, B_normalized], axis=-1)

    end_time = time.time()
    total_time = end_time - start_time
    print(f"Total time taken for Fourier Transform: {total_time:.2f} seconds")

    # Display the aligned image
    plt.imshow(aligned_img)
    plt.title("Aligned Image (Fourier-based with Laplacian of Gaussian)")
    plt.axis('off')  # Turn off axis labels
    plt.show()

    # Visualize inverse Fourier Transforms with preprocessing
    B_to_G_FT = np.fft.fft2(B_lap) * np.conjugate(np.fft.fft2(G_lap))
    B_to_R_FT = np.fft.fft2(B_lap) * np.conjugate(np.fft.fft2(R_lap))

    B_to_G_inv = np.fft.ifft2(B_to_G_FT).real
    B_to_R_inv = np.fft.ifft2(B_to_R_FT).real

    # Normalize and plot the inverse Fourier Transforms with preprocessing
    plt.figure(figsize=(12, 6))

    # B to G Inverse Fourier Transform Visualization
    plt.subplot(1, 2, 1)
    plt.imshow(normalize_image(B_to_G_inv), cmap='gray')
    plt.title("Inverse Fourier Transform: B to G (with Preprocessing)")
    plt.colorbar()

    # B to R Inverse Fourier Transform Visualization
    plt.subplot(1, 2, 2)
    plt.imshow(normalize_image(B_to_R_inv), cmap='gray')
    plt.title("Inverse Fourier Transform: B to R (with Preprocessing)")
    plt.colorbar()

    plt.show()

    # Visualizing inverse Fourier Transforms without preprocessing
    B_to_G_FT_no_preprocessing = np.fft.fft2(B) * np.conjugate(np.fft.fft2(G))
    B_to_R_FT_no_preprocessing = np.fft.fft2(B) * np.conjugate(np.fft.fft2(R))

    B_to_G_inv_no_preprocessing = np.fft.ifft2(B_to_G_FT_no_preprocessing).real
    B_to_R_inv_no_preprocessing = np.fft.ifft2(B_to_R_FT_no_preprocessing).real

    # Normalize and plot the inverse Fourier Transforms without preprocessing
    plt.figure(figsize=(12, 6))

    # B to G Inverse Fourier Transform without Preprocessing
    plt.subplot(1, 2, 1)
    plt.imshow(normalize_image(B_to_G_inv_no_preprocessing), cmap='gray')
    plt.title("Inverse Fourier Transform: B to G (without Preprocessing)")
    plt.colorbar()

    # B to R Inverse Fourier Transform without Preprocessing
    plt.subplot(1, 2, 2)
    plt.imshow(normalize_image(B_to_R_inv_no_preprocessing), cmap='gray')
    plt.title("Inverse Fourier Transform: B to R (without Preprocessing)")
    plt.colorbar()

    plt.show()

if __name__ == '__main__':
    fourier_alignment('00125v.jpg')
    # fourier_alignment('00149v.jpg')
    # fourier_alignment('00153v.jpg')
    # fourier_alignment('00351v.jpg')
    # fourier_alignment('00398v.jpg')
    # fourier_alignment('01112v.jpg')
    # fourier_alignment('01657u.tif')
    # fourier_alignment('01047u.tif')
    # fourier_alignment('01861a.tif')
