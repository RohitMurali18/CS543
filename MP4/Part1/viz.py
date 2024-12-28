import os
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

def visualize_images_grid(folder_path, cols=8):
    image_files = [f for f in sorted(os.listdir(folder_path)) if f.endswith('.pgm')]
    if len(image_files) == 0:
        print("No .pgm files")
        return
    rows = (len(image_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, rows * 2))
    axes = axes.flatten()
    for idx, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        image = Image.open(image_path)
        image_array = np.array(image)
        axes[idx].imshow(image_array, cmap='gray')
        axes[idx].set_title(image_file, fontsize=8)
        axes[idx].axis('off')
    for ax in axes[len(image_files):]:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

folder_path = r"C:\Users\Rohit\OneDrive\Desktop\CS543\MP4\Part1\croppedyale\yaleB01"  
visualize_images_grid(folder_path)
