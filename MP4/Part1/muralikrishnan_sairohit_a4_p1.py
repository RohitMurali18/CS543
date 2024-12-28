# imports
import os
import random
import sys
import glob
import re
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image
import cv2
import time


#####################################
### Provided functions start here ###
#####################################

# Image loading and saving

def LoadFaceImages(pathname, subject_name, num_images):
    """
    Load the set of face images.  
    The routine returns
        ambimage: image illuminated under the ambient lighting
        imarray: a 3-D array of images, h x w x Nimages
        lightdirs: Nimages x 3 array of light source directions
    """

    def load_image(fname):
        return np.asarray(Image.open(fname))

    def fname_to_ang(fname):
        yale_name = os.path.basename(fname)
        return int(yale_name[12:16]), int(yale_name[17:20])

    def sph2cart(az, el, r):
        rcos_theta = r * np.cos(el)
        x = rcos_theta * np.cos(az)
        y = rcos_theta * np.sin(az)
        z = r * np.sin(el)
        return x, y, z

    ambimage = load_image(
        os.path.join(pathname, subject_name + '_P00_Ambient.pgm'))
    im_list = glob.glob(os.path.join(pathname, subject_name + '_P00A*.pgm'))
    if num_images <= len(im_list):
        im_sub_list = np.random.choice(im_list, num_images, replace=False)
    else:
        print(
            'Total available images is less than specified.\nProceeding with %d images.\n'
            % len(im_list))
        im_sub_list = im_list
    im_sub_list.sort()
    imarray = np.stack([load_image(fname) for fname in im_sub_list], axis=-1)
    Ang = np.array([fname_to_ang(fname) for fname in im_sub_list])

    x, y, z = sph2cart(Ang[:, 0] / 180.0 * np.pi, Ang[:, 1] / 180.0 * np.pi, 1)
    lightdirs = np.stack([y, z, x], axis=-1)
    return ambimage, imarray, lightdirs

def save_outputs(subject_name, albedo_image, surface_normals):
    im = Image.fromarray((albedo_image*255).astype(np.uint8))
    im.save("%s_albedo.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,0]*128+128).astype(np.uint8))
    im.save("%s_normals_x.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,1]*128+128).astype(np.uint8))
    im.save("%s_normals_y.jpg" % subject_name)
    im = Image.fromarray((surface_normals[:,:,2]*128+128).astype(np.uint8))
    im.save("%s_normals_z.jpg" % subject_name)


# Plot the height map

def set_aspect_equal_3d(ax):
    """https://stackoverflow.com/questions/13685386"""
    """Fix equal aspect bug for 3D plots."""
    xlim = ax.get_xlim3d()
    ylim = ax.get_ylim3d()
    zlim = ax.get_zlim3d()
    from numpy import mean
    xmean = mean(xlim)
    ymean = mean(ylim)
    zmean = mean(zlim)
    plot_radius = max([
        abs(lim - mean_)
        for lims, mean_ in ((xlim, xmean), (ylim, ymean), (zlim, zmean))
        for lim in lims
    ])
    ax.set_xlim3d([xmean - plot_radius, xmean + plot_radius])
    ax.set_ylim3d([ymean - plot_radius, ymean + plot_radius])
    ax.set_zlim3d([zmean - plot_radius, zmean + plot_radius])


def display_output(albedo_image, height_map):
    """
    Display the albedo and height map results.
    """
    fig = plt.figure()
    plt.imshow(albedo_image, cmap='gray')
    plt.axis('off')
    plt.title("Albedo Image")
    
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(projection='3d')
    ax.view_init(20, 20)
    X = np.arange(albedo_image.shape[0])
    Y = np.arange(albedo_image.shape[1])
    X, Y = np.meshgrid(Y, X)
    
    # Flip and normalize height map and albedo image
    H = np.flipud(np.fliplr(height_map))
    H = np.nan_to_num(H)  # Ensure no NaNs
    A = np.flipud(np.fliplr(albedo_image))
    A = A / np.max(A)  # Normalize to [0, 1]
    A = np.stack([A, A, A], axis=-1)  # Convert to RGB for facecolors
    
    ax.xaxis.set_ticks([])
    ax.xaxis.set_label_text('Z')
    ax.yaxis.set_ticks([])
    ax.yaxis.set_label_text('X')
    ax.zaxis.set_ticks([])
    ax.zaxis.set_label_text('Y')
    
    # Plot the 3D surface
    surf = ax.plot_surface(
        H, X, Y, cmap='gray', facecolors=A, linewidth=0, antialiased=False
    )
    set_aspect_equal_3d(ax)

    plt.show()


# Plot the surface normals

def plot_surface_normals(surface_normals):
    """
    surface_normals: h x w x 3 matrix.
    """
    fig = plt.figure()
    ax = plt.subplot(1, 3, 1)
    ax.axis('off')
    ax.set_title('X')
    im = ax.imshow(surface_normals[:,:,0])
    ax = plt.subplot(1, 3, 2)
    ax.axis('off')
    ax.set_title('Y')
    im = ax.imshow(surface_normals[:,:,1])
    ax = plt.subplot(1, 3, 3)
    ax.axis('off')
    ax.set_title('Z')
    im = ax.imshow(surface_normals[:,:,2])


#######################################
### Your implementation starts here ###
#######################################

def preprocess(ambimage, imarray):
    """
    preprocess the data: 
        1. subtract ambient_image from each image in imarray.
        2. make sure no pixel is less than zero.
        3. rescale values in imarray to be between 0 and 1.
    Inputs:
        ambimage: h x w
        imarray: h x w x Nimages
    Outputs:
        processed_imarray: h x w x Nimages
    """
    processed_imarray = imarray - ambimage[..., None]  

    #negative pixels to 0
    processed_imarray[processed_imarray < 0] = 0
    #normalization
    processed_imarray = processed_imarray / 255.0
   # print("Shape:",processed_imarray.shape)
    
    return processed_imarray


def photometric_stereo(imarray, light_dirs):
    """
    Inputs:
        imarray:  h x w x Nimages
        light_dirs: Nimages x 3
    Outputs:
        albedo_image: h x w
        surface_norms: h x w x 3
    """


    h, w, n = imarray.shape  
    I = imarray.reshape(-1, n).T  
    g, residula, rank, singular = np.linalg.lstsq(light_dirs, I, rcond=None)  
    # Reshape g back to (h, w, 3)
    g = g.T.reshape(h, w, 3)
    # albedo computation
    albedo_image = np.linalg.norm(g, axis=2)
    # surface normals computation
    surface_normals = g / (albedo_image[..., None] )  


    return albedo_image, surface_normals



def get_surface(surface_normals, integration_method, num_random_paths =100):
    """
    Inputs:
        surface_normals:h x w x 3
        integration_method: string in ['average', 'column', 'row', 'random']
    Outputs:
        height_map: h x w
    """
    """
    Compute the surface height map by integrating the surface normals.

    Args:
        surface_normals: numpy array of shape (h, w, 3), containing x, y, z components of normals.
        integration_method: string, one of ['row', 'column', 'average', 'random'].
        num_random_paths: number of random paths for the 'random' method.

    Returns:
        height_map: numpy array of shape (h, w), the computed height map.
    """
    img_height, img_width, _ = surface_normals.shape
    height_map = np.zeros((img_height, img_width))

    gradient_x = surface_normals[:, :, 0] / surface_normals[:, :, 2]
    gradient_y = surface_normals[:, :, 1] / surface_normals[:, :, 2]

    if integration_method == 'row':
        for col in range(img_width):
            height_map[0, col] = np.sum(gradient_x[0, :col+1])  

        for row in range(1, img_height):
            for col in range(img_width):
                height_map[row, col] = height_map[row - 1, col] + gradient_y[row, col]  

    elif integration_method == 'column':
        for row in range(img_height):
            height_map[row, 0] = np.sum(gradient_y[:row+1, 0])  

        for row in range(img_height):
            for col in range(1, img_width):
                height_map[row, col] = height_map[row, col - 1] + gradient_x[row, col]  

    elif integration_method == 'average':
        row_first_map = np.zeros_like(height_map)
        for col in range(img_width):
            row_first_map[0, col] = np.sum(gradient_x[0, :col+1])  

        for row in range(1, img_height):
            for col in range(img_width):
                row_first_map[row, col] = row_first_map[row - 1, col] + gradient_y[row, col]  

        
        column_first_map = np.zeros_like(height_map)
        for row in range(img_height):
            column_first_map[row, 0] = np.sum(gradient_y[:row+1, 0])  

        for row in range(img_height):
            for col in range(1, img_width):
                column_first_map[row, col] = column_first_map[row, col - 1] + gradient_x[row, col]  

        height_map = 0.5 * row_first_map + 0.5 * column_first_map

    elif integration_method == 'random':
        for row in range(img_height):
            for col in range(img_width):
                total_height = 0
                for _ in range(num_random_paths):
                    pivot_row = random.randint(0, row)
                    pivot_col = random.randint(0, col)

                    
                    height_a = np.sum(gradient_y[:pivot_row, 0])  
                    height_a += np.sum(gradient_x[pivot_row, :pivot_col])  

                    height_b = np.sum(gradient_y[pivot_row:row, pivot_col])  # 
                    height_b += np.sum(gradient_x[row, pivot_col:col])  

                    total_height += (height_a + height_b)

                height_map[row, col] = total_height / num_random_paths

    else:
        raise ValueError(f"Unsupported integration method: {integration_method}")

    return height_map

def measure_execution_time(surface_normals, methods, num_random_paths=100):
    execution_times = {}

    for method in methods:
        times = []
        for _ in range(5):  
            start_time = time.time()
            height_map = get_surface(surface_normals, method, num_random_paths)
            end_time = time.time()
            times.append(end_time - start_time)
        
        execution_times[method] = np.mean(times)

    return execution_times



# Main function
if __name__ == '__main__':
    root_path = r'C:\Users\Rohit\OneDrive\Desktop\CS543\MP4\Part1\croppedyale'
    subject_name = 'yaleB01'
    integration_method = 'average'
    save_flag = True

    full_path = full_path = os.path.join(root_path, subject_name)
    ambient_image, imarray, light_dirs = LoadFaceImages(full_path, subject_name,
                                                        64)

    processed_imarray = preprocess(ambient_image, imarray)
    # plt.imshow(processed_imarray[:, :, 0], cmap='gray')  
    # plt.title("First Processed Image")
    # plt.axis('off')  
    # plt.show()


    albedo_image, surface_normals = photometric_stereo(processed_imarray,
                                                     light_dirs)
    # plt.imshow(albedo_image, cmap='gray')  
    # plt.title("albedo Image")
    # plt.axis('off')  
    # plt.show()
    # plt.imshow(surface_normals, cmap='gray')  
    # plt.title("surface normals")
    # plt.axis('off')  
    # plt.show()
    

    height_map = get_surface(surface_normals, 'average')
    # methods = ['random']
    # execution_times = measure_execution_time(surface_normals, methods, num_random_paths=100)

    # print("Execution Times (in seconds):")
    # for method, exec_time in execution_times.items():
    #     print(f"{method.capitalize()}: {exec_time:.4f} seconds")

    # plt.imshow(height_map, cmap='gray')  
    # plt.title("height map")
    # plt.axis('off')  
    # plt.show()

    if save_flag:
        save_outputs(subject_name, albedo_image, surface_normals)

    plot_surface_normals(surface_normals)

    display_output(albedo_image, height_map)





