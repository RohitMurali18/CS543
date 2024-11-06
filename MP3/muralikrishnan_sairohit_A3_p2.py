import numpy as np
import skimage
import cv2
import matplotlib.pyplot as plt
from scipy.spatial import distance
import scipy 
import skimage.transform
from pylab import *
from scipy import signal
from scipy import *
import numpy as np
from PIL import Image


def sift(image):
    sift = cv2.SIFT_create()
    
    # Detecting keypoints and compute descriptors
    keypoints, descriptors = sift.detectAndCompute(image, None)
    
    # Drawing keypoints on the image
    image_with_keypoints = cv2.drawKeypoints(image, keypoints, None, color=(0, 0, 255), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    return image_with_keypoints, keypoints, descriptors


def display_keypoints_image(image):

    # Display the image in a window
    cv2.imshow("Keypoints Image",image)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def display_matched_image(img1,kp1,img2,kp2,matches):
    matched_img = cv2.drawMatches(img1, kp1, img2, kp2, matches,None,matchColor=(0, 0, 255), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imshow('Matches', matched_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def calculate_euclidean_distance(k1,k2,d1,d2,threshold):
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
    matches = bf.match(d1,d2)
    filtered_matches = [match for match in matches if match.distance < threshold]
    filtered_matches = sorted(filtered_matches, key=lambda x: x.distance)
    filtered_matches = np.array(filtered_matches)
    filtered_matches_coords = []
    for match in filtered_matches:
        # Get the keypoint coordinates using queryIdx and trainIdx
        coord1 = k1[match.queryIdx].pt  # Coordinates from the first image's keypoints
        coord2 = k2[match.trainIdx].pt  # Coordinates from the second image's keypoints
        
        # Append the coordinates as a pair in the list
        filtered_matches_coords.append([coord1[0], coord1[1], coord2[0], coord2[1]])
    
    # Convert to a numpy array for easier manipulation if needed
    filtered_matches_coords = np.array(filtered_matches_coords)
    
    return filtered_matches,filtered_matches_coords

    return filtered_matches

def ransac(img1, img2, matches, threshold_ransac):
    iteration_times = 1000
    inliners = 0
    max_inliners = 0

    for iter in range(0, iteration_times):
        subset_idx = np.random.choice(matches.shape[0], size=4, replace=False)
        subset = matches[subset_idx]

        H = find_H(subset)

        # The Homography matrix should be a full rank
        if np.linalg.matrix_rank(H) >= 3:
            errors = find_errors(matches, H)
        idx = np.where(errors < threshold_ransac)[0]
        inliers_points = matches[idx]

        # find the best number of inliners 
        inliners = len(inliers_points)
        if inliners >= max_inliners:
            best_inliners = inliers_points.copy()
            max_inliners = inliners
            best_H = H.copy()
            avg_residual = sum(find_errors(matches[idx], H)) / inliners
    print("Total number of inliners: " + str(max_inliners) + " average residual: " + str(avg_residual))
    #show_inlier_matches(img1, img2, which_inliners)
    return best_H,best_inliners

def find_H(subset):
        M = []
        for i in range(subset.shape[0]):
            point1 = np.append(subset[i][0:2], 1)
            point2 = np.append(subset[i][2:4], 1)
            row1 = [0, 0, 0, point1[0], point1[1], point1[2], -point2[1]*point1[0], -point2[1]*point1[1], -point2[1]*point1[2]]
            row2 = [point1[0], point1[1], point1[2], 0, 0, 0, -point2[0]*point1[0], -point2[0]*point1[1], -point2[0]*point1[2]]
            M.append(row1)
            M.append(row2)

        M = np.array(M)
        U, s, V = np.linalg.svd(M)
        H = V[len(V)-1].reshape(3, 3)

        # normalizing the matrix
        if H[2, 2] != 0:
            H = H / H[2, 2]
        return H

def find_errors(matches, H):
    num_pairs = len(matches)
    matchingpoints1 = np.concatenate((matches[:, 0:2], np.ones((1, num_pairs)).T), axis=1)
    matchingpoints2 = matches[:, 2:4]
    transformed_p1 = np.zeros((num_pairs, 2))
    for i in range(num_pairs):
        transformed_p1[i] = (np.matmul(H, matchingpoints1[i]) / np.matmul(H,matchingpoints1[i])[-1])[0:2]

    # Finding the error for each matching pair
    errors = np.linalg.norm(matchingpoints2 - transformed_p1, axis=1) ** 2
    return errors


def plot_inlier_matches(ax, img1, img2, inliers):
    """
    Plot the matches between two images according to the matched keypoints
    :param ax: plot handle
    :param img1: left image
    :param img2: right image
    :inliers: x,y in the first image and x,y in the second image (Nx4)
    """
    res = np.hstack([img1, img2])
    ax.set_aspect('equal')
    ax.imshow(res, cmap='gray')
    
    ax.plot(inliers[:,0], inliers[:,1], '+r')
    ax.plot(inliers[:,2] + img1.shape[1], inliers[:,3], '+r')
    ax.plot([inliers[:,0], inliers[:,2] + img1.shape[1]],
            [inliers[:,1], inliers[:,3]], 'r', linewidth=0.4)
    ax.axis('off')

    
def warp_images(right, left, H):
    transform = skimage.transform.ProjectiveTransform(H)
    warp = skimage.transform.warp

    r, c = right.shape[:2]
    cornerpoints = np.array([[0, 0], [0, r], [c, 0], [c, r]])

    # Warping the left image corners to their new points
    warped_corners = transform(cornerpoints)

    all_corners = np.vstack((warped_corners, cornerpoints))

    corner_min = np.min(all_corners, axis=0)
    corner_max = np.max(all_corners, axis=0)

    output_shape = np.ceil((corner_max - corner_min)[::-1])

    # Offset transformation to align both images
    offset = skimage.transform.SimilarityTransform(translation=-corner_min)

    # Right image stays in its original coordinates
    right_warped = warp(right, offset.inverse, output_shape=output_shape, cval=0)

    # Left image is warped based on the transformation and offset
    left_warped = warp(left, (transform + offset).inverse, output_shape=output_shape, cval=0)

    # Create masks to identify where each image contributes to the final image
    right_mask = (right_warped != 0).astype(int)
    left_mask = (left_warped != 0).astype(int)

    # Calculate the overlap and ensure no division by zero
    overlap = right_mask + left_mask
    overlap = np.where(overlap < 1, 1, overlap)

    # Merge images by averaging pixel values in overlapping regions
    merged = (right_warped + left_warped) / overlap

    # Convert the merged image to RGB format
    stitched_img = np.array((255 * merged).astype('uint8'))
    stitched_img = Image.fromarray(stitched_img, mode='RGB')
    stitched_img = np.asarray(stitched_img)

    return stitched_img


def main(Image1,Image2,Image1C,Image2C):
    threshold = 1500
    threshold_ransac = 5
    image_with_keypoints1, keypoints1, descriptors1 = sift(Image1)
    image_with_keypoints2, keypoints2, descriptors2 = sift(Image2)
    #display_keypoints_image(image_with_keypoints1)
    matches,matches_coords = calculate_euclidean_distance(keypoints1,keypoints2,descriptors1,descriptors2,threshold)
    #display_matched_image(Image1,keypoints1,Image2,keypoints2,matches)
    homography_matrix,inliers = ransac(Image1,Image2,matches_coords,threshold_ransac)
    print(homography_matrix)
    fig, ax = plt.subplots(figsize=(20, 10))
    #plot_inlier_matches(ax, Image1, Image2, inliers)
    fig.savefig('inlier_matches.png', dpi=300, bbox_inches='tight')
    plt.show()
    stitched_image = warp_images(Image2C,Image1C,homography_matrix)
    # cv2.imshow("Stitched image",stitched_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    cv2.imwrite("stitched_image-2.jpg", stitched_image)
    return stitched_image




Im1 = cv2.imread(r"C:\Users\Rohit\OneDrive\Desktop\cv-code\MP3\Images\Input\part2\pier1.JPG")
Im2 = cv2.imread(r"C:\Users\Rohit\OneDrive\Desktop\cv-code\MP3\Images\Input\part2\pier2.JPG")
Im3 = cv2.imread(r"C:\Users\Rohit\OneDrive\Desktop\cv-code\MP3\Images\Input\part2\pier3.JPG")
Image1c = cv2.normalize(Im1, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
Image2c = cv2.normalize(Im2, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
Image3c = cv2.normalize(Im3, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
Image1 = cv2.cvtColor(Im1, cv2.COLOR_BGR2GRAY)
Image2 = cv2.cvtColor(Im2, cv2.COLOR_BGR2GRAY)
Image3 = cv2.cvtColor(Im3, cv2.COLOR_BGR2GRAY)

stiched_image1c = main(Image1,Image2,Image1c,Image2c)
stitched_image1c = cv2.normalize(stiched_image1c, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

stiched_image1 = cv2.cvtColor(stiched_image1c, cv2.COLOR_BGR2GRAY)
stiched_image2 = main(stiched_image1,Image3,stiched_image1c,Image3c)