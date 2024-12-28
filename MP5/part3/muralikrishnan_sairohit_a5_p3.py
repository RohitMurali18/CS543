import numpy as np
import cv2
import time

def compute_cost(left_window, right_window, method):
    # NCC uses normalized cross-correlation; lower cost = better match.
    if method == "SSD":
        difference = left_window - right_window
        cost = np.sum(difference) ** 2
        return cost
    elif method == "SAD":
        cost = np.linalg.norm(left_window - right_window, ord=1)
        return cost
    elif method == "NCC":
        left_mean = np.mean(left_window)
        right_mean = np.mean(right_window)
        left_centered = left_window - left_mean
        right_centered = right_window - right_mean
        numerator = np.sum(left_centered * right_centered)
        denominator = np.sqrt(np.dot(left_centered.ravel(), left_centered.ravel()) * 
                              np.dot(right_centered.ravel(), right_centered.ravel()))
        cost = -numerator / (denominator if denominator != 0 else 1e-10)
        return cost
    else:
        raise ValueError(f"Invalid method '{method}'. Choose 'SSD', 'SAD', or 'NCC'.")


def stereo_matching(left_img, right_img, window_size=20, max_disparity=64, cost_function="SSD"):
    # Computes a disparity map by comparing patches between the left and right images.
    if left_img.ndim == 3:
        left_gray = cv2.cvtColor(left_img, cv2.COLOR_BGR2GRAY)
        right_gray = cv2.cvtColor(right_img, cv2.COLOR_BGR2GRAY)
    else:
        left_gray = left_img
        right_gray = right_img

    height, width = left_gray.shape
    hw = window_size // 2
    left_padded = cv2.copyMakeBorder(left_gray, hw, hw, hw, hw, cv2.BORDER_CONSTANT, value=0)
    right_padded = cv2.copyMakeBorder(right_gray, hw, hw, hw, hw, cv2.BORDER_CONSTANT, value=0)

    disparity_map = np.zeros((height, width), dtype=np.float32)

    # Efficient extraction of row-wise patches using sliding_window_view.
    for y in range(hw, height + hw):
        left_row_windows = np.lib.stride_tricks.sliding_window_view(left_padded[y - hw:y + hw + 1, :], (2 * hw + 1, 2 * hw + 1))
        for x in range(hw, width + hw):
            left_window = left_row_windows[0, x - hw]
            best_disparity = 0
            best_cost = float('inf')
            for d in range(max_disparity):
                if x - d < hw:
                    break
                right_window = right_padded[y - hw:y + hw + 1, (x - d) - hw:(x - d) + hw + 1]
                cost = compute_cost(left_window, right_window, cost_function)
                if cost < best_cost:
                    best_cost = cost
                    best_disparity = d
            disparity_map[y - hw, x - hw] = best_disparity

    disparity_map = (disparity_map / disparity_map.max() * 255).astype(np.uint8)
    return disparity_map

def process_disparity(left_img, right_img, window_size, max_disparity, cost_function):
    # Saves computed disparity map for the given cost function.
    start_time = time.time()
    disparity_map = stereo_matching(left_img, right_img, window_size=window_size, max_disparity=max_disparity, cost_function=cost_function)
    end_time = time.time()
    cv2.imwrite('output_image.png', disparity_map)
    print(f"Time for {cost_function}: {end_time - start_time}")

if __name__ == "__main__":
    # Runs disparity computation on sample stereo images.
    left_image = cv2.imread(r'C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part3\moebius1.png')
    right_image = cv2.imread(r'C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part3\moebius2.png')

    matching_costs = ["SSD"]
    window_size = 30
    max_disparity = 30

    for cost in matching_costs:
        process_disparity(left_image, right_image, window_size, max_disparity, cost)
