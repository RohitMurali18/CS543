from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
from numpy.linalg import inv, svd

# Load images
# I1 = Image.open(r"C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\library1.jpg")
# I2 = Image.open(r"C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\library2.jpg")
# I1 = Image.open(r"C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\lab1.jpg")
# I2 = Image.open(r"C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\lab2.jpg")

I1 = Image.open(r"C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\gaudi1.jpg")
I2 = Image.open(r"C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\gaudi2.jpg")

lab_matches = np.loadtxt(r"C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\lab_matches.txt")
library_matches = np.loadtxt(r"C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\library_matches.txt")
kp_pos3D = np.loadtxt(r"C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\lab_3d.txt")

P0 = np.loadtxt(r'C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\library1_camera.txt')
P1 = np.loadtxt(r'C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\library2_camera.txt')

I1 = np.array(I1.convert('RGB')).astype(float)
I2 = np.array(I2.convert('RGB')).astype(float)

matches = np.loadtxt(r"C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part1\lab_matches.txt")

I3 = np.zeros((I1.shape[0], I1.shape[1] * 2, 3), dtype=float)
I3[:, :I1.shape[1], :] = I1 / 255.0
I3[:, I1.shape[1]:, :] = I2 / 255.0

# Step 1: Compute centroids and center the points
centroid_0 = np.mean(matches[:, :2], axis=0)
centroid_1 = np.mean(matches[:, 2:], axis=0)
matches_centered_0 = matches[:, :2] - centroid_0
matches_centered_1 = matches[:, 2:] - centroid_1

# Step 2: Calculate scaling factors
scale_0 = np.sqrt(2) / np.mean(np.sqrt(matches_centered_0[:, 0]**2 + matches_centered_0[:, 1]**2))
scale_1 = np.sqrt(2) / np.mean(np.sqrt(matches_centered_1[:, 0]**2 + matches_centered_1[:, 1]**2))

# Step 3: Define normalization transforms
T0 = np.array([
    [scale_0, 0, -scale_0 * centroid_0[0]],
    [0, scale_0, -scale_0 * centroid_0[1]],
    [0, 0, 1]
])

T1 = np.array([
    [scale_1, 0, -scale_1 * centroid_1[0]],
    [0, scale_1, -scale_1 * centroid_1[1]],
    [0, 0, 1]
])

# Step 4: Normalize the points
matches_0_h = np.c_[matches[:, :2], np.ones(matches.shape[0])]
matches_1_h = np.c_[matches[:, 2:], np.ones(matches.shape[0])]
matches_normalized_0 = (T0 @ matches_0_h.T).T
matches_normalized_1 = (T1 @ matches_1_h.T).T
matches_normalized = np.c_[matches_normalized_0[:, :2], matches_normalized_1[:, :2]]

print("Original Means:", np.mean(matches, axis=0))
print("Normalized Means:", np.mean(matches_normalized, axis=0))

def fit_fundamental(matches):
    x0, y0 = matches[:, 0], matches[:, 1]
    x1, y1 = matches[:, 2], matches[:, 3]
    A = np.zeros((matches.shape[0], 9))
    for i in range(matches.shape[0]):
        A[i] = [x1[i]*x0[i], x1[i]*y0[i], x1[i],
                y1[i]*x0[i], y1[i]*y0[i], y1[i],
                x0[i], y0[i], 1]
    _, _, Vt = la.svd(A)
    F = Vt[-1].reshape(3, 3)
    U, S, Vt = la.svd(F)
    S[2] = 0
    F_rank2 = U @ np.diag(S) @ Vt
    return F_rank2

# Compute Fundamental Matrix
F_normalized = fit_fundamental(matches_normalized)
F = T1.T @ F_normalized @ T0
print("Fundamental Matrix F:")
print(F)

# Compute epipolar lines in the second image
matches_h = np.c_[matches[:, :2], np.ones((matches.shape[0], 1))]
epipolar_lines = (F @ matches_h.T).T

matches_h_second = np.c_[matches[:, 2:], np.ones((matches.shape[0], 1))]
residuals = np.sum((matches_h_second @ F) * matches_h, axis=1)
print(f"Residual Mean Squared Error: {np.mean(residuals ** 2)}")

line_magnitudes = np.sqrt(epipolar_lines[:, 0]**2 + epipolar_lines[:, 1]**2)
epipolar_lines /= line_magnitudes[:, None]

distances_to_lines = np.sum(matches_h_second * epipolar_lines, axis=1)
closest_points = matches[:, 2:4] - distances_to_lines[:, None] * epipolar_lines[:, :2]

offset_vector = np.c_[-epipolar_lines[:, 1], epipolar_lines[:, 0]]
endpoint_1 = closest_points - 10 * offset_vector
endpoint_2 = closest_points + 10 * offset_vector

fig, ax = plt.subplots(figsize=(12, 12))
ax.set_aspect('equal')
ax.imshow(np.array(I2).astype(int))
ax.plot(matches[:, 2], matches[:, 3], '+r', label="Matching Points")
for i in range(matches.shape[0]):
    ax.plot([matches[i, 2], closest_points[i, 0]],
            [matches[i, 3], closest_points[i, 1]], 'r', alpha=0.7)
for i in range(matches.shape[0]):
    ax.plot([endpoint_1[i, 0], endpoint_2[i, 0]],
            [endpoint_1[i, 1], endpoint_2[i, 1]], 'g', alpha=0.7, label="Epipolar Lines" if i == 0 else "")
ax.legend()
plt.show()

def evaluate_points(M, points_2d, points_3d):
    N = len(points_3d)
    points_3d = np.hstack((points_3d, np.ones((N, 1))))
    points_3d_proj = np.dot(M, points_3d.T).T
    u = points_3d_proj[:, 0] / points_3d_proj[:, 2]
    v = points_3d_proj[:, 1] / points_3d_proj[:, 2]
    residual = np.sum(np.hypot(u-points_2d[:, 0], v-points_2d[:, 1]))
    points_3d_proj = np.hstack((u[:, np.newaxis], v[:, np.newaxis]))
    return points_3d_proj, residual

sample_size = kp_pos3D.shape[0]
kp_homo3D = np.hstack((kp_pos3D, np.ones((sample_size, 1))))
kp_pos0 = lab_matches[:, :2]
kp_pos1 = lab_matches[:, 2:]

def construct_A(kp_2D, kp_3D_homo):
    sample_size = kp_2D.shape[0]
    A = []
    for j in range(sample_size):
        X, Y, Z, W = kp_3D_homo[j]
        u, v = kp_2D[j]
        A.append([0, 0, 0, 0, -X, -Y, -Z, -W, v*X, v*Y, v*Z, v*W])
        A.append([X, Y, Z, W, 0, 0, 0, 0, -u*X, -u*Y, -u*Z, -u*W])
    return np.array(A)

A0 = construct_A(kp_pos0, kp_homo3D)
_, _, Vt = svd(A0)
P0 = Vt[-1].reshape(3, 4)
A1 = construct_A(kp_pos1, kp_homo3D)
_, _, Vt = svd(A1)
P1 = Vt[-1].reshape(3, 4)

# Print the 3x4 projection matrices
print("Projection matrix P0:")
print(P0)
print("Projection matrix P1:")
print(P1)

points_proj0, residual0 = evaluate_points(P0, kp_pos0, kp_pos3D)
mse0 = np.mean((points_proj0 - kp_pos0) ** 2)
print("Camera 0: residual =", residual0, ", mse =", mse0)

points_proj1, residual1 = evaluate_points(P1, kp_pos1, kp_pos3D)
mse1 = np.mean((points_proj1 - kp_pos1) ** 2)
print("Camera 1: residual =", residual1, ", mse =", mse1)

def compute_camera_center(P):
    M = P[:, :3]
    c = -np.dot(inv(M), P[:, 3])
    return np.append(c, 1)

camera_center0 = compute_camera_center(P0)
camera_center1 = compute_camera_center(P1)
print("Camera Center 0:", camera_center0)
print("Camera Center 1:", camera_center1)

kp_pos0 = library_matches[:, :2]
kp_pos1 = library_matches[:, 2:]
kp_projected = np.ones((kp_pos0.shape[0], 4))
for i in range(kp_pos0.shape[0]):
    x0 = np.zeros((3, 3))
    x0[0,:] = np.array([0,-1, kp_pos0[i, 1]])
    x0[1,:] = np.array([1,0, -1*kp_pos0[i, 0]])
    x0[2,:] = np.array([-1*kp_pos0[i, 1], kp_pos0[i, 0], 0])
    xp0 = x0 @ P0

    x1 = np.zeros((3, 3))
    x1[0,:] = np.array([0,-1, kp_pos1[i, 1]])
    x1[1,:] = np.array([1,0, -1*kp_pos1[i, 0]])
    x1[2,:] = np.array([-1*kp_pos1[i, 1], kp_pos1[i, 0], 0])
    xp1 = x1 @ P1

    xp = np.concatenate((xp0, xp1), axis=0)
    U, s, Vt = la.svd(xp[:,:3], full_matrices=False)
    sinv = 1/s
    Sigma_inv = np.diag(sinv)
    X = -1 * Vt.T @ Sigma_inv @ U.T @ xp[:,3]
    kp_projected[i,:-1] = X

fig = plt.figure(figsize=(20,20))
ax = plt.subplot(221, projection='3d')
ax.scatter(kp_projected[:,0], kp_projected[:,1], kp_projected[:,2])
ax.scatter(camera_center0[0], camera_center0[1], camera_center0[2], c='r')
ax.scatter(camera_center1[0], camera_center1[1], camera_center1[2], c='r')
print(kp_projected.shape)

kp_pos0 = lab_matches[:, :2]
kp_pos1 = lab_matches[:, 2:]
kp_projected = np.zeros((kp_pos0.shape[0], 4))

def skew_symmetric_matrix(x, y):
    return np.array([
        [0, -1, y],
        [1,  0, -x],
        [-y, x, 0]
    ])

for i in range(kp_pos0.shape[0]):
    x0_skew = skew_symmetric_matrix(kp_pos0[i, 0], kp_pos0[i, 1]) @ P0
    x1_skew = skew_symmetric_matrix(kp_pos1[i, 0], kp_pos1[i, 1]) @ P1
    xp = np.vstack((x0_skew, x1_skew))
    U, s, Vt = svd(xp)
    X = Vt[-1]
    kp_projected[i, :] = X / X[-1]

fig = plt.figure(figsize=(20, 20))
ax = fig.add_subplot(221, projection='3d')
ax.scatter(kp_projected[:, 0], kp_projected[:, 1], kp_projected[:, 2])
ax.scatter(camera_center0[0], camera_center0[1], camera_center0[2], c='r', label='Camera 0')
ax.scatter(camera_center1[0], camera_center1[1], camera_center1[2], c='r', label='Camera 1')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.legend()
plt.show()

print(kp_projected.shape)
