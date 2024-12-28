import cv2
import glob
import matplotlib.pyplot as plt
import numpy as np
from numpy.linalg import svd, eigh, cholesky, norm, inv
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

def QMat(mot):
    sz = mot.shape[0] // 2
    L = np.zeros((sz * 2, 6))

    for i in range(sz):
        r1 = 2 * i
        r2 = 2 * i + 1
        n1, n2 = mot[r1], mot[r2]

        L[r1] = [
            n1[0] * n2[0],
            n1[0] * n2[1] + n1[1] * n2[0],
            n1[0] * n2[2] + n1[2] * n2[0],
            n1[1] * n2[1],
            n1[1] * n2[2] + n1[2] * n2[1],
            n1[2] * n2[2],
        ]

        L[r2] = [
            n1[0] ** 2 - n2[0] ** 2,
            2 * (n1[0] * n1[1] - n2[0] * n2[1]),
            2 * (n1[0] * n1[2] - n2[0] * n2[2]),
            n1[1] ** 2 - n2[1] ** 2,
            2 * (n1[1] * n1[2] - n2[1] * n2[2]),
            n1[2] ** 2 - n2[2] ** 2,
        ]

    _, _, Vt = svd(L)

    Q = np.array([[Vt[-1][0], Vt[-1][1], Vt[-1][2]],
                  [Vt[-1][1], Vt[-1][3], Vt[-1][4]],
                  [Vt[-1][2], Vt[-1][4], Vt[-1][5]]])

    eig_vals, _ = eigh(Q)
    if np.any(eig_vals <= 0):
        return None
    else:
        return cholesky(Q)

def factor(CM):
    U, s, Vt = svd(CM)
    DR = np.dot(U[:, :3], np.dot(np.diag(s[:3]), (Vt.T[:, :3]).T))
    st = np.dot(np.sqrt(np.diag(s[:3])), (Vt.T[:, :3]).T)
    mot = np.dot(U[:, :3], np.sqrt(np.diag(s[:3])))
    err = norm(CM - DR)
    print(f"Reconstruction Error = {err}")
    return mot, st

def transform(mot, st, Q):
    return np.dot(mot, Q), np.dot(inv(Q), st)

def visualize(st):
    pts = st.T
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c='b', marker='o')
    ax.set_title('3D Structure')
    plt.show()

def calc_res(CM, mot, st):
    res = []
    sz = mot.shape[0] // 2
    pts = np.dot(mot, st)
    
    for i in range(sz):
        proj = pts[2 * i:2 * i + 2, :]
        orig = CM[2 * i:2 * i + 2, :]
        res.append(np.sum((orig - proj) ** 2))

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, sz + 1), res)
    plt.title('Reprojection Error')
    plt.grid(True)
    plt.xlabel('Frame No')
    plt.ylabel('Residual (px^2)')
    plt.show()

    print(f"Total Residual: {np.sum(res):.2f} px^2")

def viz_points(mot, st, frames=[1, 50, 101]):
    plt.figure(figsize=(15, 5))
    for idx, frm in enumerate(frames, 1):
        img_path = f'C:\\Users\\Rohit\\OneDrive\\Desktop\\CS543\\MP5\\part2\\images\\frame{frm:08d}.jpg'
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        proj_pts = mot[2 * (frm - 1):2 * (frm - 1) + 2, :] @ st
        proj_pts = proj_pts + means[frm-1]

        plt.subplot(1, 3, idx)
        plt.imshow(img)
        plt.scatter(proj_pts[0], proj_pts[1], c='r', marker='.', label='Proj')
        plt.title(f'Frame {frm}')
        plt.axis('off')

    plt.tight_layout()
    plt.show()

# Main
path = r'C:\Users\Rohit\OneDrive\Desktop\CS543\MP5\part2\images\measurement_matrix.txt'
mat = np.loadtxt(path)
print(f"Matrix Shape: {mat.shape}")

M = mat.shape[0] // 2 
N = mat.shape[1]   
CMat = mat - np.mean(mat, axis=1, keepdims=True)
means =  np.mean(mat, axis=1, keepdims=True)
print(f"Views (M): {M}, Points (N): {N}, CMat Shape: {CMat.shape}")

mot, st = factor(CMat)
print(f"Motion Shape: {mot.shape}, Structure Shape: {st.shape}")

Q = QMat(mot)
if Q is not None:
    print("Q Matrix:")
    print(Q)
else:
    print("No Q Matrix Found.")

mot, st = transform(mot, st, Q)
visualize(st)
viz_points(mot, st)
calc_res(CMat, mot, st)
