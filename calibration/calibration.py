###############################################################
# 
# Refences:
# https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
# https://stackoverflow.com/questions/34316306/opencv-fisheye-calibration-cuts-too-much-of-the-resulting-image
#
###############################################################

import cv2
#assert cv2.__version__[0] == '4', 'The fisheye module requires opencv version >= 3.0.0'
import numpy as np
import glob
import matplotlib.pyplot as plt

'''
标定
'''

#########################################################
#
#   设置棋盘格的信息，生成物点
#
#########################################################

CHECKERBOARD = (6, 9)
objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
_img_shape = None

#########################################################
#
#   设置标定的一些参数
#
#########################################################

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW

#########################################################
#
#   对于每一张图像，检测棋盘格角点
#
#########################################################

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

# images = glob.glob('./calibration/1/*.jpg') + glob.glob('./calibration/2/*.jpg')
images = glob.glob('./calibration/intrinsic/SVGA/*.jpg')
print('num of images:', len(images))

for fname in images:

    img = cv2.imread(fname)
    if _img_shape == None:
        _img_shape = img.shape[:2]
    else:
        assert _img_shape == img.shape[:2], "All images must share the same size."
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_FAST_CHECK + cv2.CALIB_CB_NORMALIZE_IMAGE)
    
    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)

        # 可视化检测到的角点
        grid_corner = corners[:, 0, :]
        for i in range(grid_corner.shape[0]):
            img = cv2.circle(img, (int(grid_corner[i, 0]), int(grid_corner[i, 1])), radius=3, color=(0, 0, 255), thickness=1)
        #plt.imshow(img[:, :, [2,1,0]])
        #plt.show()

        imgpoints.append(corners)

#########################################################
#
#   初始化 K D R t 等参数，通过标定得到这些值
#
#########################################################

N_OK = len(objpoints)
K = np.zeros((3, 3))
D = np.zeros((4, 1))
rvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]
tvecs = [np.zeros((1, 1, 3), dtype=np.float32) for i in range(N_OK)]
rms, _, _, _, _ = \
    cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        gray.shape[::-1],
        K,
        D,
        rvecs,
        tvecs,
        calibration_flags,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
    )

print("Found " + str(N_OK) + " valid images for calibration")
print("DIM=" + str(_img_shape[::-1]))
print("rms=" + str(rms))
print("K=np.array(" + str(K.tolist()) + ")")
print("D=np.array(" + str(D.tolist()) + ")")
