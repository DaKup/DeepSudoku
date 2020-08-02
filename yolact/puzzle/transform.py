import sys
import matplotlib.pyplot as plt
from skimage import io, img_as_float
import cv2
import numpy as np

def main():
    if len(sys.argv) != 2:
        raise ValueError('Wrong arguments')
    img_filename = sys.argv[1]
    
    img = img_as_float(io.imread(img_filename))
    
    # corners = [(538, 285), (1995, 270), (2439, 1424), (73, 1463)]

    #tl = np.array([538, 285])
    #tr = np.array([1995, 270])
    #br = np.array([2439, 1424])
    #bl = np.array([73, 1463])
    
    tl = np.array([201, 17.5])
    tr = np.array([807, 26])
    br = np.array([720, 611])
    bl = np.array([24.5, 530])
    

    warped = warp_image(img, tl, tr, br, bl)

    plt.figure(1)
    plt.imshow(img)
    # plt.show()
    plt.figure(2)
    plt.imshow(warped)
    plt.show()

    io.imsave("warped.png", warped)

def warp_image(image, tl, tr, br, bl):
    
    width = max(np.linalg.norm(tl-tr), np.linalg.norm(bl-br))
    height = max(np.linalg.norm(tl-bl), np.linalg.norm(tr-br))
    size = int(max(width, height))
    width, height = size, size

    src = np.array([tl, tr, br, bl], dtype=np.float32)

    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]],
        dtype = "float32")

    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, M, (width, height))


def backproject(src, dest, pts):
    sh, sw, sc = src.shape
    dh, dw, dc = dest.shape
    
    pts_src = np.array([
        [0, 0],
        [sw - 1, 0],
        [sw - 1, sh - 1],
        [0, sh - 1]],
        dtype = "float32")
    
    pts_dest = np.array([
        [pts[0][0], pts[0][1]],
        [pts[1][0], pts[1][1]],
        [pts[2][0], pts[2][1]],
        [pts[3][0], pts[3][1]]],
        dtype = "float32")
    
    mask = np.zeros(dest.shape, dtype = np.uint8)
    cv2.fillConvexPoly(mask, np.int32(pts_dest), (255,) * dc)
    mask = cv2.bitwise_not(mask)
    
    h, _ = cv2.findHomography(pts_src, pts_dest, cv2.RANSAC, 5.0)
    
    final = cv2.bitwise_or(cv2.warpPerspective(src, h, (dw, dh)), cv2.bitwise_and(dest, mask))
    
    # warped = cv2.warpPerspective(src, h, (dh, dw))
    # original_frame = cv2.bitwise_and(dest, mask)
    # original_inner = cv2.bitwise_not(original_frame)
    # inner_original_frame = cv2.bitwise_not(original_frame)
    # inner_warped = cv2.bitwise_and(inner_original_frame, warped)
    return final


if __name__ == '__main__':
    main()

