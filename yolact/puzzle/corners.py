import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt


def plot_img(img, name, gray = False):
    plt.figure(name)
    
    if (gray):
        plt.imshow(img, cmap = 'gray')
    else:
        plt.imshow(img)


def get_sudoku_corners(img):
    #plot_img(img, "img", True)
    
    blur = cv2.GaussianBlur(img, (11, 11), 0)
    #plot_img(blur, "blur", True)
    
    ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    mor = cv2.morphologyEx(blur, cv2.MORPH_CLOSE, ker)
    #plot_img(close, "close", True)
    
    div = np.float32(blur) / mor
    #plot_img(div, "div", True)
    
    norm = np.uint8(cv2.normalize(div, div, 0, 255, cv2.NORM_MINMAX))
    #plot_img(res, "res", True)
    
    # gnorm = cv2.cvtColor(norm, cv2.COLOR_BGR2GRAY)
    gnorm = norm
    #plot_img(res2, "res2", True)

    thresh = cv2.adaptiveThreshold(gnorm, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 19, 2)
    #plot_img(thresh, "thresh", True)
    
    try:
        contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    except:
        im, contours, hier = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    #imgc = np.ones(img.shape)
    #conts = cv2.drawContours(imgc, contours, -1, (0, 255, 0), 3)
    
    #plot_img(conts, "conts", True)
    #plt.show()

    biggest = None
    max_area = 0
    
    for c in contours:
        area = cv2.contourArea(c)
        
        if area > 100:
            per = cv2.arcLength(c, True)
            app = cv2.approxPolyDP(c, .02 * per, True)
            
            if area > max_area and len(app) == 4:
                biggest = app
                max_area = area
    
    M = cv2.moments(biggest)
    cx = int(M['m10'] / M['m00'])
    cy = int(M['m01'] / M['m00'])
    
    tl = None
    tr = None
    br = None
    bl = None
    
    for a in range(4):
        dx = biggest[a][0][0] - cx
        dy = biggest[a][0][1] - cy
        
        if dx < 0 and dy < 0:
            tl = (biggest[a][0][0], biggest[a][0][1])
        elif dx > 0 and dy < 0:
            tr = (biggest[a][0][0], biggest[a][0][1])
        elif dx > 0 and dy > 0:
            br = (biggest[a][0][0], biggest[a][0][1])
        elif dx < 0 and dy > 0:
            bl = (biggest[a][0][0], biggest[a][0][1])
    
    if (not tl or not tr or not br or not bl):
        return None
    
    return np.array([tl, tr, br, bl])


def main():
    if len(sys.argv) != 2:
        raise ValueError('Wrong arguments')
    img_filename = sys.argv[1]
    
    img = cv2.imread(img_filename)

    corners = get_sudoku_corners(img)

    print(corners)

    plt.show()
    
    print("done")


if __name__ == "__main__":
    main()

