import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from corners import get_sudoku_corners
from transform import warp_image, backproject
from ocr import read_number, read_number2
from solve_puzzle import solve_sudoku, is_solvable


def plot_img(img, name, gray = False):
    plt.figure(name)
    
    if (gray):
        plt.imshow(img, cmap = 'gray')
    else:
        plt.imshow(img)


def solver(img):
    #plot_img(img, "img")
    
    width = 1000
    height = 1000
    dim = (width, height)
    resized_c = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    gray = cv2.cvtColor(resized_c, cv2.COLOR_BGR2GRAY)
    plot_img(gray, "gray", True)
    plt.show()

    # plot_img(resized, "gray", True)

    c = get_sudoku_corners(gray)
    
    if (not(c is None)):

        # ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # width = int(gray.shape[1] * 10)
        # height = int(gray.shape[0] * 10)
        # dim = (width, height)
        # resized = cv2.resize(gray, dim, interpolation = cv2.INTER_CUBIC)
        # ret, thr = cv2.threshold(resized, 180, 255, cv2.THRESH_OTSU)
        # kernel = np.ones((3,3),np.uint8)
        # thr = cv2.erode(thr,kernel,iterations = 1)

        # ret, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        #plot_img(thr, "thr", True)
        #plt.show()
        wc = warp_image(img, c[0], c[1], c[2], c[3])
        # wb = warp_image(thr, c[0], c[1], c[2], c[3])
        w = warp_image(gray, c[0], c[1], c[2], c[3])

        # width = int(w.shape[1] * 10)
        # height = int(w.shape[0] * 10)

        # width = 1000
        # height = 1000
        # dim = (width, height)
        # resized = cv2.resize(w, dim, interpolation = cv2.INTER_CUBIC)
        # resized_c = cv2.resize(wc, dim, interpolation = cv2.INTER_CUBIC)
        
        # ret, thr = cv2.threshold(resized, 180, 255, cv2.THRESH_OTSU)
        thr = cv2.adaptiveThreshold(w, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 27, 11)
        kernel = np.ones((3,3),np.uint8)
        thr = cv2.erode(thr,kernel,iterations = 1)

        wb = thr
        # w = resized
        # wc = resized_c


        # plot_img(wb, "before", True)
        # plt.show()

        inv = 255 - wb
        horizontal_img = inv.copy()
        vertical_img = inv.copy()

        # remove long/wide structures i.e. lines
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (100,1))
        horizontal_img = cv2.erode(horizontal_img, kernel, iterations=3)
        horizontal_img = cv2.dilate(horizontal_img, kernel, iterations=3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,100))
        vertical_img = cv2.erode(vertical_img, kernel, iterations=3)
        vertical_img = cv2.dilate(vertical_img, kernel, iterations=3)

        mask_img = horizontal_img + vertical_img
        wb = np.bitwise_or(wb, mask_img)

        # remove little noise
        kernel = np.ones((3,3),np.uint8)
        wb = cv2.dilate(wb,kernel,iterations = 1)


        # # plot_img(wc, "wc")
        # plot_img(wb, "after", True)
        # plt.show()
        # # print(read_number2(w))
        
        puzzle = None
        he = 0
        wi = 0
        max_rot = 0
        max_nums = 0
        rotated = np.copy(wb)
        
        for rot in range(4):
            curr_puzz = np.zeros((9, 9), np.int8)
            
            che, cwi = rotated.shape
            
            che = che // 9
            cwi = cwi // 9
            
            for i in range(9):
                for j in range(9):
                    cell = rotated[i * che : (i + 1) * che, j * cwi : (j + 1) * cwi]
                    
                    curr_puzz[i, j] = read_number(cell)
                
                    # plt.subplot(9, 9, 9*i + j + 1)
                    # plt.imshow(cell, cmap = 'gray')
            
            nums = curr_puzz[np.where(curr_puzz != 0)].size
            
            if (nums > max_nums):
                puzzle = np.copy(curr_puzz)
                he = che
                wi = cwi
                max_rot = rot * 90
                max_nums = nums
            
            rotated = cv2.rotate(rotated, cv2.ROTATE_90_CLOCKWISE)
        
        solved = np.copy(puzzle)
        
        print(puzzle)
        
        if (is_solvable(solved)):
            solve_sudoku(solved)
            
            if (max_rot == 90):
                wc = cv2.rotate(wc, cv2.ROTATE_90_CLOCKWISE)
            elif (max_rot == 180):
                wc = cv2.rotate(wc, cv2.ROTATE_180)
            elif (max_rot == 270):
                wc = cv2.rotate(wc, cv2.ROTATE_90_COUNTERCLOCKWISE)
            
            for i in range(9):
                for j in range(9):
                    if (puzzle[i, j] == 0):
                        wc = cv2.putText(wc, str(solved[i, j]), (int((j + .25)*wi), int((i + .75)*he)), cv2.FONT_HERSHEY_SIMPLEX, he / 40, (0, 0, 0), (he // 40) + 1)
            
            if (max_rot == 90):
                wc = cv2.rotate(wc, cv2.ROTATE_90_COUNTERCLOCKWISE)
            elif (max_rot == 180):
                wc = cv2.rotate(wc, cv2.ROTATE_180)
            elif (max_rot == 270):
                wc = cv2.rotate(wc, cv2.ROTATE_90_CLOCKWISE)
            
            img = backproject(wc, img, c)
        else:
            print('Puzzle not solvable.')
    else:
        print('Puzzle not detected.')
    
    #plot_img(img, "img")

    # plt.show()
    
    return img


def main():
    if len(sys.argv) != 2:
        raise ValueError('Wrong arguments')
    img_filename = sys.argv[1]
    
    img = cv2.imread(img_filename)
    #img = cv2.rotate(img, cv2.ROTATE_180)
    solver(img)


if __name__ == "__main__":
    main()

