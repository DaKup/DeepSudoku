import sys
import cv2
import pytesseract
import matplotlib.pyplot as plt
from skimage import io, img_as_float
import numpy as np


def read_number2(img):    
    return pytesseract.image_to_string(img)


def crop_image_to_black(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img < 255 - tol
    m,n = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]

def crop_image_to_white(img,tol=0):
    # img is 2D image data
    # tol  is tolerance
    mask = img > tol
    m,n = img.shape
    mask0,mask1 = mask.any(0),mask.any(1)
    col_start,col_end = mask0.argmax(),n-mask0[::-1].argmax()
    row_start,row_end = mask1.argmax(),m-mask1[::-1].argmax()
    return img[row_start:row_end,col_start:col_end]


def read_number(img):
    #ret2,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    
    # width = int(img.shape[1] * 10)
    # height = int(img.shape[0] * 10)
    # dim = (width, height)
    # resized = cv2.resize(img, dim, interpolation = cv2.INTER_CUBIC)
    # ret, thresh = cv2.threshold(resized, 180, 255, cv2.THRESH_OTSU)
    # kernel = np.ones((3,3),np.uint8)
    # resized = cv2.erode(thresh,kernel,iterations = 1)

    crop_x = int(img.shape[0]/8)
    crop_y = int(img.shape[1]/8)
    cropped = img[crop_x:-crop_x, crop_y:-crop_y]
    cropped = crop_image_to_white(cropped)
    # cropped = crop_image_to_white(cropped)

    border_x = 10
    border_y = 10
    white_border = np.ones(shape=(cropped.shape[0]+border_x, cropped.shape[1]+border_y)) * 255
    white_border[5:-5, 5:-5] = cropped

    # white_border = cv2.dilate(thresh,kernel,iterations = 2)
    # # white_border = cv2.bitwise_not(white_border)

    # plt.gray()
    # plt.imshow(white_border)
    # plt.show()

    # test = pytesseract.image_to_boxes(resized, config = '-l eng --psm 10 --oem 1 -c tessedit_char_whitelist=123456789')
    text = pytesseract.image_to_string(white_border, config = '-l eng --psm 6 --oem 1 -c tessedit_char_whitelist=123456789')
    
    # if text == '7':
    #     stop = 0
    # print(text)
    #     # breakpoint = 5

    # plt.gray()
    # plt.imshow(white_border)
    # plt.show()

    if len(text) != 1 or text not in "123456789":
        return 0
    
    return int(text)


def main():

    if len(sys.argv) != 2:
        raise ValueError('Wrong arguments')
    img_filename = sys.argv[1]

    # img = img_as_float(io.imread(img_filename))
    img = cv2.imread(img_filename)
    plt.figure(1)
    plt.gray()
    plt.imshow(img)
    # plt.show()

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret2,img = cv2.threshold(img,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #plt.figure(2)
    #plt.gray()
    #prep = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(prep)
    #plt.show()
    # cv2.imshow('input', img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #io.imsave("ocr.png", prep)

    # text = pytesseract.image_to_string(img_filename, lang='eng', config='psm 10 --oem 3 -c tessedit_char_whitelist=123456789')
    # text = pytesseract.image_to_string(img, lang='eng', config='psm 10 tessedit_char_whitelist=123456789')
    text = pytesseract.image_to_string(img, lang='eng', config='-l eng --psm 10 --oem 1 tessedit_char_whitelist=123456789')#psm 10 tessedit_char_whitelist=123456789')

    if len(text) != 1:
        print("no number")
    else:
        number = int(text)

    print(text)


if __name__ == "__main__":
    main()

