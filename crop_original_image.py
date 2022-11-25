import cv2
import numpy as np


def crop_original_image(img: np.ndarray):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    white_rectangle = img_gray * 255

    white_rectangle = np.where(white_rectangle > 0.5, 255, 0)

    white_rectangle = white_rectangle.astype(np.uint8)

    contours, _ = cv2.findContours(white_rectangle, cv2.CHAIN_APPROX_SIMPLE, cv2.RETR_TREE)

    original_pic_contour = sorted(contours, key=lambda c: cv2.contourArea(c), reverse=True)[0]

    x, y, w, h = cv2.boundingRect(original_pic_contour)

    return x, y, w, h
