import cv2
import numpy as np

from segment_image import blur_background, preprocess


img = cv2.imread('<IMAGE_NAME>')
img_orig = np.squeeze(preprocess(img), 0)

blured_img = blur_background(img)

blured_img = blured_img.astype(np.uint8)

cv2.imshow("blured", blured_img)
cv2.imwrite('blured.jpg', blured_img)
cv2.waitKey(0)
