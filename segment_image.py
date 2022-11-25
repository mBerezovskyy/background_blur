import cv2
import numpy as np
import tensorflow as tf

from crop_original_image import crop_original_image

HEIGHT, WIDTH = 640, 640


def preprocess(img_raw: np.ndarray):
    img_resized = tf.image.resize_with_pad(img_raw, 640, 640)
    img_resized = tf.expand_dims(img_resized, 0)
    return tf.cast(img_resized, tf.float32)


def segment(img: np.ndarray):
    preprocessed_img = preprocess(img)

    interpreter = tf.lite.Interpreter(model_path="xception.tflite")
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], preprocessed_img)
    interpreter.invoke()

    output_data = interpreter.get_tensor(output_details[0]['index'])

    output_data = np.squeeze(output_data)

    return np.argmax(output_data, 2) * 255


def blur_background(img: np.ndarray):
    img_preprocessed = np.squeeze(preprocess(img), 0)
    segmented_img = segment(img)
    segmented_img = segmented_img.astype(np.uint8)

    segmented_img = np.stack([segmented_img, segmented_img, segmented_img], axis=-1).reshape(
        (img_preprocessed.shape[0], img_preprocessed.shape[1], 3))

    x, y, w, h = crop_original_image(img_preprocessed)

    segmented_img = segmented_img[y:y + h, x + 2:x + w - 2]
    img_preprocessed = img_preprocessed[y:y + h, x + 2:x + w - 2]
    blured_img = np.where(segmented_img > 0.5, img_preprocessed, cv2.GaussianBlur(img_preprocessed, (13, 13), 5))

    return blured_img
