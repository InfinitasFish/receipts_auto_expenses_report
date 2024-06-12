from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt
from math import ceil, floor
from PIL import Image
from pathlib import Path


def open_image_by_path(image_path):
    return cv2.imread(image_path, -1)


def pad_image_correctly(cv_image):
    h, w = cv_image.shape[0], cv_image.shape[1]
    result_size = ceil(max(h, w) / 32) * 32
    top, bottom = floor((result_size - h) / 2), ceil((result_size - h) / 2)
    left, right = floor((result_size - w) / 2), ceil((result_size - w) / 2)
    padded_image = cv2.copyMakeBorder(cv_image, top, bottom, left, right, cv2.BORDER_REPLICATE)

    return padded_image


if __name__ == '__main__':
    # TODO:
    # 1) image padding -> inference (победа)
    # 2) anchor points init by hand ?

    nano_model = YOLO('best_nano.pt')


    images_path = Path('test_images')
    for i, image in enumerate(list(images_path.glob('*.*'))):
        cv_image = open_image_by_path(image)
        padded_image = pad_image_correctly(cv_image)
        results = nano_model(padded_image)
        for r in results:
            im_array = r.plot()  # plot a BGR numpy array of predictions
            im = Image.fromarray(im_array[..., ::-1])  # RGB PIL image
            plt.figure(figsize=(10, 12))
            plt.imshow(im)
            plt.show()


    print('hello')
