import ultralytics
import cv2
import numpy as np
from math import floor, ceil


def initialize_yolo_obb(model_path):
    model = ultralytics.YOLO(model_path)
    return model


def pad_image_correctly(cv_image):
    h, w = cv_image.shape[0], cv_image.shape[1]
    result_size = ceil(max(h, w) / 32) * 32
    top, bottom = floor((result_size - h) / 2), ceil((result_size - h) / 2)
    left, right = floor((result_size - w) / 2), ceil((result_size - w) / 2)
    padded_image = cv2.copyMakeBorder(cv_image, top, bottom, left, right, cv2.BORDER_REPLICATE)

    return padded_image


def yolo_obb_pipeline(yolo_model, np_image):

    padded_image = pad_image_correctly(np_image)
    results = yolo_model(padded_image)
    result_separated_images = get_cropped_result_boxes(results)
    return result_separated_images


def get_cropped_result_boxes(obb_results):
    cropped_images = []
    for result in obb_results:

        for box in result.obb.xyxyxyxy:

            image_cv = result.orig_img.astype(np.uint8)

            cnt = box.cpu().numpy().astype(int)
            rect = cv2.minAreaRect(cnt)
            should_rotate_90 = rect[2] > 45

            # the order of the box points: bottom left, top left, top right,
            # bottom right
            box = cv2.boxPoints(rect)
            box = np.intp(box)

            cv2.drawContours(image_cv, [box], 0, (0, 0, 255), 2)

            # get width and height of the detected rectangle
            width = int(rect[1][0])
            height = int(rect[1][1])

            src_pts = box.astype("float32")
            # coordinate of the points in box points after the rectangle has been
            # straightened
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")

            # the perspective transformation matrix
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)

            # directly warp the rotated rectangle to get the straightened rectangle
            warped = cv2.warpPerspective(image_cv, M, (width, height))

            if should_rotate_90:
                warped = cv2.rotate(warped, cv2.ROTATE_90_CLOCKWISE)

            cropped_images.append(warped)

    return cropped_images

