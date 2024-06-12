import numpy as np
import cv2
from pathlib import Path
import os
import matplotlib.pyplot as plt
import uuid


def good_generate_sample(backgroung_images_path, receipts_images_path, max_receipts=4, abs_angle=30, binarize_receipt=False, full_height=False):

    back_img = get_random_back_image(backgroung_images_path)

    if max_receipts == 0:
        return back_img, []

    # get num of receipts
    receipts_to_gen = np.random.randint(1, max_receipts+1)

    # init anchor points with some offset
    if full_height:
        cur_x, cur_y = 0, 0
    else:
        cur_x, cur_y = 30, 30

    # receipt / back.width max scale
    max_of_background = 1.0 / receipts_to_gen - cur_x / back_img.shape[1]

    # list of yolo points for each receipt
    receipts_yolo_points = []

    for i in range(receipts_to_gen):
        if binarize_receipt:
            receipt_img = get_random_binarized_receipt_image(receipts_images_path)
        else:
            receipt_img = get_random_receipt_image(receipts_images_path)

        original_four_points = [(0, 0), (receipt_img.shape[1], 0), (receipt_img.shape[1], receipt_img.shape[0]), (0, receipt_img.shape[0])]

        # rotating image
        angle = np.random.randint(-abs_angle, abs_angle+1)
        receipt_img, rotation_mat = rotate_image(receipt_img, angle)

        # calculating rotated points of the receipt
        receipt_rotated_points = []
        for point in original_four_points:
        # convert to homogenous coordinates in np array format first so that you can pre-multiply M
            rotated_point = rotation_mat.dot(np.array(point + (1,)))
            receipt_rotated_points.append(rotated_point)

        # resize receipt if needed AND ALSO SCALE ROTATED POINTS ACCORDINGLY
        if receipt_img.shape[1] > back_img.shape[1] * max_of_background:
            # const keeps resulting image max 'max_of_background' width of back image
            resizing_ratio = back_img.shape[1] * max_of_background / receipt_img.shape[1]
            receipt_img = cv2.resize(receipt_img, (0, 0), fx=resizing_ratio, fy=resizing_ratio)
            for point in receipt_rotated_points:
                point[0] *= resizing_ratio
                point[1] *= resizing_ratio

        if receipt_img.shape[0] > back_img.shape[0] * 0.95:
            resizing_ratio = back_img.shape[0] * 0.95 / receipt_img.shape[0]
            receipt_img = cv2.resize(receipt_img, (0, 0), fx=resizing_ratio, fy=resizing_ratio)
            for point in receipt_rotated_points:
                point[0] *= resizing_ratio
                point[1] *= resizing_ratio

        # scale receipt to the full height of the background
        if full_height:
            resizing_ratio_y = back_img.shape[0] / receipt_img.shape[0]
            resizing_ratio_x = resizing_ratio_y
            if receipt_img.shape[1] * resizing_ratio_x > back_img.shape[1]:
                resizing_ratio_x = back_img.shape[1] / receipt_img.shape[1]
            print(resizing_ratio_y, resizing_ratio_x)
            receipt_img = cv2.resize(receipt_img, (0, 0), fx=resizing_ratio_x, fy=resizing_ratio_y)
            print(back_img.shape, receipt_img.shape)
            for point in receipt_rotated_points:
                point[0] *= resizing_ratio_x
                point[1] *= resizing_ratio_y

        # print(f'receipt shape resized {receipt_img.shape}')

        # add offset of anchor points to rotated points
        for point in receipt_rotated_points:
            point[0] += cur_x
            point[1] += cur_y

        # calculate normalized points for yolo format
        yolo_norm_points = normalize_yolo_box_1(receipt_rotated_points, back_img.shape[0], back_img.shape[1])

        receipts_yolo_points.append(yolo_norm_points)

        # placing receipt on the background
        # area for placing png receipt on the background
        x1, x2 = cur_x, cur_x + receipt_img.shape[1]
        y1, y2 = cur_y, cur_y + receipt_img.shape[0]

        if not binarize_receipt:
            # dirty way to take care of the alpha channel
            alpha_s = receipt_img[:, :, 3] / 255.0
            alpha_l = 1.0 - alpha_s

            for c in range(0, 3):
                back_img[y1:y2, x1:x2, c] = (alpha_s * receipt_img[:, :, c] +
                                             alpha_l * back_img[y1:y2, x1:x2, c])
        else:
            receipt_img = cv2.cvtColor(receipt_img, cv2.COLOR_GRAY2BGR)
            print(receipt_img.shape)
            back_img[y1:y2, x1:x2, :] = receipt_img


        # update anchor points
        if angle > 0:
            cur_x = int(receipt_rotated_points[2][0])
            cur_y = int(receipt_rotated_points[1][1])
        else:
            cur_x = int(receipt_rotated_points[1][0])
            cur_y = int(receipt_rotated_points[0][1])

        # plotting for testing
        # for point in receipt_rotated_points:
        #     back_img = cv2.circle(back_img, (int(point[0]), int(point[1])), radius=20, color=(100, 200, 200), thickness=-1)

    # plt.figure()
    # plt.imshow(back_img)
    # plt.show()

    return back_img, receipts_yolo_points


def generate_yolo_item_from_sample(backgroung_images_path, receipts_images_path, synt_yolo_ds_path, split_name,
                                   max_receipts=4, abs_angle=30, binarize_receipt=False, full_height=False):

    # choosing appropriate directory for image and labels
    images_path = os.path.join(synt_yolo_ds_path, 'images', split_name)
    labels_path = os.path.join(synt_yolo_ds_path, 'labels', split_name)

    # generate sample image, get receipt points
    sample_img, receipts_yolo_points = good_generate_sample(backgroung_images_path, receipts_images_path,
                                                            max_receipts, abs_angle, binarize_receipt, full_height)

    # for each image generate {filename}.txt with all points and class_id 0
    generated_filename = uuid.uuid4().hex

    with open(os.path.join(labels_path, f'{generated_filename}.txt'), 'w') as f:
        for coords in receipts_yolo_points:
            f.write(f'0 ' + ' '.join(map(str, coords)) + '\n')

    # and save each sample image
    cv2.imwrite(os.path.join(images_path, f'{generated_filename}.jpg'), sample_img)


def normalize_yolo_box_1(four_p_box, height, width):
    int_cords = []
    for p in four_p_box:
        int_cords.append(p[0])
        int_cords.append(p[1])

    x0, y0, x1, y1, x2, y2, x3, y3 = int_cords

    # aahhhware
    x0 = np.min((np.max((0.0, x0 / width)), 1.0))
    y0 = np.min((np.max((0.0, y0 / height )), 1.0))
    x1 = np.min((np.max((0.0, x1 / width)), 1.0))
    y1 = np.min((np.max((0.0, y1 / height)), 1.0))
    x2 = np.min((np.max((0.0, x2 / width)), 1.0))
    y2 = np.min((np.max((0.0, y2 / height)), 1.0))
    x3 = np.min((np.max((0.0, x3 / width)), 1.0))
    y3 = np.min((np.max((0.0, y3 / height)), 1.0))


    return x0, y0, x1, y1, x2, y2, x3, y3


def rotate_image(mat, angle):
    """
    Rotates an image (angle in degrees) and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape

    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0])
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    xy_center = (rotation_mat[0, 2], rotation_mat[1, 2])

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))

    return rotated_mat, rotation_mat


def get_random_back_image(backgroung_images_path):
    back_path_list = list(backgroung_images_path.glob('*.*'))
    back_idx = np.random.randint(0, len(back_path_list))

    # TODO for testing white back
    # back_idx = 1

    back_img = cv2.imread(str(back_path_list[back_idx]), -1)
    # back_img = cv2.cvtColor(back_img, cv2.COLOR_BGR2RGB)

    # back_img shape is always the same, 1000x1000
    # cuz yolo squeezes images to squares
    back_img = cv2.resize(back_img, (1000, 1000))

    return back_img


def get_random_receipt_image(receipts_images_path):
    receipts_path_list = list(receipts_images_path.glob('*.*'))
    receipt_idx = np.random.randint(0, len(receipts_path_list))
    receipt_img = cv2.imread(str(receipts_path_list[receipt_idx]), -1)

    return receipt_img


def get_random_binarized_receipt_image(receipts_images_path):
    receipts_path_list = list(receipts_images_path.glob('*.*'))
    receipt_idx = np.random.randint(0, len(receipts_path_list))

    receipt_img = cv2.imread(str(receipts_path_list[receipt_idx]), -1)

    receipt_img = binarize_image(receipt_img)[0]

    return receipt_img


def check_binarized_receipt_images(receipts_images_path, count):

    for image_path in receipts_images_path.glob('*.*'):
        if count < 0:
            break
        print(image_path.name)
        cv_image = cv2.imread(str(image_path), -1)
        cv_image_binarized = binarize_image(cv_image)[1]

        plt.imshow(cv_image_binarized, cmap='gray')
        plt.show()

        count -= 1


def binarize_image(cv_image):
    # to grayscale before binarization
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)

    # adaptive thresh
    adaptive_threshold_mean = cv2.adaptiveThreshold(cv_image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 5, 8)
    adaptive_threshold_gaussian = cv2.adaptiveThreshold(cv_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 5, 8)

    return adaptive_threshold_mean, adaptive_threshold_gaussian


if __name__ == '__main__':

    print('hello')

    backgroung_img_path = Path('C:/Users/vshishaev/Desktop/backgrounds')
    binarize_receipts_images_path = Path('C:/Users/vshishaev/Desktop/cropped_images_binarization')
    png_receipts_images_path = Path('C:/Users/vshishaev/Desktop/clear_images')
    bin_samples_saving_path = Path('C:/Users/vshishaev/Desktop/bin_dataset')
    std_samples_saving_path = Path('C:/Users/vshishaev/Desktop/std_dataset')
    full_height_samples_saving_path = Path('C:/Users/vshishaev/Desktop/full_h_dataset')
    only_back_saving_path = Path('C:/Users/vshishaev/Desktop/back_dataset')


    for i in range(100):
        generate_yolo_item_from_sample(backgroung_img_path, png_receipts_images_path, std_samples_saving_path,
                                       split_name='train', max_receipts=2, abs_angle=30, binarize_receipt=False, full_height=False)

    for i in range(10):
        generate_yolo_item_from_sample(backgroung_img_path, png_receipts_images_path, std_samples_saving_path,
                                       split_name='val', max_receipts=2, abs_angle=30, binarize_receipt=False, full_height=False)

