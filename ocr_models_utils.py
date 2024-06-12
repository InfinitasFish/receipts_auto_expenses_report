from torch import nn
import torch
import cv2
from itertools import groupby
import time
import numpy as np
import math


alphabets = ['ЁАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЫЪЭЮЯ 0123456789!@№%?()-=+.,/«»:;~""<>',
             'ABCDEFGHIJKLMNOPRSTUVWXYZ 0123456789!@№%?()-=+.,/«»:;~""<>']


def order_boxes(boxes):
    def contour_dist(box1, box2):
        def axiswise_dist(s1, s2):
            """
            s = (left coord, right coord)
            """
            left = min(s1, s2, key=lambda x: x[0])
            right = max(s1, s2, key=lambda x: x[0])

            return max(0, right[0] - left[1])

        delta_x = axiswise_dist((box1[0], box1[0] + box1[2]), (box2[0], box2[0] + box2[2]))
        delta_y = axiswise_dist((box1[1], box1[1] + box1[3]), (box2[1], box2[1] + box2[3]))

        if delta_x == delta_y == 0:
            return 0
        if min(delta_x, delta_y) == 0 and max(delta_x, delta_y) > 0:
            return max(delta_x, delta_y)
        return math.sqrt(delta_x * delta_x + delta_y * delta_y)

    y_sorted_boxes = np.array(sorted(boxes.copy(), key=lambda x: x[1]))

    epsilon = np.mean(boxes[:, 3])

    cluster_labels = np.full((len(y_sorted_boxes, )), fill_value=-1)

    cluster_labels[0] = 0

    for i in range(len(y_sorted_boxes)):
        if cluster_labels[i] == -1:
            cluster_labels[i] = max(cluster_labels) + 1
        for j in range(len(y_sorted_boxes)):
            if contour_dist(y_sorted_boxes[i], y_sorted_boxes[j]) < epsilon:
                min_l = min(cluster_labels[i], cluster_labels[j])
                max_l = max(cluster_labels[i], cluster_labels[j])
                if min_l == -1:
                    cluster_labels[i] = max_l
                    cluster_labels[j] = max_l
                    continue
                cluster_labels[cluster_labels == max_l] = min_l

    cluster_labels = np.array(cluster_labels)

    for i in range(max(cluster_labels)):
        if i not in cluster_labels:
            cluster_labels[cluster_labels[cluster_labels > i].min()] = i

    sorted_boxes = list()
    for c in np.unique(cluster_labels):
        cluster_boxes = y_sorted_boxes[cluster_labels == c]

        group = [cluster_boxes[0]]
        for i in range(1, len(cluster_boxes)):
            if abs(cluster_boxes[i][1] - cluster_boxes[i - 1][1]) < (epsilon*0.5):
                group.append(cluster_boxes[i])
            else:
                sorted_boxes.extend(sorted(group, key=lambda x: x[0]))
                group = [cluster_boxes[i]]
        sorted_boxes.extend(sorted(group, key=lambda x: x[0]))

    return sorted_boxes


def prep_image_for_detection(image, imgsz=1024):
    coef = min(imgsz / image.shape[0], imgsz / image.shape[1])
    image_ = cv2.resize(image.copy(), dsize=None, fx=coef, fy=coef, interpolation=cv2.INTER_CUBIC)
    image_ = cv2.copyMakeBorder(image_, 0, imgsz-image_.shape[0], 0, imgsz-image_.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))

    image_ = torch.FloatTensor(image_) / 255
    image_ = torch.permute(image_, (2, 0, 1)).unsqueeze(0)

    return image_, coef


def non_max_suppression_fast(boxes, overlapThresh):
    if len(boxes) == 0:
        return []
    if boxes.dtype.kind == "i":
        boxes = boxes.astype("float")
    pick = []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    area = (x2 - x1 + 1) * (y2 - y1 + 1)

    idxs = np.arange(boxes.shape[0])

    while len(idxs) > 0:
        # grab the last index in the indexes list and add the
        # index value to the list of picked indexes
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        # find the largest (x, y) coordinates for the start of
        # the bounding box and the smallest (x, y) coordinates
        # for the end of the bounding box
        xx1 = np.maximum(x1[i], x1[idxs[:last]])
        yy1 = np.maximum(y1[i], y1[idxs[:last]])
        xx2 = np.minimum(x2[i], x2[idxs[:last]])
        yy2 = np.minimum(y2[i], y2[idxs[:last]])
        # compute the width and height of the bounding box
        w = np.maximum(0, xx2 - xx1 + 1)
        h = np.maximum(0, yy2 - yy1 + 1)
        # compute the ratio of overlap
        overlap = (w * h) / area[idxs[:last]]
        # delete all indexes from the index list that have
        idxs = np.delete(idxs, np.concatenate(([last], np.where(overlap > overlapThresh)[0])))
        # return only the bounding boxes that were picked using the
        # integer data type
    return boxes[pick].astype("int")


def get_boxes(model, image):
    image_, scale = prep_image_for_detection(image)
    H, W = image.shape[:2]

    output = model(image_)
    output = output.detach().cpu().numpy()[0]

    boxes = list()

    output = output[output[:, 4] > 0.25]

    output[:, 5:] *= output[:, 4:5]

    output[:, 0] = (output[:, 0] - output[:, 2] / 2)  # xc to top left x
    output[:, 1] = (output[:, 1] - output[:, 3] / 2)  # yc to top left y

    output[:, :4] /= scale

    for detection in output:
        x, y, w, h = detection[:4]

        conf = detection[4]
        class_ohe = detection[5:]

        boxes.append([x, y, w, h, conf, np.argmax(class_ohe)])

    boxes = np.array(boxes)
    if boxes.size == 0:
        return [], [], []

    boxes[(boxes[:, 2] > 0) & (boxes[:, 3] > 0)]
    confs = boxes[:, 4]
    cls = boxes[:, 5]
    boxes = boxes[:, :4]

    nmsstart = time.time()
    boxes = boxes[np.argsort(confs)]

    xyxy_boxes = boxes.copy()
    xyxy_boxes[:, 2] += xyxy_boxes[:, 0]
    xyxy_boxes[:, 3] += xyxy_boxes[:, 1]

    xyxy_boxes = non_max_suppression_fast(xyxy_boxes, 0.25)
    boxes = xyxy_boxes.copy()
    boxes[:, 2] -= boxes[:, 0]
    boxes[:, 3] -= boxes[:, 1]

    # print('nms time:', time.time()-nmsstart)

    boxes = np.maximum(boxes, 0)
    boxes[:, 0] = np.minimum(boxes[:, 0], image.shape[1])
    boxes[:, 1] = np.minimum(boxes[:, 1], image.shape[0])
    boxes[:, 2] = np.minimum(boxes[:, 2], image.shape[1]-boxes[:, 0])
    boxes[:, 3] = np.minimum(boxes[:, 3], image.shape[0]-boxes[:, 1])

    boxes = np.round(boxes[:, :4]).astype(int)
    return boxes, confs, cls


def safe_convert_to_grayscale(image_to_convert):
    if len(image_to_convert.shape) == 2:
        return image_to_convert
    if len(image_to_convert.shape) == 3:
        if image_to_convert.shape[2] == 1:
            return image_to_convert[:, :, 0]
        if image_to_convert.shape[2] == 3:
            return cv2.cvtColor(image_to_convert, cv2.COLOR_BGR2GRAY)
        if image_to_convert.shape[2] == 4:
            return cv2.cvtColor(image_to_convert, cv2.COLOR_BGRA2GRAY)

    raise ValueError('invalid shape')


def decode_texts(logits, alphabet, blank_idx):
    if blank_idx < 0:
        blank_idx = len(alphabet)
    best_path_indices = np.argmax(logits, axis=-1)
    best_chars_collapsed = [[alphabet[idx-(idx >= blank_idx)] for idx, _ in groupby(e) if idx != blank_idx and idx < len(alphabet)]
                            for e in best_path_indices]
    return [''.join(e) for e in best_chars_collapsed]


def prepare_segment_for_recognition(segment, target_shape=(32, 256)):
    segment = safe_convert_to_grayscale(segment)

    coef = min(target_shape[0] / segment.shape[0], target_shape[1] / segment.shape[1])

    segment = cv2.resize(segment, dsize=None, fx=coef, fy=coef, interpolation=cv2.INTER_AREA if coef < 1 else cv2.INTER_CUBIC)
    segment = cv2.copyMakeBorder(segment, 0, target_shape[0]-segment.shape[0], 0, target_shape[1]-segment.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    segment = torch.FloatTensor(segment / 255)

    return segment.unsqueeze(0).unsqueeze(0)


def recognize_text(model, image, boxes, alphabets, batch_size=64):
    result = list()
    segments = [image[y: y+h, x: x+w] for (x, y, w, h) in boxes]
    segments = [prepare_segment_for_recognition(segment) for segment in segments]
    segments = torch.cat(segments, axis=0)

    for i in range(1, segments.shape[0] // batch_size + (segments.shape[0] % batch_size != 0) + 1):
        with torch.no_grad():
            y_text, y_script = model(segments[(i-1)*batch_size: i*batch_size])
        rus_indices = np.where(y_script.argmax(-1).numpy() == 0)
        eng_indices = np.where(y_script.argmax(-1).numpy() == 1)

        rus_texts = decode_texts(y_text[rus_indices].cpu().numpy(), alphabets[0], blank_idx=0)
        eng_texts = decode_texts(y_text[eng_indices].cpu().numpy(), alphabets[1], blank_idx=0)

        output = list()
        ridx = 0
        eidx = 0

        for idx in y_script.argmax(-1):
            if idx == 0:
                output.append(rus_texts[ridx])
                ridx += 1
            else:
                output.append(eng_texts[eidx])
                eidx += 1

        result.extend(output)

    return result


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, pool_ksize=(2, 2)):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1),
                                   nn.LeakyReLU(0.1),
                                   nn.BatchNorm2d(out_channels),
                                   nn.MaxPool2d(pool_ksize))

    def forward(self, x):
        return self.block(x)


class ScriptClassificationModel(nn.Module):
    def __init__(self, nscripts, input_shape):
        super(ScriptClassificationModel, self).__init__()

        self.feature_extractor = nn.Sequential(nn.Conv1d(input_shape[0], input_shape[0]//2, 4),
                                               nn.LeakyReLU(0.1),
                                               nn.BatchNorm1d(input_shape[0]//2),
                                               nn.MaxPool1d(4),

                                               nn.Conv1d(input_shape[0]//2, input_shape[0]//4, 4),
                                               nn.LeakyReLU(0.1),
                                               nn.BatchNorm1d(input_shape[0]//4),
                                               nn.MaxPool1d(4),)

        self.fc = nn.Sequential(nn.Linear(64, 32),
                                nn.ReLU(),
                                nn.Linear(32, nscripts),
                                nn.Softmax(dim=-1))

    def forward(self, x):
        x = self.feature_extractor(x)
        x = x.view((x.shape[0], -1))

        x = self.fc(x)

        return x


class RecognitionHead(nn.Module):
    def __init__(self, alphabet_len):
        super(RecognitionHead, self).__init__()

        self.lstm1 = nn.LSTM(256, 256, batch_first=True)
        self.lstm2 = nn.LSTM(256, 256, batch_first=True)

        self.fc = nn.Sequential(nn.Linear(256, alphabet_len+1),
                                nn.Softmax(dim=2))

    def forward(self, x):
        x = torch.permute(x, (0, 2, 1))

        x, h = self.lstm1(x, None)
        x, h = self.lstm2(x, h)
        x = self.fc(x)

        return x


class MCRNN(nn.Module):
    def __init__(self, alphabet_lens):
        super(MCRNN, self).__init__()

        self.feature_extractor = nn.Sequential(ConvBlock(1, 16),
                                               ConvBlock(16, 32, (2, 1)),
                                               ConvBlock(32, 64),
                                               ConvBlock(64, 128),
                                               ConvBlock(128, 256, (2, 1)))

        self.script_classifier = ScriptClassificationModel(len(alphabet_lens), (256, 32))

        self.recognition_heads = nn.ModuleList([RecognitionHead(max(alphabet_lens)) for _ in alphabet_lens])

        self.output_dim = max(alphabet_lens) + 1

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(self, x):
        x = self.feature_extractor(x).squeeze(2)

        script_probs = self.script_classifier(x)
        script_indices = script_probs.argmax(-1)

        output = torch.zeros((x.shape[0], x.shape[2], self.output_dim), device=self.device)

        for sidx in torch.unique(script_indices):
            output[script_indices == sidx] = self.recognition_heads[sidx](x[script_indices == sidx])

        return output, script_probs


def calc_iou(box1: np.ndarray, box2: np.ndarray) -> float:
    intersection_area = calc_intersection_area(box1, box2)

    return intersection_area / (box1[2] * box1[3] + box2[2] * box2[3] - intersection_area)


def calc_intersection_area(box1, box2):
    xA = max(box1[0], box2[0])
    yA = max(box1[1], box2[1])
    xB = min(box1[0] + box1[2], box2[0] + box2[2])
    yB = min(box1[1] + box1[3], box2[1] + box2[3])
    intersection_area = max(0, xB - xA) * max(0, yB - yA)

    return intersection_area


def evaluate_ocr_result(gt_boxes, pred_boxes, gt_words, pred_words, iou_thr=0.5):
    TP = 0
    FP = 0
    FN = 0

    matched_indices = list()
    used_boxes_mask = np.zeros((pred_boxes.shape[0],), dtype=bool)
    for i, gt_box in enumerate(gt_boxes):
        best_iou = 0
        best_idx = None
        for j, pred_box in enumerate(pred_boxes):
            pred_box = pred_box.ravel()
            gt_box = pred_box.ravel()
            if used_boxes_mask[j]:
                continue
            iou = calc_iou(pred_box, gt_box)
            if iou > best_iou:
                best_iou = iou
                best_idx = j

        if best_iou > iou_thr:
            matched_indices.append((i, best_idx))
            used_boxes_mask[best_idx] = 1
            TP += 1
        else:
            FN += 1

    FP = used_boxes_mask.shape[0] - sum(used_boxes_mask)

    precision = TP / (TP + FP)
    recall = TP / (TP + FN)

    f1_score = 2 * precision * recall / (precision + recall)

    lratios = list()
    wers = list()

    for i, j in matched_indices:
        lratios.append(ratio(gt_words[i].lower(), pred_words[j].lower()))
        wers.append(gt_words[i].lower() == pred_words[j].lower())

    return sum(lratios) / len(lratios), sum(wers) / len(wers), precision, recall, f1_score
