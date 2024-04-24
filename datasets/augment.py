import cv2 as cv


def cut_paste(img1, img2, bboxes):
    for bbox in bboxes:
        bbox = [int(b) for b in bbox]
        img1[bbox[1]:bbox[3], bbox[0]:bbox[2]] = img2[bbox[1]:bbox[3], bbox[0]:bbox[2]]
    return img1


def flip_horizontal(img, bbox):
    h, w, _ = img.shape
    img = cv.flip(img, 1)
    bbox[:, 1], bbox[:, 3] = w - bbox[:, 3], w - bbox[:, 1]
    return img, bbox


def flip_vertical(img, bbox):
    h, w, _ = img.shape
    img = cv.flip(img, 0)
    bbox[:, 0], bbox[:, 2] = w - bbox[:, 2], w - bbox[:, 0]
    return img, bbox

