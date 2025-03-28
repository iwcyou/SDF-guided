"""
Based on https://github.com/asanakoy/kaggle_carvana_segmentation
"""
import cv2
import numpy as np


def randomHueSaturationValue(
    image,
    hue_shift_limit=(-30, 30),
    sat_shift_limit=(-5, 5),
    val_shift_limit=(-15, 15),
    u=0.5,
):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(
            hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image


def randomShiftScaleRotate(
    image,
    mask,
    shift_limit=(-0.1, 0.1),
    scale_limit=(-0.1, 0.1),
    aspect_limit=(-0.1, 0.1),
    rotate_limit=(-0, 0),
    borderMode=cv2.BORDER_CONSTANT,
    u=0.5,
):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height]])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array(
            [width / 2 + dx, height / 2 + dy]
        )

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(
            image,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(0, 0, 0),
        )
        mask = cv2.warpPerspective(
            mask,
            mat,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=borderMode,
            borderValue=(0, 0, 0),
        )

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        if image.ndim <= 2:
            image = cv2.flip(image, 1)
        else:
            for i in range(image.shape[2]):
                image[:, :, i] = cv2.flip(image[:, :, i], 1)

        if mask.ndim <= 2:
            mask = cv2.flip(mask, 1)
        else:
            for i in range(mask.shape[2]):
                mask[:, :, i] = cv2.flip(mask[:, :, i], 1)

    return image, mask

def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        if image.ndim <= 2:
            image = cv2.flip(image, 0)
        else:
            for i in range(image.shape[2]):
                image[:, :, i] = cv2.flip(image[:, :, i], 0)

        if mask.ndim <= 2:
            mask = cv2.flip(mask, 0)
        else:
            for i in range(mask.shape[2]):
                mask[:, :, i] = cv2.flip(mask[:, :, i], 0)

    return image, mask

def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)

    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)

    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        image = np.rot90(image)
        mask = np.rot90(mask)

    return image, mask
