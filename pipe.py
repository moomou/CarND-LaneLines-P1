#!/usr/bin/env python
import os

from PIL import Image
import cv2
import matplotlib.image as mpimg
import numpy as np

import helper
from importlib import reload

helper = reload(helper)


def detect_lane(rgb_img):
    '''Detects and draw lane overlay; returns img as np array'''
    # steps
    # conver to grayscale
    # run canny edge detection
    # run hough transform to detect lines
    # combine line segs

    h, w, chan = rgb_img.shape

    img = helper.grayscale(rgb_img)
    img = helper.gaussian_blur(img, 5)
    # mask other regions
    region = np.array([
        # middle center
        (int(w / 2), int(h / 2)),
        # bottom left
        (int(w * 0.1), h),
        # bottom right
        (int(w * 0.9), h),
    ])

    img = helper.canny(img, 30, 150)
    img = helper.region_of_interest(img, [region])

    img = helper.hough_lines(
        img,
        rho=2,
        theta=np.pi / 180,
        threshold=64,
        min_line_len=50,
        max_line_gap=40)
    img = helper.weighted_img(img, rgb_img)

    return img


def process_image(img_root='test_images'):
    for img in os.listdir(img_root):
        image = mpimg.imread(os.path.join(img_root, img))
        img_array = detect_lane(image)

        im = Image.fromarray(img_array)
        im.save(os.path.join(img_root + '_output', img))


if __name__ == '__main__':
    import fire

    fire.Fire({'img': process_image})
