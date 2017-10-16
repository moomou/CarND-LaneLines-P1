#!/usr/bin/env python
import os
from collections import defaultdict

import matplotlib.image as mpimg
import numpy as np
from PIL import Image

import helper
import util
from importlib import reload

helper = reload(helper)
util = reload(util)
_state_cache = defaultdict(dict)


def detect_lane(rgb_img, state_id=None):
    '''Detects and draw lane overlay; returns img as np array'''
    global _state_cache

    h, w, chan = rgb_img.shape

    img = helper.grayscale(rgb_img)
    img = helper.gaussian_blur(img, 5)

    # mask other regions
    region = np.array([
        # middle center
        (int(w / 2), int(h / 2)),
        # bottom left
        (int(w * 0.1), int(h * 0.90)),
        # bottom right
        (int(w * 0.9), int(h * 0.90)),
    ])
    img = helper.canny(img, 30, 150)
    img = helper.region_of_interest(img, [region])

    state = _state_cache[state_id] if state_id else None
    img = helper.hough_lines(
        img,
        rho=2,
        theta=np.pi / 180,
        threshold=64,
        min_line_len=50,
        max_line_gap=40,
        state=state)
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
