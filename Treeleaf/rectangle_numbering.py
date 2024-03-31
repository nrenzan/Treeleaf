import math
from typing import *

import cv2
import numpy as np


def read_and_resize_image(img_path: str, scale_ratio: float = 0.2) -> List[any]:
   
    img = cv2.imread(
        img_path, cv2.IMREAD_UNCHANGED)  # original size (4094, 2898, 3)

    # cropping the image around the rectangles
    cropped_img = img[150:2150, 250:2700, :]  # cropped size (2000, 2450, 3)

    width = int(cropped_img.shape[1] * scale_ratio)
    height = int(cropped_img.shape[0] * scale_ratio)
    dim = (width, height)

    resized = cv2.resize(cropped_img, dim, interpolation=cv2.INTER_AREA)
    return resized


def get_corners(img, max_corners: int = 24) -> List:
    
    if img.shape.__len__() > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = np.float32(img)
    corners = cv2.goodFeaturesToTrack(img, max_corners, 0.01, 10)
    corners = np.int0(corners)
    corners = [a.tolist() for a in corners.reshape(max_corners, 2)]
    return corners


def get_rectangle_bounding_box(img) -> List[List[int]]:
   
    if img.shape.__len__() > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    binary = cv2.bitwise_not(img)

    (contours, _) = cv2.findContours(
        binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    rectangles = []
    for contour in contours:
        (x, y, w, h) = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if area > 50:
            rectangles.append((x, y, w, h))
    return rectangles


def get_len_of_lines_for_each_rect(rectangles, corners) -> Dict:
   
    lengths = {}
    for rect in rectangles:
        (x, y, w, h) = rect
        c = [x + w // 2, y + h // 2]  # center of rectangle

        # filter the points that are closest to the center of rectangles
        filtered = list(filter(lambda x: math.dist(c, x) < w, corners))

        # mapping points with their distances from center
        ps = {int(math.dist(c, p)): p for p in filtered}
        ps = dict(sorted(ps.items()))

        # two closest points from the center is the coordinate of line
        l1, l2 = list(ps.values())[:2]

        # length of line
        d = int(math.dist(l1, l2))
        lengths[d] = rect

    # sorting as per the length of the line
    return dict(sorted(lengths.items()))


def generate_numbers_on_image(img, rectangles, corners) -> None:
    
    # finding the length of line for each rectangles
    lengths = get_len_of_lines_for_each_rect(
        rectangles=rectangles, corners=corners)

    # writing numbers on the image
    i = 1
    for k, v in lengths.items():

        font = cv2.FONT_HERSHEY_SIMPLEX  # font

        (x, y, w, h) = v
        org = (x + 45, y + h + 30)  # position of text

        # fontScale
        fontScale = 1
        # Blue color in BGR
        color = (0, 0, 255)
        # Line thickness of 2 px
        thickness = 2

        # putting text on the image
        cv2.putText(img, f"{i}", org, font, fontScale,
                    color, thickness, cv2.LINE_AA)
        i += 1

    print("Generating final numbered image....")
    cv2.imwrite("numbering.jpg", img)


if __name__ == "__main__":
    image_path = "./slide2.jpg"

    # read, crop and resize image
    resized = read_and_resize_image(image_path, scale_ratio=0.25)

    # get bounding box of the rectangles i.e 4 rectangles
    rects = get_rectangle_bounding_box(resized)

    # get corners coordinates of all the shapes i.e 24 corners
    corners = get_corners(resized)

    # generate number and save image
    generate_numbers_on_image(img=resized, rectangles=rects, corners=corners)