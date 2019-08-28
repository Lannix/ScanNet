"""
ScanNet
"""

from scan_lib import scan_data as sd
from PIL import Image
import cv2
import imutils
from imutils.perspective import four_point_transform
from skimage.filters import threshold_local
import numpy as np


# param file_path: str
# param to_gray_scale: bool
# return: numpy array
def find_im_in_docs(file_path):
    docs = []
    images = open_file(file_path)
    for im in images:
        new_docs = find_docs_cv2(im)
    return docs


# param path: str
# return: PIL.Image.Image
def open_file(path):
    return sd.open_file(path)


# param pil_img: PIL.Image.Image
# return: numpy array
def convert_pil_to_cv2(pil_img):
    numpy_image = np.array(pil_img)
    cv_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)
    return cv_image


# param image: numpy array
# return: PIL.Image.Image
def convert_cv2_to_pil(cv_image):
    cv_image = cv_image.astype(np.uint8)
    cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(cv_image)
    return pil_img


# Convert the image to grayscale, blur it, and find edges
# in the image.
#
# param image: numpy array
# return: numpy array
def find_edges_by_cv2(image, is_gray=True):
    if not is_gray:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.Canny(image, 75, 200)
    return image


# param image: numpy array
# return: seq of lists of points
def find_contours_by_cv2(image, grab=True):
    contours = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    if grab:
        contours = imutils.grab_contours(contours)
    return contours


# param contours: turple
# return: turple
def find_largest_areas(contours, cntrs_cnt=1):
    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    return contours


# param contours: numpy array of points
# return: numpy array of points
def find_rects(contours):
    main_contours = []
    for contour in contours:
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        area = int(rect[1][0] * rect[1][1])
        if area > 500:
            main_contours.append(box)
    return np.asarray(main_contours)


# param orig_image: numpy array
# param points: numpy array
# return: numpy array
def four_points_transform_by_cv2(image, points):
    return four_point_transform(image, points)


# param image: numpy array
# return: numpy array
def to_grayscale_cv2(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    t = threshold_local(image, 11, offset=10, method="gaussian")
    image = (image > t).astype("uint8") * 255
    return image


# param image: ndarray
# return: ndarray
def draw_contours(image):
    orig_image = image.copy()
    ratio = orig_image.shape[0] / 500.0
    image = imutils.resize(image, height=500)

    image = mean_shift_cv2(image)
    image = thresholding_otsu_cv2(image)
    image = remove_noise_cv2(image)
    image = find_edges_by_cv2(image)

    image = imutils.resize(image, height=int(500 * ratio))
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    image_contuored = cv2.drawContours(orig_image, contours, -1, (255, 0, 0), 2)
    return image_contuored


# param img: numpy array
# return: numpy array
def thresholding_otsu_cv2(img, blur=True, inverse=True, to_grayscale=True):
    orig_height = img.shape[0]
    img = imutils.resize(img, height=500)
    if blur:
        img = cv2.medianBlur(img, 5)
    if to_grayscale:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if inverse:
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    else:
        ret, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    img = imutils.resize(img, height=orig_height)
    return img


def remove_noise_cv2(img, iterations=2):
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=iterations)
    kernel = np.ones((3, 3), np.uint8)
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=iterations)
    return img


# param image: ndarray
# return: ndarray
def mean_shift_cv2(image):
    return cv2.pyrMeanShiftFiltering(image, 21, 51)


# param image: ndarray
# return: list of ndarray
def dilate_cv2(image, iterations=1):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.dilate(image, kernel, iterations=iterations)


# param image: ndarray
# return: list of ndarray
def erod_cv2(img, iterations=1):
    kernel = np.ones((5, 5), np.uint8)
    return cv2.erode(img, kernel, iterations=iterations)


# param image: ndarray
# return: list of ndarray
def find_docs_cv2(image):
    orig_image = image.copy()
    ratio = orig_image.shape[0] / 500.0
    image = imutils.resize(image, height=500)

    image = mean_shift_cv2(image)
    image = thresholding_otsu_cv2(image)
    image = remove_noise_cv2(image)
    # image = dilate_cv2(image, 1)
    image = find_edges_by_cv2(image)

    contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_TC89_KCOS)
    contours = imutils.grab_contours(contours)
    main_contours = find_rects(contours) * ratio

    images = []
    for contour in main_contours:
        points = np.asarray(contour).reshape(4, 2)
        document = four_point_transform(orig_image.copy(), points)
        shape = document.shape
        images.append(document)
    return images
