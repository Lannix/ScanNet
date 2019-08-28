from PIL import Image
from wand.image import Image as WandImage
import wand.color as wand_color
import numpy as np
import io
import os
from scan_lib import scan_cv2 as sc2
import itertools as it
from scan_lib import nero
from scan_lib import scan_augment as aug
import imutils
import cv2


# Читает png, jpg, pdf файлы
# :param path: str
# return: list of numpy
def open_file(path):
    image_ext = ['png', 'jpg']
    file_ext = ['pdf']
    ext = path.split('.')[-1]

    if ext in image_ext:
        image = Image.open(path)
        image.convert('RGB')
        image = sc2.convert_pil_to_cv2(image)
        return [image]
    elif ext in file_ext:
        if ext == 'pdf':
            images = convert_pdf_to_png(path)
            images_cv2 = []
            for im in images:
                images_cv2.append(sc2.convert_pil_to_cv2(im))
            return images_cv2


# :param path - filename: str
# :param res - resolution: int
# return: seq of PIL.Image.Image
def convert_pdf_to_png(path, res=200):
    all_pdf_pages = WandImage(filename=path, resolution=res)
    images = []
    for i, page in enumerate(all_pdf_pages.sequence):
        with WandImage(page) as img:
            img.format = 'png'
            img.background_color = wand_color.Color('white')
            img.alpha_channel = 'remove'

            img_buffer = np.asarray(bytearray(img.make_blob(format='png')), dtype='uint8')
            bytes_io = io.BytesIO(img_buffer)
            pil_img = Image.open(bytes_io)
            pil_img.convert('RGB')
            images.append(pil_img)
    return images


class DataGen:
    def __init__(self, images_dir, masks_dir, reverse=False):
        self.generator = generator_fun(images_dir, masks_dir, reverse)

    def generate(self, num):
        if num == 0:
            return next(self.generator)
        elif num == 1:
            return next(self.generator)
        elif num == 2:
            image, mask = next(self.generator)
            model = nero.read_model(1)
            pred = nero.execute_model(mask, model)
            return image, pred
        elif num == 3:
            image, mask = next(self.generator)
            model = nero.read_model(2)
            pred = nero.execute_model(mask, model)
            return image, pred


def generator_fun(images_dir, masks_dir, reverse):
    dirs = ['png', 'jpg', 'pdf']
    images_paths = []
    masks_paths = []
    for path in os.listdir(path=images_dir):
        if path.split('.')[-1] in dirs:
            images_paths.append(images_dir + '//' + path)
    for path in os.listdir(path=masks_dir):
        if path.split('.')[-1] in dirs:
            masks_paths.append(masks_dir + '//' + path)

    images_paths = it.cycle(sorted(images_paths, key=lambda x: int(x.split('//')[-1].split('.')[0][3:]), reverse=reverse))
    masks_paths = it.cycle(sorted(masks_paths, key=lambda x: int(x.split('//')[-1].split('.')[0][3:]), reverse=reverse))
    while True:
        image = resize(open_file(next(images_paths))[0])
        image = set_white_background(image)
        # image = aug.augmentation_simple(image)
        image = set_white_background(image)

        mask = resize(sc2.find_docs_cv2(open_file(next(masks_paths))[0])[0])
        mask = set_white_background(mask)
        yield image, mask


def resize(image, max_dim=512):
    if image.shape[0] > image.shape[1]:
        image = imutils.resize(image, height=max_dim)
    else:
        image = imutils.resize(image, width=max_dim)
    return image


def set_white_background(image):
    ones = np.ones([512, 512, 3], dtype=np.uint8) * 255
    h = image.shape[0]
    w = image.shape[1]
    if h % 2 == 1:
        h = h - 1
    if w % 2 == 1:
        w = w - 1
    image = cv2.resize(image, (w, h))
    ones[256 - int(h / 2):256 + round(h / 2), 256 - int(w / 2):256 + round(w / 2)] = image
    return ones
