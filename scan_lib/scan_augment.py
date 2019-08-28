import albumentations as al
import cv2


def augmentation_simple(image, p=1., sub_p=0.3):
    augmentation_fun = al.Compose(
        [
            al.OneOf([
                al.IAAAdditiveGaussianNoise(),
                al.GaussNoise(),
            ], p=sub_p),
            al.OneOf([
                al.MotionBlur(p=sub_p),
                al.MedianBlur(blur_limit=3, p=sub_p),
                al.Blur(blur_limit=3, p=sub_p),
            ], p=sub_p),
            al.OneOf([
                al.OpticalDistortion(p=sub_p),
                al.GridDistortion(p=sub_p),
                al.IAAPiecewiseAffine(p=sub_p),
            ], p=sub_p),
            al.OneOf([
                al.CLAHE(clip_limit=3),
                al.IAASharpen(),
                al.IAAEmboss(),
                al.RandomBrightnessContrast()
            ], p=sub_p)
        ], p=p)
    return augmentation_fun(image)


def augmentation_hard(image, p=1., sub_p=0.3):
    augmentation_fun = al.Compose(
        [
            al.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=8, p=sub_p,
                                border_mode=cv2.BORDER_CONSTANT),
            al.ElasticTransform(sub_p),
            al.OneOf([
                al.IAAAdditiveGaussianNoise(),
                al.GaussNoise(),
            ], p=sub_p),
            al.OneOf([
                al.MotionBlur(p=sub_p),
                al.MedianBlur(blur_limit=3, p=sub_p),
                al.Blur(blur_limit=3, p=sub_p),
            ], p=sub_p),
            al.OneOf([
                al.OpticalDistortion(p=sub_p),
                al.GridDistortion(p=sub_p),
                al.IAAPiecewiseAffine(p=sub_p),
            ], p=sub_p),
            al.OneOf([
                al.CLAHE(clip_limit=3),
                al.IAASharpen(),
                al.IAAEmboss(),
                al.RandomBrightnessContrast()
            ], p=sub_p)
        ], p=p)
    return augmentation_fun(image)


def augment(image, au_type='simple', p=1.):
    if au_type == 'simple':
        augmentation_fun = augmentation_simple(p)
    elif au_type == 'hard':
        augmentation_fun = augmentation_hard(p)
    elif au_type == 'compose':
        # todo
        augmentation_fun = augmentation_simple(p)
    else:
        augmentation_fun = augmentation_simple(p)
    return augmentation_fun(image)
