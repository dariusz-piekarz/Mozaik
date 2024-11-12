from PIL import Image
from numpy import array, int8, mean, argmin, ndarray, dtype
from numpy.linalg import norm
from typing import Union, Any
from random import choice


def image_to_channels(image: Image) -> tuple[ndarray[Any, dtype], ndarray[Any, dtype], ndarray[Any, dtype]]:
    r: Image
    g: Image
    b: Image
    r, g, b = image.split()
    return array(r), array(g), array(b)


def project_image_to_color(image: Image, color: Union[tuple[int], list[int], ndarray]) -> Image:
    r: ndarray
    g: ndarray
    b: ndarray
    r, g, b = image_to_channels(image)
    color_ratio: tuple[float, float, float] = color[0] / 255, color[1] / 255, color[2] / 255
    r = (r * color_ratio[0]).astype(int8)
    g = (g * color_ratio[1]).astype(int8)
    b = (b * color_ratio[2]).astype(int8)
    r_channel: Image = Image.fromarray(r, mode="L")
    g_channel: Image = Image.fromarray(g, mode="L")
    b_channel: Image = Image.fromarray(b, mode="L")
    return Image.merge("RGB", (r_channel, g_channel, b_channel))


def avg_image_rgb(image: Image) -> ndarray:
    r: ndarray
    g: ndarray
    b: ndarray
    r, g, b = image_to_channels(image)
    return array([mean(r), mean(g), mean(b)])


def select_closest_pict(images: list[Image], images_and_mean_rgb: list[ndarray], default_image_array: ndarray) -> Image:
    ind: int = argmin(array([norm(im_mean_rgb - default_image_array) for im_mean_rgb in images_and_mean_rgb]))
    return images[ind]


def select_closest_pict_random(images: list[Image],
                               images_and_mean_rgb: list[ndarray],
                               default_image_array: ndarray, rank: int = 30) -> Image:
    distances: ndarray = array([norm(im_mean_rgb - default_image_array) for im_mean_rgb in images_and_mean_rgb])
    sorted_indices: ndarray = distances.argsort()[:rank]
    random_index: int = int(choice(sorted_indices))
    return images[random_index]
