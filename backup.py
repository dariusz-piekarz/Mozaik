from PIL import Image
from numpy import array, int8, mean, argmin, ndarray, dtype
from numpy.linalg import norm
from typing import Union, Optional, Any
from glob import glob
from functools import reduce
from math import ceil
from joblib import Parallel, delayed
from loguru import logger
from numba import jit
from pytools.metaclass import time_decor


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


@time_decor
def uniform_resize(images: list[Image], size: tuple[int, int] = (100, 100)) -> list[Image]:
    if len(images) > 100:
        return Parallel(n_jobs=-1,
                        prefer='processes',
                        return_as='list')(delayed(lambda img: img.resize(size))(image) for image in images)
    else:
        return [im.resize(size) for im in images]


def avg_image_rgb(image: Image) -> ndarray:
    r: ndarray
    g: ndarray
    b: ndarray
    r, g, b = image_to_channels(image)
    return array([mean(r), mean(g), mean(b)])


def select_closest_pict(images: list[Image], images_and_mean_rgb: list[ndarray], default_image_array: ndarray) -> Image:
    ind: int = argmin(array([norm(im_mean_rgb - default_image_array) for im_mean_rgb in images_and_mean_rgb]))
    return images[ind]


@jit(nopython=True)
def _select_closest_pict(images: list[ndarray],
                         images_and_mean_rgb: list[ndarray],
                         default_image_array: ndarray) -> ndarray:
    return images[argmin(norm(array(images_and_mean_rgb) - default_image_array))]


@time_decor
def reorder_pictures(images: list[Image], default_image: Image, strategy: str = 'pixel_mean') -> list[list[Image]]:
    if strategy not in ['pixel_mean', 'straight', 'duplication']:
        raise ValueError("`strategy` admits only one of values 'pixel_mean', 'straight', 'duplication'!")

    size: tuple[int, int] = default_image.size
    no_pixels: int = reduce(lambda x, y: x * y, size)

    if strategy == 'straight':
        if len(images) != no_pixels:
            raise ValueError("Choice of `strategy` requires having the number of `images`"
                             " the same as number of pixels in the `default_image`.")

        return [[images[i * size[0] + j] for j in range(size[0])] for i in range(size[1])]

    elif strategy == 'duplication':
        duplicated_images: list[Image] = (images * ceil(no_pixels / len(images)))[:no_pixels]
        return [[duplicated_images[i * size[0] + j] for j in range(size[0])] for i in range(size[1])]

    else:
        images_and_mean_rgb: list[ndarray] = []
        default_image_array: ndarray = array(default_image)
        Parallel(n_jobs=-1)(images_and_mean_rgb.append(avg_image_rgb(im)) for im in images)
        return [[select_closest_pict(images, images_and_mean_rgb, default_image_array[i, j]) for j in range(size[1])]
                for i in range(size[0])]


@time_decor
def project(images: list[list[Image]], image: Image) -> list[list[Image]]:
    image_arr: ndarray = array(image)
    return [[project_image_to_color(images[i][j], image_arr[j, i])
             for j in range(image.size[1])]
            for i in range((image.size[0]))]


@time_decor
def glue_image(images: list[list[Image]], show: bool = True, save_to: Optional[str] = None) -> Image:
    image_width: int
    image_height: int
    image_width, image_height = images[0][0].size
    new_width: int = len(images[0]) * image_width
    new_height: int = len(images) * image_height

    new_image: Image = Image.new('RGB', (new_width, new_height))

    for i in range(len(images)):
        for j in range(len(images[0])):
            new_image.paste(images[i][j], (i * image_width, j * image_height))

    if show:
        new_image.show()
    if save_to:
        new_image.save(save_to)

    return new_image


@time_decor
def load_pictures(pictures: Union[list[str], list[Image.Image]],
                  picture: Union[str, Image.Image]) -> tuple[list[Image.Image], Image.Image]:
    image: Image
    images: list[Image]
    if all([isinstance(path, str) for path in pictures]):
        logger.info("Loading images.")
        images = [Image.open(path).convert("RGB") for path in glob(pictures + "\\*")]
    elif all([isinstance(im, Image.Image) for im in pictures]):
        images = pictures
    else:
        raise ValueError("All values in the list `pictures` must be either string paths or PIL.Images!")

    if isinstance(picture, str):
        image: Image = Image.open(picture).convert("RGB")
    elif isinstance(picture, Image.Image):
        image = picture
    else:
        raise ValueError("The `picture` must be either string path or PIL.Image!")
    return images, image


def original_picture_resize(image: Image, image_size: tuple[int, int]) -> Image:
    original_proportion: float = reduce(lambda x, y: x / y, image.size)
    if original_proportion != reduce(lambda x, y: x / y, image_size):
        new_size_width: tuple[int, int] = (image_size[0], int(original_proportion * image_size[0]))
        new_size_height: tuple[int, int] = (int(image_size[1] * original_proportion), image_size[1])
        logger.warning(f"Original proportions perturbed."
                       f" Proposed proportions are {new_size_width} or {new_size_height}. "
                       f"Press 1 if you want to select {new_size_width}, "
                       f"press 2 if you want   to select {new_size_height},"
                       f" press 3 if you want to keep {image_size}.")
        response: str = input()
        if response == '1':
            return image.resize(new_size_width)
        elif response == '2':
            return image.resize(new_size_height)
        else:
            return image.resize(image_size)
    else:
        return image


@time_decor
def image_from_images(pictures: Union[str, list[Image]],
                      picture: Union[str, Image],
                      strategy: str = 'pixel_mean',
                      image_size: tuple[int, int] = (200, 200),
                      sub_images_size: tuple[int, int] = (50, 50),
                      show: bool = True,
                      save_to: Optional[str] = None) -> Image:

    images: list[Image]
    image: Image

    images, image = load_pictures(pictures, picture)

    logger.info("Images loaded.")

    image = original_picture_resize(image, image_size)

    logger.info("Pictures rescaling started.")

    images = uniform_resize(images, sub_images_size)

    logger.info("Pictures rescaling finished. Reordering pictures started.")

    reordered_pictures: list[list[Image]] = reorder_pictures(images, image, strategy)

    logger.info("Reordering pictures finished. Images filtration to pixels started.")

    proj_array: list[list[Image]] = project(reordered_pictures, image)

    logger.info("Images filtration finished. Combining pictures started.")

    return glue_image(proj_array, show, save_to)
