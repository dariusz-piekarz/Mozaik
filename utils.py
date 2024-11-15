from PIL import Image
from numpy import array, int8, mean, argmin, ndarray, dtype
from numpy.linalg import norm
from typing import Union, Any
from random import choice


def image_to_channels(image: Image) -> tuple[ndarray[Any, dtype], ndarray[Any, dtype], ndarray[Any, dtype]]:
    """
    Convert an image to its RGB channels.

    :param image: Image.Image - The image to convert
    :return: tuple[ndarray[Any, dtype], ndarray[Any, dtype], ndarray[Any, dtype]] - RGB channels as arrays
    """

    r: Image
    g: Image
    b: Image
    r, g, b = image.split()
    return array(r), array(g), array(b)


def project_image_to_color(image: Image, color: Union[tuple[int], list[int], ndarray]) -> Image:
    """
    Projects (filtrates) an image to a specific color.

    :param image: Image.Image - The image to project
    :param color: Union[tuple[int], list[int], ndarray] - The color to project to (in RGB format)
    :return: Image.Image - The projected image
    """

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
    """
    Calculate the average RGB color of an image.

    :param image: Image.Image - The image to calculate the average color for
    :return: ndarray - The average RGB color of the image as a ndarray
    """

    r: ndarray
    g: ndarray
    b: ndarray
    r, g, b = image_to_channels(image)
    return array([mean(r), mean(g), mean(b)])


def select_closest_pict(images: list[Image], images_and_mean_rgb: list[ndarray], default_image_array: ndarray) -> Image:
    """
    Select the image from the list that is closest to the default image based on the average RGB color.

    :param images: list[Image.Image] - The list of images to select from
    :param images_and_mean_rgb: list[ndarray] - The list of average RGB colors for the images
    :param default_image_array: ndarray - The average RGB color of the default image
    :return: Image.Image - The selected image
    """

    ind: int = argmin(array([norm(im_mean_rgb - default_image_array) for im_mean_rgb in images_and_mean_rgb]))
    return images[ind]


def select_closest_pict_random(images: list[Image],
                               images_and_mean_rgb: list[ndarray],
                               default_image_array: ndarray, rank: int = 30) -> Image:
    """
    Select a random image from the list that is closest to the default image based on the average RGB color.

    :param images: list[Image.Image] - The list of images to select from
    :param images_and_mean_rgb: list[ndarray] - The list of average RGB colors for the images
    :param default_image_array: ndarray - The average RGB color of the default image
    :param rank: int - The number of closest images to select (default is 30)
    :return: Image.Image - The selected image
    """

    distances: ndarray = array([norm(im_mean_rgb - default_image_array) for im_mean_rgb in images_and_mean_rgb])
    sorted_indices: ndarray = distances.argsort()[:rank]
    random_index: int = int(choice(sorted_indices))
    return images[random_index]
