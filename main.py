from core import image_from_images
from pytools.config import Config
from pathlib import Path
from os.path import isdir, isfile, dirname
from typing import Optional


def main() -> None:
    cfg_path: str = str(Path(__file__)) + "\\config.json"
    cfg: Config = Config(cfg_path)
    show: bool = cfg.show == "True"
    output_path: str = cfg.output_image_path if cfg.output_image_path != "" else None

    image_from_images(cfg.image_path,
                      cfg.filler_images_dir_path,
                      cfg.strategy,
                      tuple(cfg.image_size),
                      tuple(cfg.sub_image_size),
                      show,
                      output_path)


def main2() -> None:

    image_path: str = ""
    while not isfile(image_path):
        image_path = input("Provide the main image location: ")

    filler_images_dir_path: str = ""
    while not isdir(filler_images_dir_path):
        filler_images_dir_path = input("Provide filling images folder location: ")

    strategy: str = ""
    while strategy not in ['pixel_mean', 'straight', 'duplication']:
        strategy = input("Choose a strategy (pixel_mean, straight, duplication): ")

    image_size: tuple[int, ...] = (0, 0)
    while len(image_size) != 2 or not all(isinstance(x, int) and x > 0 for x in image_size):
        image_size = tuple(map(int, input("Enter image size (width, height): ").split()))

    sub_image_size: tuple[int, ...] = (0, 0)
    while len(sub_image_size) != 2 or not all(isinstance(x, int) and x > 0 for x in sub_image_size):
        sub_image_size = tuple(map(int, input("Enter sub-image size (width, height): ").split()))

    show: bool = True
    show_input: str = input("Do you want to display the result image (True/False)?")
    if show_input.lower() in ['false', 'no', 'n', 'f']:
        show = False
    elif show_input.lower() in ['true', 'yes', 'y', 't', '']:
        show = True
    else:
        raise ValueError("Unknown value of `show` parameter")

    output_path: Optional[str] = None
    output_path_input: str = input("Do you want to save the result image to a file (True/False)?")
    if output_path_input.lower() in ['false', 'no', 'n', 'f', '']:
        output_path = None
    elif output_path_input.lower() in ['true', 'yes', 't', 'y']:
        output_path = input("Enter the output file path: ")
        if not isdir(dirname(output_path)):
            raise ValueError("The output folder does not exists!")

    image_from_images(image_path,
                      filler_images_dir_path,
                      strategy,
                      image_size,
                      sub_image_size,
                      show,
                      output_path)


if __name__ == '__main__':
    main()
