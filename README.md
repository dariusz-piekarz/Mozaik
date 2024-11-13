# Mozaik
App replace pixels on selected picture by other images.
Allows several strategies:
1) `straight`: number of pixels of the main image == number of images you want to replace its pixels by.

2) `duplication`: number of replacing image is less than total number of pixels of the original image. Images are duplicated to cover number of pixels in the original photo.

3) `pixel_mean`: To each image is assign its mean RGB and thenit is searched for each pixel an image that is the closest to this pixel with respec to the $l^2$ distance.

4)  `pixel_mean_random`: As in the case 3), mean RGB is calculated for each picture. Then it is searached for 30 lowest $l^2$ distancecs and using uniform distribution a picture is randomly selected.
