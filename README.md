# Mozaik
App replace pixels on selected picture by other images.
Pictures are loaded from the disc, fillers should be is dedicated directory.
It allows to show the output, save it, rescale either the main picture and unoformly standardise the size of images-pixels (i.e. images that will replace pixels).
Output image admits several strategies of selection pixel-image. Below stategies and samaple results:
1) `straight`: number of pixels of the main image == number of images you want to replace its pixels by.

2) `duplication`: number of replacing image is less than total number of pixels of the original image. Images are duplicated to cover number of pixels in the original photo.
![](https://github.com/dariusz-piekarz/Mozaik/blob/master/duplication.png)
3) `pixel_mean`: To each image is assign its mean RGB and thenit is searched for each pixel an image that is the closest to this pixel with respec to the $\mathcal{l}^2$ distance.
![](https://github.com/dariusz-piekarz/Mozaik/blob/master/pixel_mean.png)
4)  `pixel_mean_random`: As in the case 3), mean RGB is calculated for each picture. Then it is searached for 30 lowest $l^2$ distancecs and using uniform distribution a picture is randomly selected.
![](https://github.com/dariusz-piekarz/Mozaik/blob/master/pixel_mean_random.png)
