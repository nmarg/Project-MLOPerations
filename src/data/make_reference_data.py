from csv import writer
from pathlib import Path

import numpy as np
from PIL import Image, ImageStat


def calculate_brightness(image):
    histogram = image.histogram()
    pixels = sum(histogram)
    brightness = scale = len(histogram)

    for index in range(0, scale):
        ratio = histogram[index] / pixels
        brightness += ratio * -scale + index

    return brightness / scale


def calculate_contrast(image):
    stat = ImageStat.Stat(image)
    return stat.stddev[0]


def calculate_sharpness(image):
    image_array = np.array(image)
    gy, gx = np.gradient(image_array)
    gnorm = np.sqrt(gx**2 + gy**2)
    sharpness = np.average(gnorm)
    return sharpness


def calculate_image_params(image):
    greyscale_image = image.convert("L")
    brightness = calculate_brightness(greyscale_image)
    contrast = calculate_contrast(greyscale_image)
    sharpness = calculate_sharpness(greyscale_image)

    return brightness, contrast, sharpness


if __name__ == "__main__":
    images_folder = "data/testing/images/"
    images = sorted(Path(images_folder).glob("*.jpg"))
    labels = np.genfromtxt(
        "data/testing/labels.csv",
        delimiter=",",
    )
    labels = labels[:5000].astype(int)

    if len(labels) != len(images):
        raise Exception

    N = len(labels)

    with open("data/drifting/reference_data.csv", "a") as f_object:
        for i in range(N):
            image = Image.open(images[i])
            image_stats = calculate_image_params(image)
            csv_row = list(image_stats)
            csv_row.append(labels[i])

            writer_object = writer(f_object)
            writer_object.writerow(csv_row)
            print("Saved: ", i, "/", N, " images.")

        f_object.close()
