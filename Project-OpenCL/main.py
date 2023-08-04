import math
from xxlimited import new
from PIL import Image, ImageOps
import numpy as np
import time

# A pixel is a star if its brightness is greater than THRESHOLD_FACTOR times the average
# brightness.
THRESHOLD_FACTOR = 2

# The size of the window to use when looking at a pixel's neighbors.
# This is the pixels on each side, i.e. the window is 2 * WINDOW_SIZE + 1 pixels wide.
WINDOW_SIZE = 3

IMAGES = [
    "./images/IRAS-19312-1950.jpg",  # 675x1200 = 810.000 pixels
    "./images/behemoth-black-hole.jpg",  # 2219x2243 = 4.977.217 pixels
    "./images/NGC-362.jpg",  # 2550x2250 = 5.737.500 pixels
    "./images/omega-nebula.jpg",  # 2435x3000 = 7.305.000 pixels
    "./images/andromeda-2.jpg",  # 6000x6000 = 36.000.000 pixels
    "./images/andromeda.jpg",  # 6200x6200 = 38.440.000 pixels
    "./images/cygnus-loop-nebula.jpg",  # 7000x9400 = 65.800.000 pixels
]


def load_image(path):
    """Load the image file into a PIL image."""
    return Image.open(path)


def color_to_gray(image, name="stars"):
    """Convert a color image to grayscale. This uses the formula:

    L = R * 299/1000 + G * 587/1000 + B * 114/1000

    See https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.convert
    """
    # Uncomment line below to save grayscale images to inspect them manually.
    # ImageOps.grayscale(image).save("{}.gray.png".format(name))
    return ImageOps.grayscale(image)

def calculate_indices(row,col,height,width): 
    if(row < 0):
        row = abs(row)
    if(row >= height):
        spilledover = row - height
        row = height - 1 - spilledover
    if(col < 0):
        col = abs(col)
    if(col >= width):
        spilledover = col - width
        col = width - 1 - spilledover
    return row,col            


def calculate_max_neighbors(image, row, col):
    """Calculate the maximum value of the neighbors of a pixel.

    Make sure to correctly deal with edges using one of the edge handling techniques."""
    max = 0

    for i in range(row - WINDOW_SIZE, row + WINDOW_SIZE + 1):
        for j in range(col - WINDOW_SIZE, col + WINDOW_SIZE + 1):
            # Check we are not out of bounds.
            #if (i < 0 or i >= image.shape[0]) or (j < 0 or j >= image.shape[1]):
            #    continue
            # Skip the pixel itself.
            if i == row and j == col:
                continue
            
            height = image.shape[0]
            width  = image.shape[1]
            ni,nj  = calculate_indices(i,j, height,width)
            if image[ni, nj] > max:
                max = image[ni, nj]
    return max


def is_star(image, row, col, min_brightness):
    """Check if a pixel is a star, i.e. its brightness is greater than (or equal to) the
    minimum and than the maximum brightness of its neighbors."""
    brightness = image[row, col]
    if brightness < min_brightness:
        return False
    max_brightness_neighbors = calculate_max_neighbors(image, row, col)
    return brightness >= max_brightness_neighbors


def count_stars(image, min_brightness):
    """Count the number of stars in the image."""
    star_count = 0

    for row in range(image.shape[0]):
        for col in range(image.shape[1]):
            if is_star(image, row, col, min_brightness):
                star_count += 1

    return star_count


def highlight_stars(image, min_brightness):
    """Create a new image, with the stars highlighted."""
    shape = image.shape
    star_image = np.full(shape, 255, dtype=np.uint8)

    for row in range(shape[0]):
        for col in range(shape[1]):
            if is_star(image, row, col, min_brightness):
                star_image[row, col] = 0

    return star_image


def main():
    print("Threshold factor: {}".format(THRESHOLD_FACTOR))
    print("Window size: {}".format(WINDOW_SIZE))

    for image_path in ["./images/behemoth-black-hole.jpg"]:
        print("SEQUENTIAL VERSION, image: ",image_path)
        image_name = image_path.split("/")[-1].split(".")[0]
        print()
        print("Image: {} ({})".format(image_name, image_path))

        # Load image
        img = load_image(image_path)

        start_time = time.perf_counter()

        # 1. Convert image to grayscale
        img_gray = color_to_gray(img)

        # Convert image to numpy array
        # For OpenCL, you can flatten the image to a 1D array using numpy_array.flatten()
        img_arr = np.asarray(img_gray).astype(np.uint8)
        print("Image size: {}, {} pixels".format(img_arr.shape, img_arr.size))

        # 2. Compute the threshold brightness based on the average brightness of the
        # image (rounded down)
        averageBrightness = np.average(img_arr)
        print("averageBrightnes: ", averageBrightness)
        threshold = int(THRESHOLD_FACTOR * np.average(img_arr))
        if threshold > 255:
            threshold = 255
        print("Threshold brightness: {}".format(threshold))

        # 3. Count the number of stars in the image
        star_count = count_stars(img_arr, threshold)

        end_time = time.perf_counter()

        # 4. Generate image with the stars highlighted
        # This might be useful for debugging.
        star_image = highlight_stars(img_arr, threshold)
        Image.fromarray(star_image).save("results/Sequential{}.png".format(image_name))
        # Note: even though the input is JPG, it is better to save the output as PNG,
        # to avoid compression artifacts.

        # Display results
        print("Star count: {}".format(star_count))
        print("Execution time: {:.4f}s".format(end_time - start_time))


if __name__ == "__main__":
    main()
