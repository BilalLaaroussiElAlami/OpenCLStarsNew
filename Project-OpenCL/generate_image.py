from PIL import Image
import random

# Create a new blank image with a white background
image = Image.new("RGB", (10,100), "white")
pixels = image.load()

# Fill the image with random pixel colors
for i in range(image.width):
    for j in range(image.height):
        if (random.randint(0,99) < 10):
            r = random.randint(200, 255)
            g = random.randint(200, 255)
            b = random.randint(200, 255)
        else:
            r = random.randint(0, 100)
            g = random.randint(0, 100)
            b = random.randint(0, 100)
        pixels[i, j] = (r, g, b)



# Save the image as a JPG file
image.save("images/random_image.jpg", "JPEG")
