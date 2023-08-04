#!/usr/bin/env python3
from cgi import print_directory
from distutils.ccompiler import gen_lib_options
from pickle import GLOBAL
from re import X
import sys
from threading import Thread
from tkinter import W, Widget
import pyopencl as cl
import numpy as np
import os
from PIL import Image, ImageOps
import time
import math


# Get kernel source code from file.
kernel_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "kernels.cl")
kernel = open(kernel_file).read()


# Create context, queue, and program.
context = cl.create_some_context()
queue   = cl.CommandQueue(context)
program = cl.Program(context, kernel).build()



def highlight_stars_result(height, width, array, name):
    # Create a new blank image with a white background
    image = Image.new("RGB", (width, height), "white")
    pixels = image.load()
    #f = open("demofile2.txt", "a")
    #f.write(f"width {width} height {height}\n")
    for i in range(height):
        for j in range(width):
            if array[i*width+j] == 1:
                r,g,b = 0,0,0
            else:
                r,g,b = 255,255,255
            #f.write(f"i {i} j {j}\n") 
            pixels[j, i] = (r, g, b)

    #f.close()
    image.save(f"results/Parallel{name}.png")



def load_image(path):
    """Load the image file into a PIL image."""
    image =  Image.open(path)
    img_arr = np.asarray(image).astype(np.uint8).flatten()
    r_values = img_arr[0:len(img_arr):3]
    g_values = img_arr[1:len(img_arr):3]
    b_values = img_arr[2:len(img_arr):3]
    return [r_values, g_values, b_values]



IMAGES = [
    "./images/IRAS-19312-1950.jpg",  # 675x1200 = 810.000 pixels
    "./images/behemoth-black-hole.jpg",  # 2219x2243 = 4.977.217 pixels
    "./images/NGC-362.jpg",  # 2550x2250 = 5.737.500 pixels
    "./images/omega-nebula.jpg",  # 2435x3000 = 7.305.000 pixels
    "./images/andromeda-2.jpg",  # 6000x6000 = 36.000.000 pixels
    "./images/andromeda.jpg",  # 6200x6200 = 38.440.000 pixels
    "./images/cygnus-loop-nebula.jpg",  # 7000x9400 = 65.800.000 pixels
]

image_name = "behemoth-black-hole"
image_path = f"images/{image_name}.jpg" # 2219x2243 = 4.977.217 pixels TODO leave like this or execute program for all images?
image = load_image(image_path)


#TODO not hardcode it!
HEIGHT = 2219
WIDTH  = 2243
SIZE = WIDTH*HEIGHT
h_Rvalues     = np.ascontiguousarray(image[0]).astype(np.float32)
h_Gvalues     = np.ascontiguousarray(image[1]).astype(np.float32)
h_Bvalues     = np.ascontiguousarray(image[2]).astype(np.float32)
h_GreyValues  = np.ascontiguousarray(np.zeros(SIZE).astype(np.float32)).astype(np.float32)
h_PartialSums = np.zeros(HEIGHT).astype(np.float32)
h_Stars       = np.zeros(SIZE).astype(np.int32) 

print("array length = ", SIZE)
assert(SIZE == len(h_Rvalues) == len(h_Gvalues) == len(h_Bvalues) == len(h_GreyValues))


d_Rvalues     = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Rvalues)
d_Gvalues     = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Gvalues)
d_Bvalues     = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_Bvalues)  
d_GreyValues  = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_Rvalues.nbytes)
d_PartialSums = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_PartialSums.nbytes)
d_Stars       = cl.Buffer(context, cl.mem_flags.READ_WRITE, h_Stars.nbytes)

def main():
    start_time = time.perf_counter()
    #--------------------CALCULATING GRAYSCAL VALUES------------------------------------
    # Initiate the kernel.
    greyScale = program.greyScale 
    greyScale.set_scalar_arg_dtypes([np.int32, None, None, None, None])
    # Execute d_GreyValues = greyscale(d_Rvalues,d_Gvalues,d_Bvalues)
    greyScale(queue, (SIZE,), None, SIZE, d_Rvalues, d_Gvalues, d_Bvalues, d_GreyValues)
    queue.finish()
    #--------------------CALCULATING AVERAGE BRIGHTNESS BASED ON GREYSCAL VALUES--------
    sum = program.sum
    sum.set_scalar_arg_dtypes([np.int32, None, None])

    global_work_size = (HEIGHT, )  # 1 work item per row = WIDTH work items.
    local_work_size  = (WIDTH, )    # 16 work items per group.
    
    sum(queue, global_work_size , None, WIDTH, d_GreyValues,  d_PartialSums)
  
    cl.enqueue_copy(queue, h_PartialSums, d_PartialSums)
    queue.finish()
    averageBrightness = h_PartialSums.sum() / (SIZE)
    
    # average brightness of the sequential implementation is 18.81127836111111  
    # in here it is 18.826314666666665
    # (18.826314666666665 - 18.81127836111111) / 18.81127836111111 = 0.00079932396. 
    #This is probably due to rounding errors of floating point operations, so I can assume there is no bug in my code
    #---------------------IDENTIFYING STARS----------------------------------------------
    
    FACTOR = 2.0
    THRESHOLD = math.floor(FACTOR*averageBrightness)
    
    WINDOWSIZE = 3
    identifyStars = program.identifyStars
    identifyStars.set_scalar_arg_dtypes([np.int32, np.int32, np.int32, np.float32, None, None])
    global_work_size = (HEIGHT,WIDTH)  
    identifyStars(queue, global_work_size , None,  HEIGHT, WIDTH,  WINDOWSIZE,  THRESHOLD, d_GreyValues, d_Stars)           
    queue.finish()
    cl.enqueue_copy(queue, h_Stars, d_Stars)
    end_time = time.perf_counter()

    highlight_stars_result(HEIGHT, WIDTH, h_Stars, image_name) #show result
    print("PARALLEL VERSION, image = ", image_path)
    print("average brightness:", averageBrightness)
    print("THRESHOLD: ", THRESHOLD )
    print("stars: ", h_Stars.sum())
    print("Execution time: {:.4f}s".format(end_time - start_time))

    
main()


    





