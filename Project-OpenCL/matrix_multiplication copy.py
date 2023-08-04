#!/usr/bin/env python3
import pyopencl as cl
import numpy as np
import os

# Get kernel source code from file.
kernel_file = os.path.join(os.path.dirname(os.path.realpath(__file__)), "matmul.cl")
kernel = open(kernel_file).read()

# The size of the matrices to be added together.
MATRIX_SIZE = 100
MATRIX_ORDER = MATRIX_SIZE * MATRIX_SIZE
TOLERANCE = 0.001

# Create context, queue, and program.
context = cl.create_some_context()
queue = cl.CommandQueue(context)
program = cl.Program(context, kernel).build()

# Create the input matrixes A and B.
# A matrix is represented as a flat array with is size * size in length.
# We fill the array with 1s, which should mean the result is an array with all 3's.
h_a = np.ones(MATRIX_ORDER).astype(np.float32)
h_b = np.ones(MATRIX_ORDER).astype(np.float32)

# Create the result matrix.
# We fill them with 0s to make sure there are no memory remnants left.
h_c = np.zeros(MATRIX_ORDER).astype(np.float32)

# We also create a sequential result matrix to compare the results with.
h_seq = np.zeros(MATRIX_ORDER).astype(np.float32)

# Send the data to the guest memory.
# Again, what counts is here the memory flags you put in place. Read only because they are input data.
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

# Create the memory on the device to put the result into.
# Write only memory!
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)

# Initiate the kernel.
matmul = program.matmul
matmul.set_scalar_arg_dtypes([np.int32, None, None, None])

# Execute C = A * B.
matmul(queue, (MATRIX_SIZE, MATRIX_SIZE), None, MATRIX_SIZE, d_a, d_b, d_c)

# Wait for the queue to be completely processed.
queue.finish()

# Read the array from the device.
cl.enqueue_copy(queue, h_c, d_c)

# Multiplies two matrices a and b and stores the result in c. The matrices are
# represented as flat arrays with N * N elements.
def matrix_multiplication(N, a, b, c):
    for i in range(N):
        for j in range(N):
            t = 0.0
            for k in range(N):
                t += a[i * N + k] * b[k * N + j]
            c[i * N + j] = t
    return c

# Calculate the sequential result.
matrix_multiplication(MATRIX_SIZE, h_a, h_b, h_seq)

correct = 0
for i in range(MATRIX_ORDER):
    expected = h_seq[i]
    actual = h_c[i]
    relative_error = np.absolute((actual - expected) / expected)

    # Print the index if it's wrong.
    if relative_error < TOLERANCE:
        correct += 1
    else:
        print(i, "is wrong")

print("Correct:", correct, "/", MATRIX_ORDER)

print("h_c: ", h_c)
