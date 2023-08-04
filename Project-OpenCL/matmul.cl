__kernel void matmul(const int N, __global float *A, __global float *B,
                   __global float *C) {
  // We are going to spawn as many work items as there are items in the matrix
  // (n * n). Each work item is responsible for writing a single value in
  // the result matrix. This approach ensures that no work item is writing in the
  // same result as any other work item. It also means that each work item needs
  // to multiply an entire row and column from the input matrices.
  int k;
  int i = get_global_id(0);
  int j = get_global_id(1);
  float tmp = 0;
  if ((i < N) && (j < N)) {
    tmp = 0.0f;
    for (k = 0; k < N; k++) {
      tmp += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = tmp;
  }
}