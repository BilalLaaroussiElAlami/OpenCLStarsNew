//multiply every element of an array with a constant
#define PYBIND11_DETAILED_ERROR_MESSAGES

__kernel void multiplyByConstant(const int N, const int C, __global float *A){
  // We are going to spawn as many work items as there are items in the array n
  // Each work item is responsible for writing a single value in the result array. 
  int i = get_global_id(0);
  if (i < N){
      A[i] = C * A[i];
  }
}

//model one dimensional array
__kernel void greyScale(const int N, __global float *R, __global float *G, __global float *B, __global float *L){
  //L = 0.299 * R + 0.587 * G + 0.114 * B
  #define W_r 0.299
  #define W_g 0.587
  #define W_b 0.114
  int i = get_global_id(0);
  //if(i == 0 || i == N-1) printf("Called grayscale %d\n", i);
  if (i < N){
      L[i] = W_r * R[i] + W_g * G[i] + W_b * B[i];
  }
}

//Will be used to eventually calculate the average in the host program. This function will calculate partial sums
__kernel void sum(const int N, __global float *A, __global float *Res){
  int i = get_global_id(0); 
  //if(i == 0 || i == N-1) printf("called sum %d\n", i);
  float sum = 0.0f;
  int k;
  for(k = 0; k < N; k++){
    sum += A[i*N+k];
  }
  Res[i] = sum;
}


//Expects that every element in the array IsStars is 0. When a star i is a star we will assign IsStars[i] = 1
//if WindowSize is 3 when looking in a 7*7 square 
__kernel void identifyStars(const int Height, const int Width, const int WindowSize, const float MinBrightness, __global float *L, __global int *IsStars){
  // We are going to spawn as many work items as there are items in the array n
  int i = get_global_id(0); 
  int j = get_global_id(1);
  if( (i == 0 && j == 0) || (i == Width -1 && j == Height - 1 )){
   //printf("called identifyStars i:%d j:%d Width:%d Height:%d WindowSize:%d MinBrightness: %f\n", i, j,Width,Height,WindowSize,MinBrightness);
  } 
  if(i < 11 ) { //HERE IS A BUG
      //printf("called identifyStars i:%d j:%d Width:%d Height:%d WindowSize:%d MinBrightness: %f\n", i, j,Width,Height,WindowSize,MinBrightness);
  }
  if(i >= Height) return;
  if(j >= Width) return;
  float brightnessPixel = L[i*Width+j];
  if(brightnessPixel < MinBrightness) return;
  //edge handling technique: pixel in window that are out of bounds are ignored
  //TODO change edge handling technique to MIRRORING
  int ii;
  int jj;
  float maxBrightness = 0.0f;

  //calculate max brightness of neighbours
  for(ii = i - WindowSize; ii < i + WindowSize + 1; ii++){
      for(jj = j - WindowSize; jj < j + WindowSize + 1; jj++){
        //Check we are not out of bounds.
        if(ii < 0 || ii  >= Height || jj  < 0 || jj >= Width){
          continue; 
        }
        //Skip the pixel itself.
        if(ii == i && jj == j){
          continue; 
        }
        float brightness = L[ii*Width+jj];
        if(brightness >= maxBrightness){
          maxBrightness = brightness;
        }
      }
  }
  if(brightnessPixel >= maxBrightness){
    IsStars[i*Width+j] = 1; 
   // printf("is_star? %d %d: %d\n", i,j, IsStars[i*Width+j]);
  }
  
}




