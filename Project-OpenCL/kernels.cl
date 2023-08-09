#define PYBIND11_DETAILED_ERROR_MESSAGES

//multiply every element of an array with a constant
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
    int i = get_global_id(0); //i is the row i'th rown of the array
    //if(i == 0 || i == N-1) printf("called sum %d\n", i);
    float sum = 0.0f;
    int k;
    for(k = 0; k < N; k++){
      sum += A[i*N+k];
    }
    Res[i] = sum;
  }

//if the row is out of bounds, returns the mirrored index else returns the original
int correct_row_index(int row, int height){
  if(row < 0){
    row = row*-1;
  }
  if(row >= height){
    int spilledover = row - height;
    row = height - 1 - spilledover;
  }
  return row;
}
//if the column is out of bounds, returns the mirrored index else returns the original 
int correct_col_index(int col, int width){
  if(col < 0){
    col = col*-1;
  }
  if(col >= width){
    int spilledover = col - width;
    col = width - 1 - spilledover;
  }
  return col;
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

  if(i >= Height) return;
  if(j >= Width) return;
  float brightnessPixel = L[i*Width+j];
  if(brightnessPixel < MinBrightness) return;
  int row;
  int col;
  float maxBrightness = 0.0f;

  //calculate max brightness of neighbours
  for(row = i - WindowSize; row < i + WindowSize + 1; row++){
      for(col = j - WindowSize; col < j + WindowSize + 1; col++){
        /*
        //Check we are not out of bounds.
        if(row < 0 || row  >= Height || col  < 0 || col >= Width){
          continue; 
        }*/
        //Skip the pixel itself.
        if(row == i && col == j){
          continue; 
        }
        int correct_row = correct_row_index(row, Height);
        int correct_col = correct_col_index(col, Width);
        float brightness = L[correct_row*Width+correct_col];
        if(brightness > maxBrightness){
          maxBrightness = brightness;
        }
      }
  }
  if(brightnessPixel >= maxBrightness){  
    IsStars[i*Width+j] = 1; 
  }
  
}




