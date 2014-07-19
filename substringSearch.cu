/* SCRATCH CODE
Compile with nvcc -gencode arch=compute_30,code=sm_30 substringSearch.cu
*/


#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

#define numStreams 3


/*error checking from D&W*/
void errorChecking(cudaError_t err) {
    if (err != cudaSuccess){
        printf(" %s in %s at line %d\n", cudaGetErrorString(err), 
        __FILE__, __LINE__);
        exit(EXIT_FAILURE);
    }
}


__global__ void search_kernel(char * string, char * results, int length) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int patternLength = 5;
    char pattern [6] = "hello";
    __shared__ int count;
    count = 0;
    int match = 1;
    if(i < length - patternLength){
        for(int j = 0; j < 5; ++j){
            if((pattern[j] ^ string[i + j]) == 0x0000 ){
               results[j] = string[i + j];
            }
        }
    } 
/*   __syncthreads();
 
   if(match){
       ++count;
   }

   __syncthreads();
  if( i == 0 and count == 1){
       results[0] = 'a'; results[1] ='\0';
   }

   if(i == 0 and count == 2){
       results[0] = 'b'; results[1] ='\0';
   }
*/

}


void search(char * string, char * results, int length) {
    char * string_dev,  *result_dev;
    int patternLength = 5;
    cudaError_t err;

    size_t free, free2, total;
    cudaMemGetInfo(&free, &total);
    printf("\nFree Mem:  %zu, Total Mem: %zu \n", free, total); 

    errorChecking(   cudaMalloc((void **) &string_dev, sizeof(char) * length));
    errorChecking(cudaMalloc((void **) &result_dev, sizeof(char) * patternLength));
 
    errorChecking(cudaMemcpy(string_dev, string, sizeof(char) * length, cudaMemcpyHostToDevice));
    
    cudaMemGetInfo(&free2, &total);
    printf("Free Mem:  %zu, Total Mem: %zu \n", free2, total);


    dim3 dimGrid((length + 1024 - 1) / 1024, 1, 1);
    dim3 dimBlock(1024, 1, 1);


    printf("dimGrid.x: %d Threads: %d \n" , dimGrid.x, dimGrid.x * 1024);
    search_kernel<<<dimGrid, dimBlock>>>(string_dev, result_dev, length);



    cudaThreadSynchronize(); 
    errorChecking(cudaGetLastError());
    
    errorChecking(cudaMemcpy(results, result_dev, sizeof(char) * patternLength, cudaMemcpyDeviceToHost));

    cudaFree(string_dev);
    cudaFree(result_dev);
}



int main(void) {
    int length = 1000;
    int matchat = 0;
    int patternLength = 6;

    struct timeval start, end;
    char * string;
    char * results;
    string = (char * ) malloc(length * sizeof(int));
    results = (char * ) malloc(patternLength * sizeof(int));



 
     for(int i = 0; i < length-1; ++i){
        string[i] = 'a'; 
    }   
   string[length-1] = '\0';
  /* 
   string[4] = 'h'; 
   string[5] = 'e'; 
   string[6] = 'l'; 
   string[7] = 'l'; 
   string[8] = 'o'; 
*/
   string[104] = 'h'; 
   string[105] = 'e'; 
   string[106] = 'l'; 
   string[107] = 'l'; 
   string[108] = 'o'; 


   printf("String is: %s",string);







   
    


    gettimeofday(&start, 0); 
    search(string, results, length);
    gettimeofday(&end, 0); 

    long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
    printf("GPU Time: %lld \n", elapsed);


    results[5] = '\0';
    printf("results: %s\n", results);


free(string);



    return 0;
}
