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

__global__ void search_kernel(char * string, int * results, int length) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int patternLength = 5;
    char pattern [6] = "hello";
    int letters = 0;
    if(i < length - patternLength){
        for(int j = 0; j < 5; ++j){
            if((pattern[j] ^ string[i + j]) == 0x0000 ){
                letters++; 
            }
        }
    } 
   if(letters == patternLength){
       results[i] = 1; 
   }
}

void search(char * string, int * results, int length) {
    char * string_dev;
    int *result_dev;

    size_t free, free2, total;
    cudaMemGetInfo(&free, &total);
    printf("\nFree Mem:  %zu, Total Mem: %zu \n", free, total); 

    errorChecking(cudaMalloc((void **) &string_dev, sizeof(char) * length));
    errorChecking(cudaMalloc((void **) &result_dev, sizeof(int) * length));
 
    errorChecking(cudaMemcpy(string_dev, string, sizeof(char) * length, cudaMemcpyHostToDevice));
    errorChecking(cudaMemcpy(result_dev, results, sizeof(int) * length, cudaMemcpyHostToDevice));

    cudaMemGetInfo(&free2, &total);
    printf("Free Mem:  %zu, Total Mem: %zu \n", free2, total);

    dim3 dimGrid((length + 1024 - 1) / 1024, 1, 1);
    dim3 dimBlock(1024, 1, 1);

    printf("dimGrid.x: %d Threads: %d \n" , dimGrid.x, dimGrid.x * 1024);
    search_kernel<<<dimGrid, dimBlock>>>(string_dev, result_dev, length);
    cudaThreadSynchronize(); 
    errorChecking(cudaGetLastError());
    errorChecking(cudaMemcpy(results, result_dev, sizeof(int) * length, cudaMemcpyDeviceToHost));

    cudaFree(string_dev);
    cudaFree(result_dev);
}

int count_array(int *arr, int length) {
    int count = 0;
    for(int i = 0; i < length; i++)  {
        if(arr[i] == 1)  
            count++;
    }
    return count;
}

int main(void) {
    int length = 1000;

    struct timeval start, end;
    char * string;
    int * results;
    string = (char * ) malloc(length * sizeof(int));
    results = (int *) malloc(length * sizeof(int));
 
    for(int i = 0; i < length-1; ++i){
       string[i] = 'a';
       results[i] = 0; 
   }  
   string[length-1] = '\0';
   
   string[4] = 'h'; 
   string[5] = 'e'; 
   string[6] = 'l'; 
   string[7] = 'l'; 
   string[8] = 'o'; 

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
    printf("There were %d total matches.\n", count_array(results, length));

    free(string);
    free(results);
    return 0;
}
