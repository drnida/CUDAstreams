/* SCRATCH CODE
Compile with nvcc -gencode arch=compute_30,code=sm_30 substringSearch.cu
*/


#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

#define numStreams 3


__device__ int count_dev;

/*error checking from D&W*/
void errorChecking(cudaError_t err, int line) {
    if (err != cudaSuccess){
        printf(" %s in %s at line %d\n", cudaGetErrorString(err), 
        __FILE__, line);
        exit(EXIT_FAILURE);
    }
}


__global__ void search_kernel(char * string, char * results, int * numbers, int length, int offset) {

    int i = offset + blockDim.x * blockIdx.x + threadIdx.x;
    int patternLength = 5;
    char pattern [6] = "hello";
    int match = 1;
    if(i < length - (patternLength-1)){
        for(int j = 0; j < 5; ++j){
           if((pattern[j] ^ string[i + j]) != 0x0000 ){
            //if( (pattern[j] == string[i + j]) ){
              match = 0; 
           //   results[j] = string[i + j];
            }
        }
    }
    else {
        match =0;
    }
    __syncthreads();
    results[i] = string[i]; 
    numbers[i] = i; // match;
    //atomicAdd(&count_dev, match);

   if(match != 0){
       atomicAdd(&count_dev,1);
   }

}


void search(char * string, char * results, int length) {
    char * string_dev,  *result_dev;
    //int patternLength = 5;
    //cudaError_t err;
    int * count;
    int streamOffset;
    cudaStream_t stream[numStreams];
    //size_t free, free2, total;
    int * numbers; 
    int * numbers_dev;
    int numThreads = 256;

    errorChecking( cudaMallocHost( (void**) &numbers, sizeof(int) * length ), __LINE__);
    for(int i = 0; i < length; ++i){
        numbers[i] = 0;
    }

    errorChecking( cudaMallocHost( (void**) &count, sizeof(int) ), __LINE__);

    for( int i = 0; i < numStreams; ++ i){
        errorChecking( cudaStreamCreate(&stream[i] ), __LINE__);
    }
    *count = 0; 

    int streamLength = ceil(length/numStreams);
    int streamBytes = streamLength * sizeof(char);

    printf("streamLength: %d, streamBytes %d\n", streamLength, streamBytes);    


//size     
    dim3 dimGrid( ceil(streamLength/(float)numThreads), 1, 1);
    dim3 dimBlock(numThreads, 1, 1);


    printf("dimGrid.x: %d Threads: %d \n" , dimGrid.x, dimBlock.x);

    //cudaMemGetInfo(&free, &total);
    //printf("\nFree Mem:  %zu, Total Mem: %zu \n", free, total); 


    errorChecking( cudaMalloc((void **) &string_dev, sizeof(char) * length), __LINE__);
    errorChecking( cudaMalloc((void **) &result_dev, sizeof(char) * length), __LINE__);
    errorChecking( cudaMalloc((void **) &numbers_dev, sizeof(int) * length), __LINE__);


    for(int i = 0; i < numStreams; ++i){
        streamOffset = i * streamLength;
        printf("streamOffset is: %d\n", streamOffset);
        errorChecking( cudaMemcpyAsync(&numbers_dev[streamOffset], &numbers[streamOffset], streamLength * sizeof(int), cudaMemcpyHostToDevice, stream[i] ), __LINE__);
        
        errorChecking( cudaMemcpyToSymbolAsync(count_dev, count, sizeof(int), 0, cudaMemcpyHostToDevice, stream[i]), __LINE__);
    //size
        errorChecking( cudaMemcpyAsync(&string_dev[streamOffset], &string[streamOffset],  streamLength * sizeof(char), cudaMemcpyHostToDevice, stream[i] ), __LINE__);
    }    
        //cudaMemGetInfo(&free2, &total);
        //printf("Free Mem:  %zu, Total Mem: %zu \n", free2, total);
        
    for(int i = 0; i < numStreams; ++i){
        streamOffset = i * streamLength;
        search_kernel<<<dimGrid.x, dimBlock.x, 0, stream[i]>>>(string_dev, result_dev, numbers_dev, length, streamOffset);



        errorChecking(cudaGetLastError(), __LINE__);
    }    
    for(int i = 0; i < numStreams; ++i){
        streamOffset = i * streamLength;
        errorChecking(cudaMemcpyAsync(&results[streamOffset], &result_dev[streamOffset],  streamLength * sizeof(char), cudaMemcpyDeviceToHost, stream[i]), __LINE__);
        
        errorChecking(cudaMemcpyFromSymbolAsync(count, count_dev, sizeof(int), 0, cudaMemcpyDeviceToHost, stream[i]), __LINE__);
        
        errorChecking(cudaMemcpyAsync(&numbers[streamOffset], &numbers_dev[streamOffset],  streamLength * sizeof(int), cudaMemcpyDeviceToHost, stream[i]), __LINE__);
    }
    cudaStreamSynchronize(stream[2]); 
  
    printf("Count is: %d\n", *count);
  
    printf("Numbers\n");
    for(int i = 0; i < length; ++i){
        printf("%d ",numbers[i]);
    
    }
    printf("\n");

 
    for(int i = 0; i < numStreams; ++i){ 
        errorChecking(cudaStreamDestroy(stream[i]), __LINE__); 
    }
    cudaFree(string_dev);
    cudaFree(result_dev);
    cudaFreeHost(count);
    cudaFreeHost(numbers);  
}



int main(void) {
    int length = 1024;
    //int patternLength = 6;

    struct timeval start, end;
    char * string;
    char * results;
    errorChecking( cudaMallocHost((void**) &string, length * sizeof(char)), __LINE__ );
    errorChecking( cudaMallocHost((void**) &results, length * sizeof(char)), __LINE__ );



 
     for(int i = 0; i < length-1; ++i){
        string[i] = 'a'; 
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
   string[1022] = 'b';

   printf("String is: %s\n",string);







   
    


    gettimeofday(&start, 0); 
    search(string, results, length);
    gettimeofday(&end, 0); 

    long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
    printf("GPU Time: %lld \n", elapsed);


    results[length-1] = '\0';
    printf("results: %s\n", results);



    cudaFreeHost(string);
    cudaFreeHost(results);


    return 0;
}
