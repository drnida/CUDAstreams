/* SCRATCH CODE
Compile with nvcc -gencode arch=compute_30,code=sm_30 substringSearch.cu
*/


#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

#define numStreams 3
#define BLOCK 256

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
    int tx = threadIdx.x;
    int idx = offset + blockDim.x * blockIdx.x + tx; 
    int match = 0; 

    int patternLength = 5;
    char pattern[6] = "hello";

    // This data can't match since there isn't enough room for the full pattern
    // at the end of the array 
    if(idx >= length - patternLength - 1) {
       results[idx] = string[idx];
       return;
    }

    for(int j = 0; j < patternLength; ++j){
       if((pattern[j] ^ string[idx + j]) == 0x0000 ){
          match += 1; 
       }
    }
    
    results[idx] = string[idx];

    if(match == patternLength) {
       atomicAdd(&count_dev,1);
       numbers[idx] = 1;
    }
}

void search(char * string, char * results, int length) {
    char * string_dev,  *result_dev;
    //int patternLength = 5;
    //cudaError_t err;
    //int * count;
    int * count;
    int streamOffset;
    cudaStream_t stream[numStreams];
    //size_t free, free2, total;
    int * numbers; 
    int * numbers_dev;
    int numThreads = BLOCK;

    errorChecking( cudaMallocHost( (void**) &numbers, 
        sizeof(int) * length ), __LINE__);
    for(int i = 0; i < length; ++i){
        numbers[i] = 0;
    }

    
    errorChecking( cudaMallocHost( (void**) &count, sizeof(int) * numStreams ), __LINE__);
    for( int i = 0; i < numStreams; ++ i){
        errorChecking( cudaStreamCreate(&stream[i] ), __LINE__);
        count[i] = 0;
    }
    
    
    int streamLength = ceil((float)length/numStreams);
    int streamBytes = streamLength * sizeof(char);

    printf("streamLength: %d, streamBytes %d\n", streamLength, streamBytes);    

    dim3 dimGrid( ceil(streamLength/(float)numThreads), 1, 1);
    dim3 dimBlock(numThreads, 1, 1);

    printf("dimGrid.x: %d Threads: %d \n" , dimGrid.x, dimBlock.x);

    //cudaMemGetInfo(&free, &total);
    //printf("\nFree Mem:  %zu, Total Mem: %zu \n", free, total); 

    errorChecking( cudaMalloc((void **) &string_dev, sizeof(char) * length), 
       __LINE__);
    errorChecking( cudaMalloc((void **) &result_dev, sizeof(char) * length), 
       __LINE__);
    errorChecking( cudaMalloc((void **) &numbers_dev, sizeof(int) * length), 
       __LINE__);

    for(int i = 0; i < numStreams; ++i){
        streamOffset = i * streamLength;
        printf("streamOffset is: %d\n", streamOffset);

        errorChecking( cudaMemcpyAsync(&numbers_dev[streamOffset], 
            &numbers[streamOffset], streamLength * sizeof(int), 
            cudaMemcpyHostToDevice, stream[i] ), __LINE__);
        
        errorChecking( cudaMemcpyToSymbolAsync(count_dev, count, 
            sizeof(int), 0, cudaMemcpyHostToDevice, stream[i]), __LINE__);
    
        errorChecking( cudaMemcpyAsync(&string_dev[streamOffset], 
            &string[streamOffset],  streamLength * sizeof(char), 
            cudaMemcpyHostToDevice, stream[i] ), __LINE__);
    }    
        //cudaMemGetInfo(&free2, &total);
        //printf("Free Mem:  %zu, Total Mem: %zu \n", free2, total);
        
    for(int i = 0; i < numStreams; ++i){
        streamOffset = i * streamLength;
        search_kernel<<<dimGrid.x, dimBlock.x, 0, stream[i]>>>(string_dev, 
           result_dev, numbers_dev, length, streamOffset);

        errorChecking(cudaGetLastError(), __LINE__);
    }    

    for(int i = 0; i < numStreams; ++i){
        streamOffset = i * streamLength;
        errorChecking(cudaMemcpyAsync(&results[streamOffset], 
           &result_dev[streamOffset],  streamLength * sizeof(char), 
           cudaMemcpyDeviceToHost, stream[i]), __LINE__);
        
        errorChecking(cudaMemcpyFromSymbolAsync(&count[i], count_dev, 
           sizeof(int), 0, cudaMemcpyDeviceToHost, stream[i]), __LINE__);
        
        errorChecking(cudaMemcpyAsync(&numbers[streamOffset], 
           &numbers_dev[streamOffset],  streamLength * sizeof(int), 
           cudaMemcpyDeviceToHost, stream[i]), __LINE__);
    }
    cudaStreamSynchronize(stream[2]); 
  
    
    int total = 0;
    int arrayTotal = 0; 
    for(int i =0; i < length; ++i){
        arrayTotal += numbers[i];
    }
    
    for(int i =0; i < numStreams; ++i){
        total += count[i];
    }
    
    printf("Count is: %d\nArray total is: %d\n", total, arrayTotal);
  
   /* printf("Numbers\n");
    for(int i = 0; i < length; ++i){
        printf("%d ",numbers[i]);
    
    }
    printf("\n");
*/
 
    for(int i = 0; i < numStreams; ++i){ 
        errorChecking(cudaStreamDestroy(stream[i]), __LINE__); 
    }
    cudaFree(string_dev);
    cudaFree(result_dev);
    cudaFreeHost(count);
    cudaFreeHost(numbers);  
}

// Grab data from exterior file
int get_string_from_file(char *filename, char **input) {
    FILE *file;
    int length;
    size_t result;

    file = fopen(filename, "r");
    fseek(file, 0, SEEK_END);
    length = ftell(file);
    printf("File Length is: %d bytes.\n", length);
    int paddedLength = numStreams * ceil((float)length/numStreams);
    rewind(file);

    errorChecking( cudaMallocHost((void**) input, paddedLength * sizeof(char)), 
        __LINE__ );
    if(*input == NULL) {
        fputs("Memory error", stderr);
        exit(2);
    }

    result = fread(*input, 1, length, file);
    if(result != length) {
        fputs("Reading error", stderr);
        exit(3);
    }
    fclose(file);
    if(paddedLength > length) {
        for(int i = length - 1; i < paddedLength - 1; i++) {
            (*input)[i] = 0;
        }
        (*input)[paddedLength - 1] = '\0';
    }
    return paddedLength; 
}

// Generate simple string
int generate_string(int length, char **input) {
    int paddedLength = numStreams * ceil((float)length/numStreams);
    errorChecking( cudaMallocHost((void**) input, paddedLength * sizeof(char)), 
        __LINE__ );
    if(*input == NULL) {
        fputs("Memory error", stderr);
        exit(2);
    }
    for(int i = 0; i < paddedLength - 2; i++) {
        (*input)[i] = 'a';
    }
    (*input)[paddedLength - 1] = '\0';
    return paddedLength; 
}

int main(void) {
    int length = 1024;
    char * string;
    struct timeval start, end;
    char * results;

    //length = generate_string(length, &string); 
    length = get_string_from_file("UnicodeSample.txt", &string); 
    errorChecking( cudaMallocHost((void**) &results, length * sizeof(char)), 
        __LINE__ );

    printf("Padded length is: %d bytes.\n", length);
/*
    string[4] = 'h'; 
    string[5] = 'e'; 
    string[6] = 'l'; 
    string[7] = 'l'; 
    string[8] = 'o'; 
   
    string[554] = 'h';
    string[555] = 'e';
    string[556] = 'l';
    string[557] = 'l';
    string[558] = 'o';
 
    string[1004] = 'h'; 
    string[1005] = 'e'; 
    string[1006] = 'l'; 
    string[1007] = 'l'; 
    string[1008] = 'o'; 
    string[1022] = 'b';
*/
  //  printf("String is: %s\n",string);

    gettimeofday(&start, 0); 
    search(string, results, length);
    gettimeofday(&end, 0); 

    long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
    printf("GPU Time: %lld \n", elapsed);

    results[length-1] = '\0';
    //printf("results: %s\n", results);

    cudaFreeHost(string);
    cudaFreeHost(results);
    return 0;
}
