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

__global__ void search_kernel(char * string, int length, int offset) {
    int tx = threadIdx.x;
    int idx = offset + blockDim.x * blockIdx.x + tx; 
    int match = 0; 

    int patternLength = 5;
    char pattern[6] = "hello";

    // This data can't match since there isn't enough room for the full pattern
    // at the end of the array 
    if(idx >= length - patternLength - 1) {
       return;
    }

    for(int j = 0; j < patternLength; ++j){
       if((pattern[j] ^ string[idx + j]) == 0x0000 ){
          match += 1; 
       }
    }

    if(match == patternLength) {
       atomicAdd(&count_dev,1);
    }
}

void search(char * string, int length) {
    char * string_dev;
    int count = 0;
    int streamOffset;
    cudaStream_t stream[numStreams];
    int numThreads = BLOCK;

    for( int i = 0; i < numStreams; ++ i){
        errorChecking( cudaStreamCreate(&stream[i] ), __LINE__);
    }
    
    
    int streamLength = ceil((float)length/numStreams);
    int streamBytes = streamLength * sizeof(char);

    printf("streamLength: %d, streamBytes %d\n", streamLength, streamBytes);    

    dim3 dimGrid( ceil(streamLength/(float)numThreads), 1, 1);
    dim3 dimBlock(numThreads, 1, 1);

    printf("dimGrid.x: %d Threads: %d \n" , dimGrid.x, dimBlock.x);

    errorChecking( cudaMalloc((void **) &string_dev, sizeof(char) * length), 
       __LINE__);

    errorChecking( cudaMemcpyToSymbol(count_dev, &count, 
         sizeof(int), 0, cudaMemcpyHostToDevice), __LINE__);
    

    for(int i = 0; i < numStreams; ++i){
        streamOffset = i * streamLength;
        printf("streamOffset is: %d\n", streamOffset);

        errorChecking( cudaMemcpyAsync(&string_dev[streamOffset], 
            &string[streamOffset],  streamLength * sizeof(char), 
            cudaMemcpyHostToDevice, stream[i] ), __LINE__);
    }    
        
    for(int i = 0; i < numStreams; ++i){
        streamOffset = i * streamLength;
        search_kernel<<<dimGrid.x, dimBlock.x, 0, stream[i]>>>(string_dev, 
           length, streamOffset);

        errorChecking(cudaGetLastError(), __LINE__);
    }    

    cudaStreamSynchronize(stream[2]); 

    errorChecking(cudaMemcpyFromSymbol(&count, count_dev, 
        sizeof(int), 0, cudaMemcpyDeviceToHost), __LINE__);
        
    printf("Count is: %d\n", count);
  
    for(int i = 0; i < numStreams; ++i){ 
        errorChecking(cudaStreamDestroy(stream[i]), __LINE__); 
    }
    cudaFree(string_dev);
}

// Grab data from external file
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
    length = get_string_from_file("UnicodeSample.txt", &string); 

    printf("Padded length is: %d bytes.\n", length);

    gettimeofday(&start, 0); 
    search(string, length);
    gettimeofday(&end, 0); 

    long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
    printf("GPU Time: %lld \n", elapsed);


    cudaFreeHost(string);
    return 0;
}
