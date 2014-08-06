/* 
 * Kristina Frye
 * Dean Nida
 * John Schroeder
 *
 * Compile with 
 * nvcc -gencode arch=compute_30,code=sm_30 substringSearch.cu
 */


#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

#define numStreams 3
#define BLOCK 1024 

__device__ int count_dev;

/*error checking from D&W*/
void errorChecking(cudaError_t err, int line) {
    if (err != cudaSuccess){
        printf(" %s in %s at line %d\n", cudaGetErrorString(err), 
        __FILE__, line);
        exit(EXIT_FAILURE);
    }
}

__global__ void search_kernel(char *string, int length, int offset, 
        char *pattern, int patternLength) {

    int tx = threadIdx.x;
    int idx = offset + blockDim.x * blockIdx.x + tx; 

    // dynamically allocated shared memory
    extern __shared__ char shared[];
    char *string_sh = &shared[0]; // size BLOCK + patternLength for halo data
    char *pattern_sh = &shared[BLOCK + patternLength]; 

    // These threads load the pattern into shared plus the halo data

    if(tx < patternLength) {
        pattern_sh[tx] = pattern[tx];
        shared[BLOCK + tx] = string[idx + BLOCK];
    }
    string_sh[tx] = string[idx]; 
    __syncthreads();

    // This data can't match since there isn't enough room for the full pattern
    // at the end of the array 
    if(idx >= length - patternLength - 1) {
       return;
    }

    for(int j = 0; j < patternLength; ++j){
       if((pattern_sh[j] ^ string_sh[tx + j]) != 0x0000 ) {
          return;
       }
    }

    atomicAdd(&count_dev, 1);
}

void search(char * string, int length, char *pattern, int patternLength) {
    char * string_dev, *pattern_dev;
    int count = 0;
    int streamOffset;
    cudaStream_t stream[numStreams];
    int numThreads = BLOCK;

    // Create the streams
    for( int i = 0; i < numStreams; ++ i){
        errorChecking( cudaStreamCreate(&stream[i] ), __LINE__);
    }
    
    // Determine stream size based upon number of streams and input data 
    int streamLength = ceil((float)length/numStreams);
    int streamBytes = streamLength * sizeof(char);
    //printf("streamLength: %d, streamBytes %d\n", streamLength, streamBytes);

    // Determine size of the grid and blocks 
    dim3 dimGrid( ceil(streamLength/(float)numThreads), 1, 1);
    dim3 dimBlock(numThreads, 1, 1);
    //printf("dimGrid.x: %d Threads: %d \n" , dimGrid.x, dimBlock.x);

    // Malloc memory on the device
    errorChecking( cudaMalloc((void **) &string_dev, sizeof(char) * length), 
        __LINE__);
    errorChecking( cudaMalloc((void **) &pattern_dev, 
        sizeof(char) * patternLength + 1), __LINE__);

    // Copy data to the device (streams independent)
    errorChecking( cudaMemcpyToSymbol(count_dev, &count, 
        sizeof(int), 0, cudaMemcpyHostToDevice), __LINE__);
    errorChecking( cudaMemcpy(pattern_dev, pattern, 
        patternLength + 1 * sizeof(char), cudaMemcpyHostToDevice), __LINE__);   

    // Copy streams data to the device  
    for(int i = 0; i < numStreams; ++i) {
        streamOffset = i * streamLength;
        // printf("streamOffset is: %d\n", streamOffset);

        errorChecking( cudaMemcpyAsync(&string_dev[streamOffset], 
            &string[streamOffset],  streamLength * sizeof(char), 
            cudaMemcpyHostToDevice, stream[i] ), __LINE__);
    }    
    
    // Run the kernel with streams    
    for(int i = 0; i < numStreams; ++i) {
        streamOffset = i * streamLength;
        // sharedMem stores the lengths used in the kernel for shared memory
        int sharedMem = (BLOCK+patternLength) * sizeof(char)+patternLength * sizeof(char);
        search_kernel<<<dimGrid.x, dimBlock.x, sharedMem, 
            stream[i]>>>(string_dev, length, streamOffset, pattern, 
            patternLength);

        errorChecking(cudaGetLastError(), __LINE__);
    }    

    cudaStreamSynchronize(stream[2]); 
    
    errorChecking(cudaMemcpyFromSymbol(&count, count_dev, 
        sizeof(int), 0, cudaMemcpyDeviceToHost), __LINE__);
    
    //count += boundary_check(string, pattern, patternLength, streamLength);    
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

// Generate simple string (used for test purposes)
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

    int midStream = ceil((float)length/numStreams) - 3;
    printf("Midstream: %d\n", midStream);
    (*input)[midStream] = 'h';
    (*input)[midStream+1] = 'e';
    (*input)[midStream+2] = 'l';
    (*input)[midStream+3] = 'l'; 
    (*input)[midStream+4] = 'o';

    return paddedLength; 
}

// Generate simple pattern
int get_pattern(char **pattern) {
    int patternLength = 5;    
    errorChecking( cudaMallocHost((void**) pattern, (patternLength + 1) * sizeof(char)),
        __LINE__ );
    (*pattern)[0] = 'h';
    (*pattern)[1] = 'e';
    (*pattern)[2] = 'l';
    (*pattern)[3] = 'l';
    (*pattern)[4] = 'o';
    (*pattern)[5] = '\0';

    return patternLength;
}

int main(void) {
    int length = 1024;
    char *string, *pattern;
    int patternLength;
    struct timeval start, end, diff;
    //length = generate_string(length, &string);
    length = get_string_from_file("../DATA/UnicodeSample.txt", &string); 
    patternLength = get_pattern(&pattern);
    printf("Pattern is: %s\n", pattern); 
    printf("Padded length is: %d bytes.\n", length);
    //printf("Input is: %s.\n", string);

    gettimeofday(&start, 0); 
    search(string, length, pattern, patternLength);
    gettimeofday(&end, 0); 
    //timersub(&start, &end, &diff);
    long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
    printf("GPU Time: %lld \n", elapsed);
    //printf("GPU Time (streams): %ld (msecs) \n", diff.tv_usec);

    cudaFreeHost(string);
    cudaFreeHost(pattern);
    return 0;
}
