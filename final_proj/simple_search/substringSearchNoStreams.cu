/* 
 * Compile with 
 * nvcc -gencode arch=compute_30,code=sm_30 substringSearchNoStreams.cu
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

__global__ void search_kernel(char *string, int length, char *pattern,
        int patternLength) {

    int tx = threadIdx.x;
    int idx = blockDim.x * blockIdx.x + tx; 

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

// Checks for a match where the streams overlap
int boundary_check(char *string, char *pattern, int pattern_length, int stream_length) {
    int match = 0;
    int pattern_pos, start_pos, end_pos;

    for(int i = 1; i <= numStreams; i++) {
        // There might be a valid match that includes all characters in the pattern but 
        // the last one. This is pos (stream_length - 1) - pattern_length - 1.
        // There might also be a match that includes just the first character in the
        // pattern. This match would end at pos (stream_length - 1) + pattern_length - 1 
        start_pos = stream_length * i - pattern_length - 2;
        end_pos = stream_length * i + pattern_length - 1;
        pattern_pos = 0;
        
        // if a match is found in previous for loop, break from loop.
        // The length of the interval is only long enough for a maximum of one
        // match
        if(match == pattern_length) break;
  
        for(int j = start_pos; j < end_pos; j++) {
            // Check for char match. If found, increment and continue.
            // Otherwise, reset
            if(string[j] == pattern[pattern_pos]) {
                pattern_pos++;
                match++;

                // If match is found, break from the loop
                if(match == pattern_length) break;
            }

            else {
                pattern_pos = 0;
                match = 0;
            }
        }
    }
    if(match == pattern_length) 
        return 1;
    else   
        return 0;
}

void search(char * string, int length, char *pattern, int patternLength) {
    char * string_dev, *pattern_dev;
    int count = 0;
    int numThreads = BLOCK;

    dim3 dimGrid( ceil(length/(float)numThreads), 1, 1);
    dim3 dimBlock(numThreads, 1, 1);

    printf("dimGrid.x: %d Threads: %d \n" , dimGrid.x, dimBlock.x);

    errorChecking( cudaMalloc((void **) &string_dev, sizeof(char) * length), 
        __LINE__);
    errorChecking( cudaMalloc((void **) &pattern_dev, sizeof(char) * patternLength + 1),
        __LINE__);

    errorChecking( cudaMemcpyToSymbol(count_dev, &count, 
        sizeof(int), 0, cudaMemcpyHostToDevice), __LINE__);
    errorChecking( cudaMemcpy(pattern_dev, pattern, 
        patternLength + 1 * sizeof(char), cudaMemcpyHostToDevice), __LINE__);   
    errorChecking( cudaMemcpy(string_dev, string,  length * sizeof(char), 
        cudaMemcpyHostToDevice), __LINE__);
        
    // sharedMem stores the lengths used in the kernel for shared memory
    int sharedMem = (BLOCK+patternLength) * sizeof(char)+patternLength * 
       sizeof(char);
    search_kernel<<<dimGrid.x, dimBlock.x, sharedMem >>>(string_dev, 
        length, pattern, patternLength);

    errorChecking(cudaGetLastError(), __LINE__);
    
    errorChecking(cudaMemcpyFromSymbol(&count, count_dev, 
        sizeof(int), 0, cudaMemcpyDeviceToHost), __LINE__);
    
    //count += boundary_check(string, pattern, patternLength, streamLength);    
    printf("Count is: %d\n", count);
  
    cudaFree(string_dev);
    cudaFree(pattern_dev);
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

    int midStream = ceil((float)length/numStreams) - 3;
    printf("Midstream: %d\n", midStream);
    (*input)[midStream] = 'h';
    (*input)[midStream+1] = 'e';
    (*input)[midStream+2] = 'l';
    (*input)[midStream+3] = 'l'; 
    (*input)[midStream+4] = 'o';

    return paddedLength; 
}

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
    struct timeval start, end;
    //length = generate_string(length, &string);
    length = get_string_from_file("../DATA/UnicodeSample.txt", &string); 
    patternLength = get_pattern(&pattern);
    printf("Pattern is: %s\n", pattern); 
    printf("Padded length is: %d bytes.\n", length);
    //printf("Input is: %s.\n", string);

    gettimeofday(&start, 0); 
    search(string, length, pattern, patternLength);
    gettimeofday(&end, 0); 

    long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
    printf("GPU Time: %lld \n", elapsed);

    cudaFreeHost(string);
    return 0;
}
