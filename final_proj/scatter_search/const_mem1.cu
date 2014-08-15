/* 
   Compile with nvcc -gencode arch=compute_30,code=sm_30 const_mem1.cu 
 */

#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

#define numStreams 3
#define BLOCK 256
#define BUFF_SIZE 4096
#define MAX 1000

__device__ int count_dev;
__constant__ char pattern_dev[MAX];

/*error checking from D&W*/
void errorChecking(cudaError_t err, int line) {
   if (err != cudaSuccess){
      printf(" %s in %s at line %d\n", cudaGetErrorString(err), 
            __FILE__, line);
      exit(EXIT_FAILURE);
   }
}

__global__ void search_kernel(int pattern_length, char * string, int string_length, int offset) {
   int tx = threadIdx.x;
   int idx = offset + blockDim.x * blockIdx.x + tx; 

   for(int j = 0; j < pattern_length; ++j)
      if(pattern_dev[j] ^ string[idx + j])
         return;

   atomicAdd(&count_dev,1);
}

// Utility function from K&R
void reverse(char s[]) {
   int i, j;
   char c;

   for (i = 0, j = strlen(s)-1; i<j; i++, j--) {
      c = s[i];
      s[i] = s[j];
      s[j] = c;
   }
}

// Utility function from K&R for string parsing
void itoa(int n, char s[]) {
   int i, sign;

   if ((sign = n) < 0) 
      n = -n;          
   i = 0;
   do { 
      s[i++] = n % 10 + '0';  
   } while ((n /= 10) > 0);  
   if (sign < 0)
      s[i++] = '-';
   s[i] = '\0';
   reverse(s);
}

int search(char *pattern, int pattern_length, char *string, int string_length) {
   char * string_dev;
   int count = 0;
   int streamOffset;
   cudaStream_t stream[numStreams];
   int numThreads = BLOCK;

   // Create the streams
   for( int i = 0; i < numStreams; ++ i){
      errorChecking( cudaStreamCreate(&stream[i] ), __LINE__);
   }
   int streamLength = ceil((float)string_length/numStreams);

   // Set up the threads
   dim3 dimGrid( ceil(streamLength/(float)numThreads), 1, 1);
   dim3 dimBlock(numThreads, 1, 1);

   // Allocate memory on the device
   errorChecking(cudaMalloc((void **) &string_dev, sizeof(char)*string_length), 
      __LINE__);
   errorChecking(cudaMemcpyToSymbol(count_dev, &count, sizeof(int), 0, 
      cudaMemcpyHostToDevice), __LINE__);

   // Copy the pattern to constant memory
   errorChecking(cudaMemcpyToSymbol(pattern_dev, pattern, 
      pattern_length * sizeof(char), cudaMemcpyHostToDevice), __LINE__);

   // Copy the data set to the device 
   for(int i = 0; i < numStreams; ++i){
      streamOffset = i * streamLength;
      errorChecking( cudaMemcpyAsync(&string_dev[streamOffset], 
         &string[streamOffset],  streamLength * sizeof(char), 
         cudaMemcpyHostToDevice, stream[i] ), __LINE__);
   }
   
   // Call the kernel 
   for(int i = 0; i < numStreams; ++i){
      streamOffset = i * streamLength; 
      search_kernel<<<dimGrid.x, dimBlock.x, 0, stream[i]>>>
         (pattern_length, string_dev, string_length, streamOffset);
      errorChecking(cudaGetLastError(), __LINE__);
   }    
   cudaStreamSynchronize(stream[2]); 

   // Get the results from the device
   errorChecking(cudaMemcpyFromSymbol(&count, count_dev, 
            sizeof(int), 0, cudaMemcpyDeviceToHost), __LINE__);

   // Free resources
   for(int i = 0; i < numStreams; ++i){ 
      errorChecking(cudaStreamDestroy(stream[i]), __LINE__); 
   }
   cudaFree(string_dev);
   return count;
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

// Grab the patterns from the file specified in main and write
// the results to file 
int count_keys_in_file(char *filename, char * string, int string_length) {
   FILE *infile;
   FILE *outfile;
   int length;
   char BUFFER[BUFF_SIZE];
   char TEMP[512];
   int word_count = 0;

   // initialize the pattern array
   for (int i = 0; i < BUFF_SIZE; ++i)
      BUFFER[i] = '\0';

   infile = fopen(filename, "r");
   outfile = fopen("key_value_pairs.txt", "a");

   // Get each pattern from the file
   while(fgets(BUFFER, BUFF_SIZE, infile) != NULL) {
      length = strlen(BUFFER) - 1;
      if(BUFFER[length] == '\n')
         BUFFER[length] = '\0';
      else
         ++length;

      // Call the kernel launch function and get the results 
      word_count = search(BUFFER, length, string, string_length);
      ++length;

      // Write the results to file
      BUFFER[length - 1] = ' ';
      BUFFER[length + 0] = '|';
      BUFFER[length + 1] = ' ';
      itoa(word_count, TEMP);
      strcpy(&BUFFER[length + 2], TEMP);
      BUFFER[strlen(BUFFER) + 1] = '\0';
      BUFFER[strlen(BUFFER)] = '\n';
      int blah = fputs(BUFFER, outfile);
   }

   fclose(outfile);
   fclose(infile);
   return 0; 
}

int main(void) {
   char * string;
   struct timeval start, end;
   int string_length = get_string_from_file("parsedComments.txt", &string); 

   // Start the search
   gettimeofday(&start, 0); 
   count_keys_in_file("key_list.txt", string, string_length);
   gettimeofday(&end, 0); 

   // Print the GPU time
   long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + 
      end.tv_usec-start.tv_usec;
   printf("GPU Time: %lld \n", elapsed);

   cudaFreeHost(string);
   return 0;
}
