/* 
   Compile with nvcc -gencode arch=compute_30,code=sm_30 morePerTwit.cu 
 */

#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

#define numStreams 4
#define BLOCK 256
#define BUFF_SIZE 4096

__device__ int count_dev[numStreams];

/*error checking from D&W*/
void errorChecking(cudaError_t err, int line) {
   if (err != cudaSuccess){
      printf(" %s in %s at line %d\n", cudaGetErrorString(err), 
            __FILE__, line);
      exit(EXIT_FAILURE);
   }
}

// CUDA kernel
__global__ void search_kernel(char * pattern, int pattern_length, char * string, int string_length, int streamNum) {
   int tx = threadIdx.x;
   int idx = blockDim.x * blockIdx.x + tx; 
   extern __shared__ char pattern_sh[];

   if(tx < pattern_length) {
      pattern_sh[tx] = pattern[tx];
   }
   __syncthreads();

   for(int j = 0; j < pattern_length; ++j)
      if(pattern_sh[j] ^ string[idx + j])
         return;

   atomicAdd(&count_dev[streamNum],1);
}

// From Kernighan and Ritchie, The C Programming Lanugage
void reverse(char s[])
{
   int i, j;
   char c;

   for (i = 0, j = strlen(s)-1; i<j; i++, j--) {
      c = s[i];
      s[i] = s[j];
      s[j] = c;
   }
}

// Utility functionfor string parsing
// From Kernighan and Ritchie, The C Programming Lanugage
void itoa(int n, char s[])
{
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

   printf("\n\nDEBUG: strlen(string) == %lu\n\n", strlen(*input));
   return paddedLength; 
}

// This function grabs keys from the file specified in main and writes
// the key and the int returned by search() to the key_value_pairs file
int count_keys_in_file(char *filename, char * string_dev, int string_length, 
      cudaStream_t * stream) {

   // variables
   FILE *infile;
   FILE *outfile;
   int length[numStreams];
   char BUFFER[numStreams][BUFF_SIZE];
   char TEMP[512];
   int word_count[numStreams];
   int current_string_index = 0;
   char * pattern_dev;

   // Setting up the threads and blocks
   int count = 0;
   dim3 dimGrid( ceil(string_length/1024.0), 1, 1);
   dim3 dimBlock(1024, 1, 1);

   // initalize the count
   for(int i = 0; i < numStreams; ++i) {
       word_count[i] = 0;
   }

   // initializing the input string 
   for(int j = 0 ; j < numStreams; ++j) {
       for (int i = 0; i < BUFF_SIZE; ++i) {
          BUFFER[j][i] = '\0';
       }
   }

   infile = fopen(filename, "r");
   outfile = fopen("key_value_pairs.txt", "a");

   // allocating memory on the device
   errorChecking( cudaMalloc((void **) &pattern_dev, 
      sizeof(char) * numStreams * 1024), __LINE__);

   int j;
   printf("Starting pos: %ld\n", ftell(infile));
   while( ! feof(infile) && stopme < 150){
   for( int i = 0; i < numStreams && fgets(BUFFER[i], BUFF_SIZE, infile) != NULL && stopme < 150; ++i) {
      j = i + 1;
      length[i] = strlen(BUFFER[i]) - 1;
      if(BUFFER[i][length[i]] == '\n')
         BUFFER[i][length[i]] = '\0';
      else
         ++length[i];
       
      errorChecking( cudaMemcpyToSymbolAsync(count_dev, &count, 
         sizeof(int), sizeof(int) * i, cudaMemcpyHostToDevice, stream[i]), 
         __LINE__);

      errorChecking( cudaMemcpyAsync(&pattern_dev[i*1024], BUFFER[i],  
         length[i] * sizeof(char), cudaMemcpyHostToDevice, stream[i] ), 
         __LINE__);
   }

   // Call the kernel
   for(int i = 0; i < j; ++i) {
      int sharedMem = sizeof(char) * length[i]; 
      search_kernel<<<dimGrid.x, dimBlock.x, sharedMem, stream[i]>>>
         (&pattern_dev[i*1024], length[i], string_dev, string_length, i);

       errorChecking(cudaGetLastError(), __LINE__);
   }

   // Copy the results back to the host
   for(int i = 0; i < j; ++i) {
      errorChecking(cudaMemcpyFromSymbolAsync(&word_count[i], count_dev, 
         sizeof(int), sizeof(int) * i, cudaMemcpyDeviceToHost, stream[i]), 
         __LINE__);
   }
   cudaDeviceSynchronize();

   // output the count results
   for(int i = 0; i < j  ; ++i) {
      length[i] = strlen(BUFFER[i]);
      ++length[i];
      BUFFER[i][length[i] - 1] = ' ';
      BUFFER[i][length[i] + 0] = '|';
      BUFFER[i][length[i] + 1] = ' ';
      itoa(word_count[i], TEMP);
      strcpy(&BUFFER[i][length[i] + 2], TEMP);
      BUFFER[i][strlen(BUFFER[i]) + 1] = '\0';
      BUFFER[i][strlen(BUFFER[i])] = '\n';
      int blah = fputs(BUFFER[i], outfile);
   } 
      
   fclose(outfile);
   fclose(infile);

   cudaFree(pattern_dev);
   return 0; 
}

int main(void) {
   int string_length = 1024;
   char * string;
   char * string_dev;
   struct timeval start, end;
   string_length = get_string_from_file("parsedComments.txt", &string); 

   cudaStream_t stream[numStreams];

   // Create the streams
   for( int i = 0; i < numStreams; ++ i) {
      errorChecking( cudaStreamCreate(&stream[i] ), __LINE__);
   }

   // Allocate memory for the data set and copy data set to the device
   errorChecking( cudaMalloc((void **) &string_dev, 
      sizeof(char) * string_length), __LINE__);
   errorChecking( cudaMemcpyAsync(&string_dev[0], &string[0],  
      string_length * sizeof(char), cudaMemcpyHostToDevice, stream[0] ), 
      __LINE__);

   cudaStreamSynchronize(stream[0]); 

   // Call the rest of the cuda setup code and the kernel
   gettimeofday(&start, 0); 
   count_keys_in_file("key_list.txt", string, string_length, stream);
   gettimeofday(&end, 0); 

   long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
   printf("GPU Time: %lld \n", elapsed);

   // Free resources
   for(int i = 0; i < numStreams; ++i){ 
      errorChecking(cudaStreamDestroy(stream[i]), __LINE__); 
   }
   
   cudaFree(string_dev);
   cudaFreeHost(string);
   return 0;
}
