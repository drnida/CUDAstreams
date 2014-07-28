/* SCRATCH CODE
   Compile with nvcc -gencode arch=compute_30,code=sm_30 substringSearch.cu
 */


#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

#define numStreams 3
#define BLOCK 256
#define BUFF_SIZE 4096

__device__ int count_dev;

/*error checking from D&W*/
void errorChecking(cudaError_t err, int line) {
   if (err != cudaSuccess){
      printf(" %s in %s at line %d\n", cudaGetErrorString(err), 
            __FILE__, line);
      exit(EXIT_FAILURE);
   }
}


//THE KERNEL: This, I think, is all the functionality we want out of the kernel.

__global__ void search_kernel(char * pattern, int pattern_length, char * string, int string_length, int offset) {
   int tx = threadIdx.x;
   int idx = offset + blockDim.x * blockIdx.x + tx; 
   extern __shared__ char shared[];
   char *string_sh = &shared[0];
   char *pattern_sh = &shared[BLOCK + pattern_length];

   if(tx < pattern_length) {
      pattern_sh[tx] = pattern[tx];
      shared[BLOCK + tx] = string[idx + BLOCK];
   }
   string_sh[tx] = string[idx];
   __syncthreads();

   for(int j = 0; j < pattern_length; ++j)
      if(pattern_sh[j] ^ string_sh[tx + j])
         return;

   atomicAdd(&count_dev,1);
}

//The following two functions need attribution, since I stole them from K&R
//Utility function for string parsing
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

//Utility functionfor string parsing
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

//THE PRIMARY WORKHORSE
//This function, with these arguments, should be callable in order to start a kernel
//I'm not sure how streams fit into this.  ie. should we allocate a static int in this
//function to determine which stream we kicked off, then increment it and block on it 
//(to prevent too many concurrent streams)?
int search(char * pattern, int pattern_length, char * string, int string_length) {
   //These two lines just debug to make sure my formatting is correct.
   static int debug = 0;
   return debug++;

   //Couldn't get this stuff to work
   char * string_dev;
   char * pattern_dev;
   int count = 0;
   int streamOffset;
   cudaStream_t stream[numStreams];
   int numThreads = BLOCK;

   for( int i = 0; i < numStreams; ++ i){
      errorChecking( cudaStreamCreate(&stream[i] ), __LINE__);
   }


   int streamLength = ceil((float)string_length/numStreams);
   int streamBytes = streamLength * sizeof(char);

   printf("streamLength: %d, streamBytes %d\n", streamLength, streamBytes);    

   dim3 dimGrid( ceil(streamLength/(float)numThreads), 1, 1);
   dim3 dimBlock(numThreads, 1, 1);

   printf("dimGrid.x: %d Threads: %d \n" , dimGrid.x, dimBlock.x);

   errorChecking( cudaMalloc((void **) &string_dev, sizeof(char) * string_length), 
         __LINE__);

   errorChecking( cudaMalloc((void **) &pattern_dev, sizeof(char) * pattern_length), 
         __LINE__);

   errorChecking( cudaMemcpyToSymbol(count_dev, &count, 
            sizeof(int), 0, cudaMemcpyHostToDevice), __LINE__);


   for(int i = 0; i < numStreams; ++i){
      streamOffset = i * streamLength;
      printf("streamOffset is: %d\n", streamOffset);

      errorChecking( cudaMemcpyAsync(pattern_dev, 
               &pattern,  pattern_length * sizeof(char), 
               cudaMemcpyHostToDevice, stream[i] ), __LINE__);

      errorChecking( cudaMemcpyAsync(&string_dev[streamOffset], 
               &string[streamOffset],  streamLength * sizeof(char), 
               cudaMemcpyHostToDevice, stream[i] ), __LINE__);

   }    
   int sharedMem = (BLOCK+pattern_length) * sizeof(char)+pattern_length * sizeof(char);
   for(int i = 0; i < numStreams; ++i){
      streamOffset = i * streamLength; 
      
      search_kernel<<<dimGrid.x, dimBlock.x, sharedMem, stream[i]>>>
         (pattern_dev, pattern_length, string_dev, string_length, streamOffset);

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

   printf("\n\nDEBUG: strlen(string) == %lu\n\n", strlen(*input));
   return paddedLength; 
}

//This (rather messy) function grabs keys from the file specified in main and writes
//the key and the int returned by search() to the key_value_pairs file
int count_keys_in_file(char *filename, char * string, int string_length) {
   FILE *infile;
   FILE *outfile;
   int length;
   //size_t result;
   //int pattern_length = 0;
   //char * pattern;
   char BUFFER[BUFF_SIZE];
   char TEMP[512];
   int word_count = 0;
   int current_string_index = 0;

   string_length = 1024;

   for (int i = 0; i < BUFF_SIZE; ++i)
      BUFFER[i] = '\0';

   infile = fopen(filename, "r");
   outfile = fopen("key_value_pairs.txt", "a");

   while(fgets(BUFFER, BUFF_SIZE, infile) != NULL)
   {
      length = strlen(BUFFER);
      word_count = search(BUFFER, length, &string[current_string_index], string_length);
      current_string_index += string_length;
      BUFFER[length - 1] = ' ';
      BUFFER[length + 0] = '|';
      BUFFER[length + 1] = ' ';
      itoa(word_count, TEMP);
      strcpy(&BUFFER[length + 2], TEMP);
      BUFFER[strlen(BUFFER) + 1] = '\0';
      BUFFER[strlen(BUFFER)] = '\n';
      fputs(BUFFER, outfile);
      //printf("\n%s%d", BUFFER, length);
   }

   fclose(outfile);
   fclose(infile);

   return 0; 
}

//I'm not sure what this guy is doing!
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


//I think this is what this function's final version should look like
int main(void) {
   int string_length = 1024;
   char * string;
   struct timeval start, end;
   string_length = get_string_from_file("parsedComments.txt", &string); 

   gettimeofday(&start, 0); 
   count_keys_in_file("key_list.txt", string, string_length);
   gettimeofday(&end, 0); 

   long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
   printf("GPU Time: %lld \n", elapsed);


   cudaFreeHost(string);
   return 0;
}
