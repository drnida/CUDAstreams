/* SCRATCH CODE
   Compile with nvcc -gencode arch=compute_30,code=sm_30 substringSearch.cu
 */


#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

#define numStreams 3
#define BLOCK 256
#define BUFF_SIZE 4096
//#define BYTES_TO_SEARCH 262144
#define BYTES_TO_SEARCH 1048576
//#define BYTES_TO_SEARCH 2097152
//#define BYTES_TO_SEARCH 4194304
//#define BYTES_TO_SEARCH 8388608

int create_key_results_string(char *filename, char ** string, int * key_count); 
int get_string_from_file(char *filename, char **input, int * current_file_position);

/*error checking from D&W*/
void errorChecking(cudaError_t err, int line) {
   if (err != cudaSuccess){
      printf(" %s in %s at line %d\n", cudaGetErrorString(err), 
            __FILE__, line);
      exit(EXIT_FAILURE);
   }
}


//THE KERNEL: This, I think, is all the functionality we want out of the kernel.

//Search length must be number of bytes
__global__ void search_kernel(char * patterns_and_output, int pattern_length, char * search_segment, int search_length) {
   int tx = threadIdx.x;
   int idx = blockDim.x * blockIdx.x + tx; 

   int pattern_section = 128;
   int j = 0;

   int count = 0;
   int * result = (int*)(patterns_and_output + idx*sizeof(char)*(pattern_section));
   char * pattern = (char*)(result);

   const char *s;

   //This gets the strlen of the pattern to be searched for
   for(s = pattern + pattern_section - sizeof(int); !(*s); --s) {}
   pattern_length = (s - pattern) - 4;

   //Advances the pattern pointer past the int field at the beginning
   pattern = pattern + 4;

   //Searches for the pattern in the search_segment string and increments count each time it's found
   for(int i = 0; i < search_length - pattern_section; ++i)
   {
      if(pattern[0] == search_segment[i])
      {
         for(j = 0; pattern[j] == search_segment[i + j] && j <= pattern_length && pattern[j]; ++j) {}
         if(j - 1 == pattern_length)
            ++count;
      }
   }

   atomicAdd(result, count);
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
//int search(char * pattern_, int pattern_length, char * string, int string_length) {
int search(char * keys, int keys_length, int key_count, int data_length) 
{
   char * keys_dev;
   char * data_dev0;
   char * data_dev1;
   char * data_dev2;
   char * data = NULL;
   int numThreads = BLOCK;
   int data_bookmark = 0;
   int keys_bookmark = 0;
   int kernel_count = 0;
   int segment_length = 0;

   cudaStream_t stream[3];

   for( int i = 0; i < 3; ++ i){
      errorChecking( cudaStreamCreate(&stream[i] ), __LINE__);
   }

   keys_length = create_key_results_string("key_list.txt", &keys, &key_count);

   errorChecking( cudaMalloc((void **) &keys_dev, sizeof(char) * keys_length), 
         __LINE__);
   errorChecking( cudaMemcpy(keys_dev, 
            keys,  keys_length * sizeof(char), 
            cudaMemcpyHostToDevice), __LINE__);

   errorChecking( cudaMalloc((void **) &data_dev0, sizeof(char) * BYTES_TO_SEARCH), 
         __LINE__);
   errorChecking( cudaMalloc((void **) &data_dev1, sizeof(char) * BYTES_TO_SEARCH), 
         __LINE__);
   errorChecking( cudaMalloc((void **) &data_dev2, sizeof(char) * BYTES_TO_SEARCH), 
         __LINE__);


   data_length = get_string_from_file("parsedComments.txt", &data, &data_bookmark); 

   for(int i = 0, j = 0; data_length - BYTES_TO_SEARCH * i > 0; ++i, ++j)
   {
      if(j==3)
         j = 0;

      segment_length = data_length - BYTES_TO_SEARCH * i < BYTES_TO_SEARCH ?
                        data_length - BYTES_TO_SEARCH * i : BYTES_TO_SEARCH;

      dim3 dimGrid( ceil(key_count/(float)numThreads), 1, 1);
      dim3 dimBlock(numThreads, 1, 1);

      printf("dimGrid.x: %d Threads: %d \n" , dimGrid.x, dimBlock.x);
      printf("\nsegment_length:%d \n" , segment_length);

      if(j == 0)
         errorChecking( cudaMemcpyAsync(data_dev0, 
                  (data+BYTES_TO_SEARCH*i), segment_length, 
                  cudaMemcpyHostToDevice, stream[j]), __LINE__);
      if(j == 1)
         errorChecking( cudaMemcpyAsync(data_dev1, 
                  (data+BYTES_TO_SEARCH*i), segment_length, 
                  cudaMemcpyHostToDevice, stream[j]), __LINE__);
      if(j == 2)
         errorChecking( cudaMemcpyAsync(data_dev2, 
                  (data+BYTES_TO_SEARCH*i), segment_length, 
                  cudaMemcpyHostToDevice, stream[j]), __LINE__);

      printf("\nKernel #%d\n", ++kernel_count);
      if(j == 0)
         search_kernel<<<dimGrid.x, dimBlock.x, 0, stream[j]>>>
            (keys_dev, keys_length, data_dev0, segment_length);
      if(j == 1)
         search_kernel<<<dimGrid.x, dimBlock.x, 0, stream[j]>>>
            (keys_dev, keys_length, data_dev1, segment_length);
      if(j == 2)
         search_kernel<<<dimGrid.x, dimBlock.x, 0, stream[j]>>>
            (keys_dev, keys_length, data_dev2, segment_length);
   }

   cudaDeviceSynchronize();

   errorChecking(cudaMemcpy(keys, keys_dev, keys_length * sizeof(char),
            cudaMemcpyDeviceToHost), __LINE__);

   FILE * outfile = fopen("output.txt", "a");

   char BUFFER[500];
   for(int i = 0; i < key_count; ++i) {
      int num = *(int*)(keys+128*i);

      itoa(num, BUFFER);

      fputs(BUFFER, outfile);
      fputs("  -  ", outfile);
      strcpy(BUFFER, (keys + 128*i + sizeof(int)));
      fputs(BUFFER, outfile);

      fputs("\n", outfile);
   }    

   fclose(outfile);
   for(int i = 0; i < numStreams; ++i){ 
      errorChecking(cudaStreamDestroy(stream[i]), __LINE__); 
   }

   cudaFree(data_dev0);
   cudaFree(data_dev1);
   cudaFree(data_dev2);
   cudaFree(keys_dev);
   cudaFreeHost(keys);
   cudaFreeHost(data);

   return 1;
}

// Grab data from external file
int get_string_from_file(char *filename, char **input, int * current_file_position) {
   FILE *file;
   int length;
   size_t result;
   int difference = 0;

   file = fopen(filename, "r");
   fseek(file, 0, SEEK_END);
   length = ftell(file);
   //difference = ftell(file) - *current_file_position;
   //length = difference < BYTES_TO_SEARCH ? difference : BYTES_TO_SEARCH;
   printf("File Length is: %d bytes.\n", length);
   //if(fseek(file, *current_file_position, SEEK_SET))
   //   printf("\n\nThere was an error grabbing a chunk of the search file, currfilepos: %d\n\n", *current_file_position);
   rewind(file);
   //*current_file_position += length;

   errorChecking( cudaMallocHost((void**) input, length * sizeof(char)), 
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

   return length;
}

//This (rather messy) function grabs keys from the file specified in main and writes
//the key and the int returned by search() to the key_value_pairs file
int create_key_results_string(char *filename, char ** string, int * key_count) 
{
   static int piece_of_file = 0;
   FILE *keyfile;

   char BUFFER[128];
   int keyfile_length = 0;
   int length = 0;
   int j = 0;
   int i = 0;

   for (int i = 0; i < 128; ++i)
      BUFFER[i] = '\0';

   keyfile = fopen(filename, "r");
   if(!keyfile)
      printf("keyfile didn't open");

   fseek(keyfile, 0, SEEK_END);
   keyfile_length = ftell(keyfile);
   rewind(keyfile);

   while(fgets(BUFFER, 128, keyfile))
   {  
      length = strlen(BUFFER) - 1;
      if(BUFFER[length] == '\n')
         ++i;
   }

   *key_count = i;

   errorChecking( cudaMallocHost((void**) string, 128 * i * sizeof(char)), 
         __LINE__ );
   if(*string == NULL) {
      fputs("Memory error", stderr);
      exit(2);
   }
   rewind(keyfile);

   for(i = 0; fgets(BUFFER, 128, keyfile) != NULL; ++i)
   {
      length = strlen(BUFFER) - 1;
      if(BUFFER[length] == '\n')
         BUFFER[length] = '\0';
      else
      {
         --i;
         continue;
      }

      for(j = 0; *(BUFFER + j) != '\0' && j < 128; ++j)
      {
         *(*string + 128*i + j) = *(BUFFER + j);
         if(j < 4)
            *(*string + 128*i + j) = '\0';
      }

      if(j == 128)
      {
         --i;
         continue;
      }


      for(; j < 128; ++j)
         *(*string + 128*i + j) = '\0';
   }

   return (128 * *key_count);
}

//I think this is what this function's final version should look like
int main(void) {
   int data_length = 0;
   int keys_length = 0;
   int key_count = 0;
   char * keys = NULL;
   struct timeval start, end;

   printf("%s", keys);

   gettimeofday(&start, 0); 
   search(keys, keys_length, key_count, data_length);
   gettimeofday(&end, 0); 

   long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
   printf("GPU Time: %lld \n", elapsed);

   return 0;
}