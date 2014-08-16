/* 
 * Compile with nvcc -gencode arch=compute_30,code=sm_30 gatherSearchStreams.cu 
 */


#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

#define numStreams 3
#define BLOCK 256
#define BUFF_SIZE 4096
#define BYTES_TO_SEARCH 1048576

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
   for(int i = 0; i < search_length - pattern_section; ++i) {
      if(pattern[0] == search_segment[i]) {
         for(j = 0; pattern[j] == search_segment[i + j] && 
            j <= pattern_length && pattern[j]; ++j) {}
         if(j - 1 == pattern_length)
            ++count;
      }
   }

   atomicAdd(result, count);
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

// Utility function from K&R
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

// This function calls the kernel
int search(char * keys, int keys_length, int key_count, int data_length) {
   char * keys_dev;
   char * data_dev0;
   char * data_dev1;
   char * data_dev2;
   char * data = NULL;
   int numThreads = BLOCK;
   int data_bookmark = 0;
   int segment_length = 0;

   cudaStream_t stream[3];

   // Create the streams
   for( int i = 0; i < 3; ++ i) {
      errorChecking( cudaStreamCreate(&stream[i] ), __LINE__);
   }
   
   // Get the keys from file
   keys_length = create_key_results_string("key_list.txt", &keys, &key_count);

   // Allocate memory on the device for the keys and copy to the device
   errorChecking(cudaMalloc((void **) &keys_dev, sizeof(char) * keys_length), 
      __LINE__);
   errorChecking(cudaMemcpy(keys_dev, keys, keys_length * sizeof(char), 
            cudaMemcpyHostToDevice), __LINE__);

   // Allocate memory on the device for the data set
   errorChecking(cudaMalloc((void **) &data_dev0, 
      sizeof(char) * BYTES_TO_SEARCH), __LINE__);
   errorChecking(cudaMalloc((void **) &data_dev1, 
      sizeof(char) * BYTES_TO_SEARCH), __LINE__);
   errorChecking(cudaMalloc((void **) &data_dev2, 
      sizeof(char) * BYTES_TO_SEARCH), __LINE__);

   // Get the data set from file
   data_length = get_string_from_file("parsedComments.txt", &data, 
      &data_bookmark); 

   for(int i = 0, j = 0; data_length - BYTES_TO_SEARCH * i > 0; ++i, ++j) {
      if(j==3) j = 0;

      segment_length = data_length - BYTES_TO_SEARCH * i < BYTES_TO_SEARCH ?
                        data_length - BYTES_TO_SEARCH * i : BYTES_TO_SEARCH;

      // Set up the threads
      dim3 dimGrid( ceil(key_count/(float)numThreads), 1, 1);
      dim3 dimBlock(numThreads, 1, 1);

      // Copy the data set to the device
      if(j == 0)
         errorChecking(cudaMemcpyAsync(data_dev0, (data+BYTES_TO_SEARCH*i), 
            segment_length, cudaMemcpyHostToDevice, stream[j]), __LINE__);
      if(j == 1)
         errorChecking(cudaMemcpyAsync(data_dev1, (data+BYTES_TO_SEARCH*i), 
            segment_length, cudaMemcpyHostToDevice, stream[j]), __LINE__);
      if(j == 2)
         errorChecking(cudaMemcpyAsync(data_dev2, (data+BYTES_TO_SEARCH*i), 
            segment_length, cudaMemcpyHostToDevice, stream[j]), __LINE__);

      // Call the kernel
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

   // Get the results from the device
   errorChecking(cudaMemcpy(keys, keys_dev, keys_length * sizeof(char),
      cudaMemcpyDeviceToHost), __LINE__);

   // Write the results to file
   FILE *outfile = fopen("output.txt", "a");
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

   // Free resources
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

   file = fopen(filename, "r");
   fseek(file, 0, SEEK_END);
   length = ftell(file);
   printf("File Length is: %d bytes.\n", length);
   rewind(file);
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

// This function grabs keys from the file specified 
int create_key_results_string(char *filename, char ** string, int * key_count) {
   FILE *keyfile;

   char BUFFER[128];
   int length = 0;
   int j = 0;
   int i = 0;

   for (int i = 0; i < 128; ++i)
      BUFFER[i] = '\0';

   keyfile = fopen(filename, "r");
   if(!keyfile)
      printf("keyfile didn't open");

   fseek(keyfile, 0, SEEK_END);
   rewind(keyfile);

   while(fgets(BUFFER, 128, keyfile)) {  
      length = strlen(BUFFER) - 1;
      if(BUFFER[length] == '\n')
         ++i;
   }

   *key_count = i;

   errorChecking(cudaMallocHost((void**) string, 128 * i * sizeof(char)), 
         __LINE__ );
   if(*string == NULL) {
      fputs("Memory error", stderr);
      exit(2);
   }
   rewind(keyfile);

   for(i = 0; fgets(BUFFER, 128, keyfile) != NULL; ++i) {
      length = strlen(BUFFER) - 1;
      if(BUFFER[length] == '\n')
         BUFFER[length] = '\0';
      else {
         --i;
         continue;
      }

      for(j = 0; *(BUFFER + j) != '\0' && j < 128; ++j) {
         *(*string + 128*i + j) = *(BUFFER + j);
         if(j < 4)
            *(*string + 128*i + j) = '\0';
      }

      if(j == 128) {
         --i;
         continue;
      }

      for(; j < 128; ++j)
         *(*string + 128*i + j) = '\0';
   }

   return (128 * *key_count);
}

int main(void) {
   int data_length = 0;
   int keys_length = 0;
   int key_count = 0;
   char * keys = NULL;
   struct timeval start, end;

   gettimeofday(&start, 0); 
   search(keys, keys_length, key_count, data_length);
   gettimeofday(&end, 0); 

   long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + 
      end.tv_usec-start.tv_usec;
   printf("GPU Time: %lld \n", elapsed);
   return 0;
}
