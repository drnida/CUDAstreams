/* SCRATCH CODE
   Compile with nvcc -gencode arch=compute_30,code=sm_30 substringSearch.cu
 */


#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

#define numStreams 3
#define BLOCK 256
#define BUFF_SIZE 4096

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
   //int * result = (int*)&patterns_and_output[idx * sizeof(char) * pattern_section];
   //char * pattern = (char*)(result + sizeof(int));
   char * pattern = (char*)(result);
   //*result = 0;

   //char * pattern = patterns_and_output + idx*128;

   const char *s;
   //for(s = pattern; *s || *(s + 1) || *(s+2) || *(s+3); ++s) {}

   for(s = pattern + pattern_section - sizeof(int); !(*s); --s) {}
   pattern_length = (s - pattern) - 4;

   pattern = pattern + 4;

  // for(int i = 0; i <= pattern_length; ++i)
  // {
  //    pattern[i] = 'X';
  // }

   for(int i = 0; i < search_length - pattern_section; ++i)
   {
      if(pattern[0] == search_segment[i])
      {
         for(j = 0; pattern[j] == search_segment[i + j] && j <= pattern_length && pattern[j]; ++j) {}
         if(j - 1 == pattern_length)
            ++count;
      }
   }

   *result = count;
 //  *result = pattern_length;
   //for(int i = 0; i < pattern_section - sizeof(int); ++i)
//   {
//         *(result + sizeof(int)) = 'X';
//         return;
//   }
   //for (int i = 0; i < search_length - pattern_section; ++i)
  // {
    //  if(!(*(search_segment + i) ^ *pattern)) {
//      temp_bool = 0;
//      if( search_segment[i] == pattern[0])
  //    {
   //
     //    j = 0;
       //  while(pattern[j] != 'd' && search_segment[j + i] == pattern[j])
         //{
//            if(pattern[j] == 'd')
  ///          {
     //          ++count;
       ///     }
///            ++j;
   ///      }
         //if(!pattern[j])
         //   ++count;
   //      for(j = 0; !(*(pattern + j) ^ *(search_segment + i + j)) && 
     //          j < pattern_section - sizeof(int) && 
       //        *(pattern + j); ++j){} //NOTE SEMICOLON AT THE END OF THIS LOOP, it doesn't need to do anything
      //   for(j = 0; pattern[j] == search_segment[i + j] && j < pattern_section - sizeof(int); ++j) {}
        // if(!pattern[j - 1])
     //       *result = 5;
         //if(!*(pattern + j) ^ *(search_segment + i + j))
         //if(pattern[j] == '\0')
       //     ++count;

     // }
   //for(int j = 0; j < pattern_section - sizeof(int); ++j)
     // if(pattern[j] ^ search_segment[i + j])
       //  return;

   //}
  // *result = pattern_length;
   //*result = sizeof(int);
   //extern __shared__ char shared[];
   //char *string_sh = &shared[0];
   //char *pattern_sh = &shared[BLOCK + pattern_length];

   //if(tx < pattern_length) {
      //pattern_sh[tx] = pattern[tx];
      //shared[BLOCK + tx] = string[idx + BLOCK];
   //}
   //string_sh[tx] = string[idx];
   //__syncthreads();
/*
   for(int j = 0; j < pattern_length; ++j)
      if(pattern[j] ^ string[idx + j])
         return;

   atomicAdd(&count_dev,1);
*/
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
int search(char * keys, int keys_length, int key_count, char * data, int data_length) {
   //These two lines just debug to make sure my formatting is correct.

   //Couldn't get this stuff to work
   char * keys_dev;
   char * data_dev;
   //int streamOffset;
   //cudaStream_t stream[numStreams];
   int numThreads = BLOCK;

//   for( int i = 0; i < numStreams; ++ i){
//      errorChecking( cudaStreamCreate(&stream[i] ), __LINE__);
//   }


//   int streamLength = ceil((float)string_length/numStreams);
//   int streamBytes = streamLength * sizeof(char);

//   printf("streamLength: %d, streamBytes %d\n", streamLength, streamBytes);    

   dim3 dimGrid( ceil(key_count/(float)numThreads), 1, 1);
   dim3 dimBlock(numThreads, 1, 1);

   printf("dimGrid.x: %d Threads: %d \n" , dimGrid.x, dimBlock.x);

   errorChecking( cudaMalloc((void **) &keys_dev, sizeof(char) * keys_length), 
         __LINE__);

   errorChecking( cudaMalloc((void **) &data_dev, sizeof(char) * data_length), 
         __LINE__);

//   for(int i = 0; i < numStreams; ++i){
//      streamOffset = i * streamLength;
//      printf("streamOffset is: %d\n", streamOffset);

      errorChecking( cudaMemcpy(keys_dev, 
               keys,  keys_length * sizeof(char), 
               cudaMemcpyHostToDevice), __LINE__);

      errorChecking( cudaMemcpy(data_dev, 
               data, data_length * sizeof(char), 
               cudaMemcpyHostToDevice), __LINE__);

//   }    
   //int sharedMem = (BLOCK+pattern_length) * sizeof(char)+pattern_length * sizeof(char);
//   for(int i = 0; i < numStreams; ++i){
//      streamOffset = i * streamLength; 
      
      search_kernel<<<dimGrid.x, dimBlock.x>>>
         (keys_dev, keys_length, data_dev, data_length);

      errorChecking(cudaGetLastError(), __LINE__);

      errorChecking(cudaMemcpy(keys, keys_dev, keys_length * sizeof(char),
                     cudaMemcpyDeviceToHost), __LINE__);

      for(int i = 0; i < key_count; ++i) {
         printf("KEY: %s", (keys + 128 * i + sizeof(int)));
         int * num;
         num = (int*)(keys+128*i);
         printf(" :: VALUE: %d\n", *num);
      }    

//   cudaStreamSynchronize(stream[2]); 

//   errorChecking(cudaMemcpyFromSymbol(&count, count_dev, 
//            sizeof(int), 0, cudaMemcpyDeviceToHost), __LINE__);

//   printf("Count is: %d\n", count);

//   for(int i = 0; i < numStreams; ++i){ 
//      errorChecking(cudaStreamDestroy(stream[i]), __LINE__); 
//   }
   cudaFree(data_dev);
   cudaFree(keys_dev);

   return 1;
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
   //int paddedLength = numStreams * ceil((float)length/numStreams);
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
/*   if(paddedLength > length) {
      for(int i = length - 1; i < paddedLength - 1; i++) {
         (*input)[i] = 0;
      }
      (*input)[paddedLength - 1] = '\0';
   }

   printf("\n\nDEBUG: strlen(string) == %lu\n\n", strlen(*input));
   return paddedLength;
   */
   return length;
}

//This (rather messy) function grabs keys from the file specified in main and writes
//the key and the int returned by search() to the key_value_pairs file
int create_key_results_string(char *filename, char ** string, int * key_count) 
{
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
      }

      if(j == 128)
      {
         --i;
         continue;
         //*string[127*i + j] = '\0';
      }


      for(; j < 128; ++j)
         *(*string + 128*i + j) = '\0';
   }

   return (128 * *key_count);
}

/*      word_count = search(BUFFER, length, string, string_length);
      //current_string_index += string_length;
      ++length;
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

   return 0; */
//}

/*      word_count = search(BUFFER, length, string, string_length);
      //current_string_index += string_length;
      ++length;
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

   return 0; */
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
   int data_length = 0;
   int keys_length = 0;
   int key_count = 0;
   char * data;
   char * keys;
   struct timeval start, end;
   data_length = get_string_from_file("parsedComments.txt", &data); 
   keys_length = create_key_results_string("key_list.txt", &keys, &key_count);

   printf("%s", keys);

   gettimeofday(&start, 0); 
   search(keys, keys_length, key_count, data, data_length);
   gettimeofday(&end, 0); 

   long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
   printf("GPU Time: %lld \n", elapsed);

   cudaFreeHost(keys);
   cudaFreeHost(data);
   return 0;
}
