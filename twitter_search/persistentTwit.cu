/* SCRATCH CODE
   Compile with nvcc -gencode arch=compute_30,code=sm_30 substringSearch.cu
 */


#include <stdio.h>
#include <sys/time.h>
#include <limits.h>

#define numStreams 3
#define BLOCK 256
#define BUFF_SIZE 4096

__device__ int count_dev[16];

/*error checking from D&W*/
void errorChecking(cudaError_t err, int line) {
   if (err != cudaSuccess){
      printf(" %s in %s at line %d\n", cudaGetErrorString(err), 
            __FILE__, line);
      exit(EXIT_FAILURE);
   }
}


//THE KERNEL: This, I think, is all the functionality we want out of the kernel.

__global__ void search_kernel(char * pattern, int pattern_length, char * string, int string_length, int streamNum) {
   int tx = threadIdx.x;
   int idx = blockDim.x * blockIdx.x + tx; 
   //extern __shared__ char shared[];
   //char *string_sh = &shared[0];
   //char *pattern_sh = &shared[BLOCK + pattern_length];

   //if(tx < pattern_length) {
      //pattern_sh[tx] = pattern[tx];
      //shared[BLOCK + tx] = string[idx + BLOCK];
   //}
   //string_sh[tx] = string[idx];
   //__syncthreads();

   for(int j = 0; j < pattern_length; ++j)
      if(pattern[j] ^ string[idx + j])
         return;

   atomicAdd(&count_dev[streamNum],1);
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
int count_keys_in_file(char *filename, char * string_dev, int string_length, cudaStream_t * stream) {
   FILE *infile;
   FILE *outfile;
   int length;
   //size_t result;
   //int pattern_length = 0;
   //char * pattern;
   char BUFFER[16][BUFF_SIZE];
   char TEMP[512];
   int word_count[16];
   int current_string_index = 0;
   char * pattern_dev;

   for(int i = 0; i < 16; ++i){
       word_count[i] = 0;
   }

   //string_length = 1024;
   for(int j = 0 ; j < 16; ++j){
       for (int i = 0; i < BUFF_SIZE; ++i){
          BUFFER[j][i] = '\0';
       }
   }
   infile = fopen(filename, "r");
   outfile = fopen("key_value_pairs.txt", "a");

   int stopme = 0;
   int j;
   printf("Starting pos: %ld\n", ftell(infile));
   while( ! feof(infile) && stopme < 150){
   for( int i = 0; i < 16 && fgets(BUFFER[i], BUFF_SIZE, infile) != NULL && stopme < 150; ++i) {
      //++stopme;
      j = i + 1;
         


          length = strlen(BUFFER[i]) - 1;
          if(BUFFER[i][length] == '\n')
             BUFFER[i][length] = '\0';
          else
             ++length;
      //--------------------------------------------------------------------------       

       int count = 0;



       dim3 dimGrid( ceil(string_length/1024.0), 1, 1);
       dim3 dimBlock(1024, 1, 1);

       //printf("dimGrid.x: %d Threads: %d \n" , dimGrid.x, dimBlock.x);

    //------------------------

       errorChecking( cudaMalloc((void **) &pattern_dev, sizeof(char) * length* 1024), 
             __LINE__);

       errorChecking( cudaMemcpyToSymbolAsync(count_dev, &count, 
                sizeof(int), sizeof(int) * i, cudaMemcpyHostToDevice, stream[i]), __LINE__);



          errorChecking( cudaMemcpyAsync(&pattern_dev[ i], 
                   BUFFER[i],  length * sizeof(char), 
                   cudaMemcpyHostToDevice, stream[i] ), __LINE__);



          
          search_kernel<<<dimGrid.x, dimBlock.x, 0, stream[i]>>>
             (&pattern_dev[i], length, string_dev, string_length, i);

          errorChecking(cudaGetLastError(), __LINE__);


        }




       for(int i = 0; i < j; ++i){
        errorChecking(cudaMemcpyFromSymbolAsync(&word_count[i], count_dev, 
                    sizeof(int), sizeof(int) * i, cudaMemcpyDeviceToHost, stream[i]), __LINE__);
       //printf("Pattern is: %s Pos is: %ld Count is: %d\n", BUFFER[i], ftell(infile), word_count[i]);
       }
       for(int i = 0; i < j; ++i){
        cudaStreamSynchronize(stream[i]); 
       }
      //--------------------------------------------------------------------------
          //current_string_index += string_length;
       for(int i = 0; i < j  ; ++i){
          length = strlen(BUFFER[i]);
          ++length;
          BUFFER[i][length - 1] = ' ';
          BUFFER[i][length + 0] = '|';
          BUFFER[i][length + 1] = ' ';
          itoa(word_count[i], TEMP);
          strcpy(&BUFFER[i][length + 2], TEMP);
          BUFFER[i][strlen(BUFFER[i]) + 1] = '\0';
          BUFFER[i][strlen(BUFFER[i])] = '\n';

          int blah = fputs(BUFFER[i], outfile);
      } 
      
   }

   fclose(outfile);
   fclose(infile);

       cudaFree(pattern_dev);
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
   char * string_dev;
   struct timeval start, end;
   string_length = get_string_from_file("parsedComments.txt", &string); 

   cudaStream_t stream[16];

   for( int i = 0; i < 16; ++ i){
      errorChecking( cudaStreamCreate(&stream[i] ), __LINE__);
   }


   errorChecking( cudaMalloc((void **) &string_dev, sizeof(char) * string_length), 
         __LINE__);

   errorChecking( cudaMemcpyAsync(&string_dev[0], 
           &string[0],  string_length * sizeof(char), 
           cudaMemcpyHostToDevice, stream[0] ), __LINE__);

   cudaStreamSynchronize(stream[0]); 


   gettimeofday(&start, 0); 
   count_keys_in_file("key_list.txt", string, string_length, stream);
   gettimeofday(&end, 0); 

   long long elapsed = (end.tv_sec-start.tv_sec)*1000000ll + end.tv_usec-start.tv_usec;
   printf("GPU Time: %lld \n", elapsed);




   for(int i = 0; i < 16; ++i){ 
      errorChecking(cudaStreamDestroy(stream[i]), __LINE__); 
   }
   
   cudaFree(string_dev);
   cudaFreeHost(string);
   return 0;
}
