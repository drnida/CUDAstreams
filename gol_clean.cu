#include <stdlib.h> // for rand
#include <string.h> // for memcpy
#include <stdio.h> // for printf
#include <time.h> // for nanosleep

#define WIDTH 140 
#define HEIGHT 40 
#define TILE_WIDTH 32 
#define ITERATIONS 50000

// Create a constant array for efficient use on the device
__constant__ int offsets_dev[8][2];

const int offsets[8][2] = {{-1, 1},{0, 1},{1, 1},
                           {-1, 0},       {1, 0},
                           {-1,-1},{0,-1},{1,-1}};

// Error checking code from:
// http://stackoverflow.com/questions/14038589/what-is-the-canonical-way-to-check-for-errors-using-the-cuda-runtime-api
 
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true) {
   if(code != cudaSuccess) {
      fprintf(stderr,"GPUassert: %s %s %d\n", 
          cudaGetErrorString(code), file, line);
      if(abort) exit(code);
   }
}

/*
 * I considered moving this to the GPU, but since it only runs once, it
 * seemed a bit overkill. Added a seed for better random number generation
 */
void fill_board(int *board) {
    int i;
    time_t t;

    // seed the random number
    srand((unsigned) time(&t));

    for (i=0; i<WIDTH*HEIGHT; i++)
        board[i] = rand() % 2;
}

/*
 * Print the board. This code is unchanged
 */
void print_board(int *board) {
    int x, y;
    for (y=0; y<HEIGHT; y++) {
        for (x=0; x<WIDTH; x++) {
            char c = board[y * WIDTH + x] ? '#':' ';
            printf("%c", c);
        }
        printf("\n");
    }
    printf("-----\n");
}

/*
 * GPU code for determining the next step of the game
 * The current board is input with the current_dev array
 * The next board is output with the next_dev array
 */
__global__ void step(const int *current_dev, int *next_dev) {
    // Created shared memory 2D array
    __shared__ int curr_ds[TILE_WIDTH][TILE_WIDTH];

    int i, nx, ny, num_neighbors;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int idx = tx + blockDim.x*blockIdx.x;
    int idy = ty + blockDim.y*blockIdx.y;
 
    // Make sure the threads are within the bounds of our data
    if(idx >= WIDTH || idy >= HEIGHT) return;
 
    // Put data into shared array for access by other threads
    curr_ds[tx][ty] = current_dev[idx + WIDTH * idy]; 
    __syncthreads();

    // initialize the count
    num_neighbors = 0;

    // count the neighbors
    for (i=0; i<8; i++) {
        int temp;
        
        // Determine if the neighboring cells are accessible by shared memory
        nx = tx + offsets_dev[i][0];
        ny = ty + offsets_dev[i][1];
 
        if(nx >= 0 && nx < TILE_WIDTH && ny >= 0 && ny < TILE_WIDTH) {
            // grab the data from shared memory 
            temp = curr_ds[nx][ny];
        }
        else {
            // These neighboring cells are not in shared memory. Grab from
            // global memory
            nx = (idx + offsets_dev[i][0] + WIDTH) % WIDTH;
            ny = (idy + offsets_dev[i][1] + HEIGHT) % HEIGHT;
            temp = current_dev[ny * WIDTH + nx];
        } 
        if (temp) {
            num_neighbors++;
        }
    }

    // apply the Game of Life rules to this cell
    next_dev[idy * WIDTH + idx] = 0;

    // We know that curr_ds[tx][ty] is valid shared memory, so use it
    if ((curr_ds[tx][ty] && num_neighbors==2) ||
        num_neighbors==3) {
        next_dev[idy * WIDTH + idx] = 1;
    }
}

/* 
 * This function handles the animation engine and makes all the calls
 * to the GPU
 */
void gol_device(int *current) {
    int *current_dev, *next_dev;
    int n = WIDTH * HEIGHT;
    struct timespec delay = {0, 125000000}; // 0.125 seconds
    struct timespec remaining;
    int time = ITERATIONS;
    
    // Copy constant 2D array to the device
    cudaMemcpyToSymbol(offsets_dev, offsets, sizeof(int)*8*2);

    // Allocate memory on the device
    gpuErrchk(cudaMalloc((void **) &current_dev, sizeof(int)*n));
    gpuErrchk(cudaMalloc((void **) &next_dev, sizeof(int)*n));

    // Set up the device blocks and threads. This code allows for
    // arbitrary WIDTH and HEIGHT
    int numWidth = WIDTH/TILE_WIDTH;
    if(WIDTH % TILE_WIDTH) numWidth++;
    int numHeight = HEIGHT/TILE_WIDTH;
    if(HEIGHT % TILE_WIDTH) numHeight++;
    dim3 dimThreads(TILE_WIDTH, TILE_WIDTH);
    dim3 dimBlocks(numWidth, numHeight);

    // Start the animation 
    while(time) {
        print_board(current);  
        // Copy the data to the device
        gpuErrchk(cudaMemcpy(current_dev, current, sizeof(int)*n,
            cudaMemcpyHostToDevice)); 
    
        // Call the kernal function
        step<<<dimBlocks, dimThreads>>>(current_dev, next_dev);

        // Copy the data back to the host
        gpuErrchk(cudaMemcpy(current, next_dev, sizeof(int)*n, 
            cudaMemcpyDeviceToHost));
      
        time--;
        nanosleep(&delay, &remaining);
    }
    gpuErrchk(cudaFree(current_dev));
    gpuErrchk(cudaFree(next_dev));
}

int main(void) {
    // Allocate the current array dynamically instead of using a 
    // global array
    int *current;
    current = (int*)malloc(WIDTH * HEIGHT * sizeof(int));
    fill_board(current);
    gol_device(current);
    return 0;
}
