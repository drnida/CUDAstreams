/* 
 * Kristina Frye
 * Dean Nida
 * August 7, 2014
 * HW 4, #1
 */

#include <stdio.h>

#define numAtoms 100
#define BLOCK 16 

struct atom {
   float x; float y; float z; float w;
 } a;

__constant__ struct atom atominfo[numAtoms];

void errorChecking(cudaError_t err, int line) {
    if (err != cudaSuccess){
        printf(" %s in %s at line %d\n", cudaGetErrorString(err),
               __FILE__, line);
        exit(EXIT_FAILURE);
    }
}

/* A function for printing out our data */
void print_vector(float *array, int n) {
    int i;
    for (i=0; i<n*n; i++)
        printf("%0.0f \n", array[i]);
    printf("\n");
}

void print_vector_slice(float *array, int n) {
   for (int y=0; y<n; y++) {
      for (int x=0; x<n; x++)
 	 printf("%0.0f ", array[y * n + x]);
      printf("\n");
   }
}

/* Cuda Kernel function */
__global__ void DCSv1(float *energygrid, float *gridspacing, int *numatoms){

   // Get thread addresses
   int xindex = blockDim.x * blockIdx.x + threadIdx.x;
   int yindex = blockDim.y * blockIdx.y + threadIdx.y;
   int outaddr = yindex * numAtoms + xindex;
   int num = *numatoms;  
   int gridsp = *gridspacing; 

   // Check for thread boundries
   if(xindex > num - 1 || yindex > num - 1) return;

   // Start algorithm from 12.5 in book
   float curenergy = energygrid[outaddr];
   float coorx = gridsp * xindex;
   float coory = gridsp * yindex;
   int atomid;
   float energyval=0.0f;

   for (atomid=0; atomid<num; atomid++) {
      float dx = coorx - atominfo[atomid].x;
      float dy = coory - atominfo[atomid].y;
      energyval += atominfo[atomid].w*rsqrtf(dx*dx + dy*dy + atominfo[atomid].z);
   }
   energygrid[outaddr] = curenergy + energyval;
}

/* Launches cuda kernel */
void launch_DSCv1(float * energyGrid, int boxDim, atom * molecule, 
       float gridDist){
    float *grid_dev, *spacing_dev, *spacing; 
    int *numatoms_dev;
    int *num;
    int allocateSize = sizeof(float) * boxDim*boxDim;

    // Malloc memory for local host pointer variables used for copying data
    // to the device
    spacing = (float*)malloc(sizeof(float));
    *spacing = gridDist;
    num = (int*)malloc(sizeof(int));
    *num = numAtoms;

    // Malloc memory on the device
    errorChecking( cudaMalloc((void **) &grid_dev, allocateSize), __LINE__);
    errorChecking( cudaMalloc((void **) &spacing_dev, sizeof(float)), __LINE__);
    errorChecking( cudaMalloc((void **) &numatoms_dev, sizeof(int)), __LINE__);

    // Copy data to the device 
    errorChecking(cudaMemcpy(grid_dev, energyGrid, allocateSize, 
       cudaMemcpyHostToDevice), __LINE__);
    errorChecking(cudaMemcpy(spacing_dev, spacing, sizeof(float), 
       cudaMemcpyHostToDevice), __LINE__);
    errorChecking(cudaMemcpy(numatoms_dev, num, sizeof(int), 
       cudaMemcpyHostToDevice), __LINE__); 
    errorChecking(cudaMemcpyToSymbol(atominfo, molecule, 
      sizeof(atom)*numAtoms, 0, cudaMemcpyHostToDevice), __LINE__);

    // Determine size of blocks and grid on device
    int numWidth = ceil((float)boxDim/BLOCK);
    int numHeight = ceil((float)boxDim/BLOCK);
    dim3 dimGrid(numWidth, numHeight); 
    dim3 dimBlock(BLOCK, BLOCK);

    // Call the kernel
    DCSv1<<<dimGrid, dimBlock, 0>>>(grid_dev, spacing_dev, numatoms_dev);
    errorChecking(cudaGetLastError(), __LINE__);
   
    // Step 4: Retrieve the results
    errorChecking( cudaMemcpy(energyGrid, grid_dev, allocateSize, 
       cudaMemcpyDeviceToHost), __LINE__);

    // Step 5: Free device memory
    cudaFree(grid_dev);
    cudaFree(spacing_dev);
    cudaFree(numatoms_dev);
}

int main(void) {
    atom molecule[numAtoms];   

    int boxDim = numAtoms;
    float *energyGrid;
    float gridDist = 1; 
    energyGrid = (float *) malloc(boxDim*boxDim*sizeof(float)); 

    for (int i = 0; i <  boxDim*boxDim; ++i){
       energyGrid[i] = 1;
    }
   
    for( int i = 0; i < numAtoms; ++i){
      molecule[i].x = i; 
      molecule[i].y = i; 
      molecule[i].z = i; 
      molecule[i].w = i;
    }

    launch_DSCv1(energyGrid, boxDim, molecule, gridDist);

    printf("\nEnergy grid after kernel:\n");
    print_vector(energyGrid, boxDim);

    return 0;
}
