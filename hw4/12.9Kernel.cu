/* 
 * Kristina Frye
 * Dean Nida
 * August 7, 2014
 * HW 4, #3
 */

#include <stdio.h>

#define numAtoms 100
#define BLOCKSIZEX 16
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
/* A function for printing out our data */
void print_vector_slice(float *array, int n) {
    for (int y=0; y<n; y++){
	for (int x=0; x<n; x++)
	   printf("%0.0f \n", array[y * n + x]);
	//printf("\n");
    }
}

__global__ void DCSv2(float *energygrid, float *gridspacing, int *numatoms){

   int xindex = blockDim.x * blockIdx.x + threadIdx.x;
   int yindex = blockDim.y * blockIdx.y + threadIdx.y;
   int outaddr = yindex * numAtoms + xindex;

   int num = *numatoms;   
   if(xindex > num || yindex > num) return;

   float coorx = (*gridspacing) * xindex;
   float coory = (*gridspacing) * yindex;
   int block = 16;//BLOCKSIZEX;
   float gridspacing_coalesce = (*gridspacing)*block;

   int atomid;

   float energyvalx1=0.0f;
   float energyvalx2=0.0f;
   float energyvalx3=0.0f;
   float energyvalx4=0.0f;
   float energyvalx5=0.0f;
   float energyvalx6=0.0f;
   float energyvalx7=0.0f;
   float energyvalx8=0.0f;

   for (atomid=0; atomid<num; atomid++) {
      float dy = coory - atominfo[atomid].y;
      float dyz2 = dy*dy + atominfo[atomid].z;
      float dx1 = coorx - atominfo[atomid].x;
      float dx2 = dx1 + gridspacing_coalesce;
      float dx3 = dx2 + gridspacing_coalesce;
      float dx4 = dx3 + gridspacing_coalesce;
      float dx5 = dx4 + gridspacing_coalesce;
      float dx6 = dx5 + gridspacing_coalesce;
      float dx7 = dx6 + gridspacing_coalesce;
      float dx8 = dx7 + gridspacing_coalesce;
      energyvalx1 += atominfo[atomid].w*rsqrtf(dx1*dx1 + dyz2); 
      energyvalx2 += atominfo[atomid].w*rsqrtf(dx2*dx2 + dyz2); 
      energyvalx3 += atominfo[atomid].w*rsqrtf(dx3*dx3 + dyz2); 
      energyvalx4 += atominfo[atomid].w*rsqrtf(dx4*dx4 + dyz2); 
      energyvalx5 += atominfo[atomid].w*rsqrtf(dx5*dx5 + dyz2); 
      energyvalx6 += atominfo[atomid].w*rsqrtf(dx6*dx6 + dyz2); 
      energyvalx7 += atominfo[atomid].w*rsqrtf(dx7*dx7 + dyz2); 
      energyvalx7 += atominfo[atomid].w*rsqrtf(dx8*dx8 + dyz2); 
   }
  
   // using an atomic add because multiple threads are writing
   // to the same array index
   atomicAdd(&energygrid[outaddr + 0 * block ] ,  energyvalx1 );
   atomicAdd(&energygrid[outaddr + 1 * block ] ,  energyvalx2 );
   atomicAdd(&energygrid[outaddr + 2 * block ] ,  energyvalx3 );
   atomicAdd(&energygrid[outaddr + 3 * block ] ,  energyvalx4 );
   atomicAdd(&energygrid[outaddr + 4 * block ] ,  energyvalx5 );
   atomicAdd(&energygrid[outaddr + 5 * block ] ,  energyvalx6 );
   atomicAdd(&energygrid[outaddr + 6 * block ] ,  energyvalx7 );
   atomicAdd(&energygrid[outaddr + 7 * block ] ,  energyvalx8 );
}

/* Launches cuda kernel */
void launch_DSCv1(float * energyGrid, int boxDim, atom * molecule,
       float gridDist) {
    float *grid_dev, *spacing_dev, *spacing, *grid_dev_slice, *grid_slice;
    int *numatoms_dev;
    int *num;
    int allocateSize = sizeof(float) * (boxDim+28)*(boxDim+12);
 
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
    //errorChecking( cudaMalloc((void **) &grid_dev_slice, 
       // sizeof(float)*boxDim*boxDim));

    // Copy data to the device 
    errorChecking(cudaMemcpy(spacing_dev, spacing, sizeof(float),
       cudaMemcpyHostToDevice), __LINE__);
    errorChecking(cudaMemcpy(numatoms_dev, num, sizeof(int),
       cudaMemcpyHostToDevice), __LINE__);
    errorChecking(cudaMemcpyToSymbol(atominfo, molecule,
      sizeof(atom)*numAtoms, 0, cudaMemcpyHostToDevice), __LINE__);


    // Determine size of blocks and grid on device
    int numWidth = ceil((float)boxDim+28/BLOCK);
    int numHeight = ceil((float)boxDim+12/BLOCK);
    dim3 dimGrid(numWidth, numHeight);
    dim3 dimBlock(BLOCK, BLOCK);

    errorChecking(cudaMemcpy(grid_dev, energyGrid, allocateSize,
       cudaMemcpyHostToDevice), __LINE__);

    // Call the kernel
    DCSv2<<<dimGrid, dimBlock, 0>>>(grid_dev, spacing_dev, numatoms_dev);
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

    int boxSize = (boxDim+28)*(boxDim+12);
    energyGrid = (float *) malloc(boxSize*sizeof(float));

    for (int i = 0; i <  boxSize ; ++i){
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
   // print_vector(energyGrid, boxDim);
    print_vector_slice(energyGrid, boxDim);

    return 0;
}
