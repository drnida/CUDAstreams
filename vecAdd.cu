#include <stdio.h>

#define numAtoms 100
typedef struct {
  	float x;
  	float y;
  	float z;
  	float w;
 }atom;

__constant__ atom atominfo[numAtoms];

/*error checking from D&W*/
void errorChecking(cudaError_t err, int line) {
    if (err != cudaSuccess){
        printf(" %s in %s at line %d\n", cudaGetErrorString(err),
               __FILE__, line);
        exit(EXIT_FAILURE);
    }
}

void print_vector(float *array, int n) {
    int i;
    for (i=0; i<n; i++)
        printf("%0.0f ", array[i]);
    printf("\n");
}

__global__ void DCSv1(float *energygrid, float *gridspacing, int *numatoms){

   int xindex = blockDim.x * blockIdx.x + threadIdx.x;
   int yindex = blockDim.y * blockIdx.y + threadIdx.y;
   int outaddr = yindex * numAtoms + xindex;
/*
   float curenergy = energygrid[outaddr];
   float coorx = (*gridspacing) * xindex;
   float coory = (*gridspacing) * yindex;
   int atomid;
   float energyval=0.0f;
   for (atomid=0; atomid<(*numatoms); atomid++) {
      float dx = coorx - atominfo[atomid].x;
      float dy = coory - atominfo[atomid].y;
      energyval += atominfo[atomid].w*sqrtf(dx*dx + dy*dy + atominfo[atomid].z);
   }*/
   energygrid[outaddr] = 2;
//   energygrid[outaddr] = curenergy + energyval;
}


void launch_DSCv1(float * energyGrid, int boxDim, atom * molecule, float gridDist){
    float *grid_dev, *spacing_dev, *grid; 
    int *numatoms_dev;
    int num = numAtoms;
    int *num_host;
    int allocateSize = sizeof(float) * boxDim*boxDim*boxDim;
    printf("\nInside the launch function.\n");
    print_vector(energyGrid, boxDim);
    
    //float * fltarray;
    //fltarray = (float * ) malloc( boxDim*boxDim*boxDim  * sizeof(float));
    errorChecking( cudaMalloc((void **) &grid_dev, allocateSize), __LINE__);
    errorChecking( cudaMalloc((void **) &spacing_dev, sizeof(float)), __LINE__);
    errorChecking( cudaMalloc((void **) &numatoms_dev, sizeof(int)), __LINE__);
 
/* 
    char *string_dev;
    char string [5] = {"abcd"};
    int length = 5;
    
    int size = allocateSize;
    
    float *h_A = (float *)malloc(size);
    float *d_A = NULL;
    err = cudaMalloc((void **)&d_A, size);
    err = cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    
    errorChecking(err, __LINE__);
    errorChecking(err, __LINE__);
    
    errorChecking(cudaMalloc((void **) &string_dev, sizeof(char) * length), __LINE__);
    errorChecking(cudaMemcpy(string_dev, string, sizeof(char) * length, cudaMemcpyHostToDevice), __LINE__);
*/
    errorChecking(cudaMemcpy(grid_dev, energyGrid, allocateSize, cudaMemcpyHostToDevice), __LINE__);
    errorChecking(cudaMemcpy(spacing_dev, &gridDist, sizeof(float), cudaMemcpyHostToDevice), __LINE__);
    errorChecking(cudaMemcpy(numatoms_dev, &num, sizeof(int), cudaMemcpyHostToDevice), __LINE__); 
 //   errorChecking(cudaMemcpyToSymbol(atominfo, &molecule, sizeof(atom)*numAtoms, 0, cudaMemcpyHostToDevice), __LINE__);

    dim3 dimGrid(1, 1, 1);
    dim3 dimBlock(100, 100, 1);
    
    DCSv1<<<dimGrid, dimBlock>>>(grid_dev, spacing_dev, numatoms_dev);
    errorChecking(cudaGetLastError(), __LINE__);
   
    // Step 4: Retrieve the results
    errorChecking( cudaMemcpy(energyGrid, grid_dev, allocateSize, cudaMemcpyDeviceToHost), __LINE__);

    // Step 5: Free device memory
    cudaFree(&grid_dev);
    cudaFree(&spacing_dev);
    cudaFree(&numatoms_dev);
}

int main(void) {
    atom molecule[numAtoms];   

    int boxDim = numAtoms;
    float * energyGrid;
    float gridDist = 1; 
    energyGrid = (float * ) malloc( boxDim*boxDim*boxDim  * sizeof(float)); 

    for (int i = 0; i <  boxDim*boxDim*boxDim ; ++i){
       energyGrid[i] = 1;
    }
   
    printf("Energy grid before kernel:\n");
    print_vector(energyGrid, boxDim); 

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
