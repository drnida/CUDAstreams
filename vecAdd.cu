#include <stdio.h>

#define numAtoms 100
typedef struct {
  	float x;
  	float y;
  	float z;
  	float w;
 }atom;

__constant__ atom atominfo[numAtoms];
__global__ void DCSv1(float * energygrid, float gridspacing, int numatoms){

    	int xindex = blockDim.x * blockIdx.x + threadIdx.x;
    	int yindex = blockDim.y * blockIdx.y + threadIdx.y;
        int zindex = blockDim.z * blockIdx.z + threadIdx.z;

        int outaddr = zindex * numAtoms * numAtoms + yindex * numAtoms + xindex;

	float curenergy = energygrid[outaddr];
	float coorx = gridspacing * xindex;
	float coory = gridspacing * yindex;
	int atomid;
	float energyval=0.0f;
	for (atomid=0; atomid<numatoms; atomid++) {
		float dx = coorx - atominfo[atomid].x;
		float dy = coory - atominfo[atomid].y;
		energyval += atominfo[atomid].w*sqrtf(dx*dx + dy*dy + atominfo[atomid].z);

	}
	energygrid[outaddr] = curenergy + energyval;
}


void launch_DSCv1(float * energyGrid, int boxDim, atom * molecule , float gridDist){
    float * grid_dev;
    float spacing_dev; 
    int numatoms_dev;
    int num = numAtoms;

    cudaMalloc((void **) &grid_dev, sizeof(float) * boxDim*boxDim*boxDim);
    cudaMalloc((void **) &spacing_dev, sizeof(float) );
    cudaMalloc((void **) &numatoms_dev, sizeof(int) );

    // Step 2: Copy the input vectors to the device
    cudaMemcpy(grid_dev, energyGrid, sizeof(float) * boxDim*boxDim*boxDim, cudaMemcpyHostToDevice);
    cudaMemcpy(&spacing_dev, &gridDist, sizeof(float) * boxDim*boxDim*boxDim, cudaMemcpyHostToDevice);
    cudaMemcpy(&numatoms_dev, &num, sizeof(int) *1 , cudaMemcpyHostToDevice);


    cudaMemcpyToSymbol(atominfo, &molecule,
                sizeof(atom)*numAtoms, 0, cudaMemcpyHostToDevice);


    // Step 3: Invoke the kernel
    // We allocate enough blocks (each 512 threads long) in the grid to
    // accomodate all `n` elements in the vectors. The 512 long block size
    // is somewhat arbitrary, but with the constraint that we know the
    // hardware will support blocks of that size.
    dim3 dimGrid((boxDim + 1024 - 1) / 1024, (boxDim + 1024 - 1) / 1024, 1);
    dim3 dimBlock(1024, 1024, 1);
    DCSv1<<<dimGrid, dimBlock>>>(grid_dev, spacing_dev, numatoms_dev);

    // Step 4: Retrieve the results
    cudaMemcpy(energyGrid, grid_dev, sizeof(float) * boxDim*boxDim*boxDim , cudaMemcpyDeviceToHost);

    // Step 5: Free device memory
    cudaFree(grid_dev);
    cudaFree(&spacing_dev);
    cudaFree(&numatoms_dev);
}

void print_vector(int *array, int n) {
    int i;
    for (i=0; i<n; i++)
        printf("%d ", array[i]);
    printf("\n");
}

int main(void) {
    atom molecule[numAtoms];   


    int boxDim = numAtoms;
    float * energyGrid;
    float gridDist = 1; 
     energyGrid = (float * ) malloc( boxDim*boxDim*boxDim  * sizeof(float)); 


    for (int i = 0; i <  boxDim*boxDim*boxDim ; ++i){
       energyGrid[i] = 0;
    } 

    for( int i = 0; i < numAtoms; ++i){
      molecule[i].x = i; 
      molecule[i].y = i; 
      molecule[i].z = i; 
      molecule[i].w = i;
    } 



    
    launch_DSCv1(energyGrid, boxDim, molecule , gridDist);


     for(int i = 0; i < 100; ++i){

	printf("%f", energyGrid[i]);
	}


    return 0;
}
