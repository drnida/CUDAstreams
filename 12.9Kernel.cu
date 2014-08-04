__global__ void DCSv1(float *energygrid, float *gridspacing, int *numatoms){

   int xindex = blockDim.x * blockIdx.x + threadIdx.x;
   int yindex = blockDim.y * blockIdx.y + threadIdx.y;
   int outaddr = yindex * numAtoms + xindex;
   int num = *numatoms;   
   if(xindex > num || yindex > num) return;

   float curenergy = energygrid[outaddr];
   float coorx = (*gridspacing) * xindex;
   float coory = (*gridspacing) * yindex;

   float gridspacing_coalesce = gridspacing*BLOCKSIZEX;

   int atomid;

   float energyval=0.0f;


   for (atomid=0; atomid<(*numatoms); atomid++) {
      float dy = coory - atominfo[atomid].y;
      float dyz2 = dy*dy- atominfo[atomid].z;
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
      energyvalx8 += atominfo[atomid].w*rsqrtf(dx8*dx8 + dyz2); 
   }
   energygrid[outaddr + 0 * BLOCKSIZEX ] = energyval1;
   energygrid[outaddr + 1 * BLOCKSIZEX ] = energyval1;
   energygrid[outaddr + 2 * BLOCKSIZEX ] = energyval1;
   energygrid[outaddr + 3 * BLOCKSIZEX ] = energyval1;
   energygrid[outaddr + 4 * BLOCKSIZEX ] = energyval1;
   energygrid[outaddr + 5 * BLOCKSIZEX ] = energyval1;
   energygrid[outaddr + 6 * BLOCKSIZEX ] = energyval1;
   energygrid[outaddr + 7 * BLOCKSIZEX ] = energyval1;


}
