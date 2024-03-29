#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
        cudaError_t err = stmt;                                               \
        if (err != cudaSuccess) {                                             \
            wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
            wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
            return -1;                                                        \
        }                                                                     \
    } while(0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float * A, float * B, float * C,
			             int numARows, int numAColumns,
			             int numBRows, int numBColumns,
			             int numCRows, int numCColumns) 
{
    //@@ Insert code to implement matrix multiplication here
    //@@ You have to use shared memory for this MP
	const int TILE_WIDTH = 16;
	
	__shared__	float ds_A[TILE_WIDTH][TILE_WIDTH];	//allocated shared memory for matrix A
	__shared__	float ds_B[TILE_WIDTH][TILE_WIDTH];	//allocated shared memory for matrix B
	
	//Declare variables for Block and Thread dimensions	
	int bx = blockIdx.x;	int by = blockIdx.y;
	int tx = threadIdx.x;  int ty = threadIdx.y;
	
	int Row = by * blockDim.y + ty;
	int Col = bx * blockDim.x + tx;

	float Cvalue = 0.0;	

	//Relation of source and target matrix
	//numAColumns == numBRows == n, numARows == m, numBColumns == k, numCRows == numARows, numCColumns == numBColumns 
	
	for (int t = 0; t < (numAColumns - 1)/TILE_WIDTH + 1; ++t) //iterate over number of Tiles in x-axis
	{ 
		//check bounds for input matrix A
				
		if(Row < numARows && t*TILE_WIDTH+tx < numAColumns) 
		{
			//load into shared memory
			
 			ds_A[ty][tx] = A[Row*numAColumns + t*TILE_WIDTH + tx];	//Accessing particular element in matrix, Row major access
		} 
		else 
		{
			//out of bounds, load 0.0
			ds_A[ty][tx] = 0.0;
		}
		
		if (t*TILE_WIDTH+ty < numBRows && Col < numBColumns) 
		{
 			ds_B[ty][tx] = B[(t*TILE_WIDTH + ty)*numBColumns + Col];
		} 
		else 
		{
			ds_B[ty][tx] = 0.0;
 		}
		__syncthreads();	//wait for all threads

		
		for (int i = 0; i < TILE_WIDTH; ++i) 
		{
			Cvalue += ds_A[ty][i] * ds_B[i][tx];
		}
		__syncthreads();
	}
	if (Row < numCRows && Col < numCColumns)
	{
		C[Row*numCColumns + Col] = Cvalue;	//wrtite calculated value to result matrix C
	}
}

		

int main(int argc, char ** argv) {
    wbArg_t args;
    float * hostA; // The A matrix
    float * hostB; // The B matrix
    float * hostC; // The output C matrix
    float * deviceA;
    float * deviceB;
    float * deviceC;
    int numARows; // number of rows in the matrix A
    int numAColumns; // number of columns in the matrix A
    int numBRows; // number of rows in the matrix B
    int numBColumns; // number of columns in the matrix B
    int numCRows; // number of rows in the matrix C (you have to set this)
    int numCColumns; // number of columns in the matrix C (you have to set this)

    args = wbArg_read(argc, argv);

    wbTime_start(Generic, "Importing data and creating memory on host");
    hostA = (float *) wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
    hostB = (float *) wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);
    //@@ Set numCRows and numCColumns
    
	//Result of input Matrix A(m*n) and B(n*k) is C(m*k)
	
    numCRows = numARows;		//Row length of Matrix A
    numCColumns = numBColumns;	//Column length of Matrix B
	
    //@@ Allocate the hostC matrix
    wbTime_stop(Generic, "Importing data and creating memory on host");
	
	hostC = (float *) malloc (numCRows*numCColumns*sizeof(float) );
							  

    wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
    wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

    wbTime_start(GPU, "Allocating GPU memory.");
    //@@ Allocate GPU memory here

	int sizeA = numARows * numAColumns * sizeof(float);	//calculate size of matrix A
							  
	int sizeB = numBRows * numBColumns * sizeof(float);	//calculate size of matrix B
							  
	int sizeC = numCRows * numCColumns * sizeof(float);	//calculate size of matrix C
							  
	cudaMalloc( (void **) &deviceA, sizeA);
	
	cudaMalloc( (void **) &deviceB, sizeB);
	
	cudaMalloc( (void **) &deviceC, sizeC);
	
	
    wbTime_stop(GPU, "Allocating GPU memory.");

    wbTime_start(GPU, "Copying input memory to the GPU.");
    //@@ Copy memory to the GPU here

	cudaMemcpy (deviceA, hostA, sizeA, cudaMemcpyHostToDevice); 
	cudaMemcpy (deviceB, hostB, sizeB, cudaMemcpyHostToDevice);
							  
							  
    wbTime_stop(GPU, "Copying input memory to the GPU.");
    
    //@@ Initialize the grid and block dimensions here
							  
	int TILE_WIDTH = 16;
	
	dim3 GridBlock( (numCColumns-1)/TILE_WIDTH + 1, (numCRows-1)/TILE_WIDTH + 1, 1);
	dim3 DimBlock( TILE_WIDTH, TILE_WIDTH, 1 );
    
    wbTime_start(Compute, "Performing CUDA computation");
    //@@ Launch the GPU Kernel here
	
	matrixMultiplyShared<<<GridBlock,DimBlock>>> (deviceA,deviceB,deviceC,
										 numARows, numAColumns,
										 numBRows, numBColumns,
										 numCRows, numCColumns);
	
    cudaThreadSynchronize();
    wbTime_stop(Compute, "Performing CUDA computation");
    
    wbTime_start(Copy, "Copying output memory to the CPU");
    //@@ Copy the GPU memory back to the CPU here
	
	cudaMemcpy (hostC,deviceC,sizeC,cudaMemcpyDeviceToHost);
							  
							  
    wbTime_stop(Copy, "Copying output memory to the CPU");

    wbTime_start(GPU, "Freeing GPU Memory");
    //@@ Free the GPU memory here

	cudaFree(deviceA);
	cudaFree(deviceB);
	cudaFree(deviceC);
							  
    wbTime_stop(GPU, "Freeing GPU Memory");

    wbSolution(args, hostC, numCRows, numCColumns);

    free(hostA);
    free(hostB);
    free(hostC);

    return 0;
}

