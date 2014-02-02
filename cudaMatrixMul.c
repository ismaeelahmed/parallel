#include    <wb.h>

#define wbCheck(stmt) do {                                                    \
	cudaError_t err = stmt;                                               \
if (err != cudaSuccess) {
\
	wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
	wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
	return -1;                                                        \
}                                                                     \
} while (0)

// Compute C = A * B
__global__ void matrixMultiply(float * A, float * B, float * C,
int numARows, int numAColumns,
int numBRows, int numBColumns,
int numCRows, int numCColumns) {
	//@@ Insert code to implement matrix multiplication here

	int Row = blockIdx.y * blockDim.y + threadIdx.y;
	int Col = blockIdx.x * blockDim.x + threadIdx.x;


	if ((Row < numCRows) && (Col < numCColumns))
	{
		float Cvalue = 0.0;
		for (int i = 0; i< numAColumns; ++i)
		{
			Cvalue += A[Row*numAColumns + i] * B[Col + i*numBColumns];
		}
		C[Row*numCColumns + Col] = Cvalue;
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
	hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows, &numAColumns);
	hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows, &numBColumns);


	//@@ Set numCRows and numCColumns

	//Result of input Matrix A(m*n) and B(n*k) is C(m*k)

	numCRows = numARows;		//Row length of Matrix A
	numCColumns = numBColumns;	//Column length of Matrix B

	//@@ Allocate the hostC matrix

	hostC = (float *)malloc(numCRows*numCColumns*sizeof(float));
	wbLog(TRACE, "The dimensions of C are ", numCRows, " x ", numCColumns, " = total : ", hostC);


	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
	wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here

	int size_A = numARows * numAColumns * sizeof(float);	//Total size in bytes of matrix A
	int size_B = numBRows * numBColumns * sizeof(float);
	int size_C = numCRows * numCColumns * sizeof(float);

	wbLog(TRACE, "Size of A :: ", size_A);		//Log message


	cudaMalloc((void **)&deviceA, size_A);		//Allocate memory on device for matrix A
	cudaMalloc((void **)&deviceB, size_B);
	cudaMalloc((void **)&deviceC, size_C);




	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	//@@ Copy memory to the GPU here

	cudaMemcpy(deviceA, hostA, size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(deviceB, hostB, size_B, cudaMemcpyHostToDevice);



	wbTime_stop(GPU, "Copying input memory to the GPU.");

	//@@ Initialize the grid and block dimensions here

	int TILE_WIDTH = 8;

	dim3 GridBlock((numCColumns - 1) / TILE_WIDTH + 1, (numCRows - 1) / TILE_WIDTH + 1, 1);
	dim3 DimBlock(TILE_WIDTH, TILE_WIDTH, 1);


	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Launch the GPU Kernel here

	matrixMultiply << <GridBlock, DimBlock >> >(deviceA, deviceB, deviceC,
		numARows, numAColumns,
		numBRows, numBColumns,
		numCRows, numCColumns);

	wbCheck(cudaGetLastError());

	cudaThreadSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	//@@ Copy the GPU memory back to the CPU here

	cudaMemcpy(hostC, deviceC, size_C, cudaMemcpyDeviceToHost);


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


