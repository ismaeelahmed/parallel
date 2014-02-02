// MP 1
#include	<wb.h>


#define wbCheck(stmt) do {                                                    \
	cudaError_t err = stmt;                                                \
if (err != cudaSuccess) {
\
	wbLog(ERROR, "Failed to run stmt ", #stmt);                       \
	wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));    \
	return -1;                                                        \
}                                                                     \
} while (0)


__global__ void vecAdd(float * in1, float * in2, float * out, int len) {
	//@@ Insert code to implement vector addition here

	int i = blockIdx.x * blockDim.x + threadIdx.x;

	if (i<len)
	{
		out[i] = in1[i] + in2[i];
	}


}

int main(int argc, char ** argv) {
	wbArg_t args;
	int inputLength;
	float * hostInput1;
	float * hostInput2;
	float * hostOutput;
	float * deviceInput1;
	float * deviceInput2;
	float * deviceOutput;

	args = wbArg_read(argc, argv);

	wbTime_start(Generic, "Importing data and creating memory on host");			//Timer function to calculate time
	hostInput1 = (float *)wbImport(wbArg_getInputFile(args, 0), &inputLength);		//Get input frm input dataset
	hostInput2 = (float *)wbImport(wbArg_getInputFile(args, 1), &inputLength);
	hostOutput = (float *)malloc(inputLength * sizeof(float));
	wbTime_stop(Generic, "Importing data and creating memory on host");

	wbLog(TRACE, "The input length is ", inputLength);		//log message



	wbTime_start(GPU, "Allocating GPU memory.");
	//@@ Allocate GPU memory here

	int size = inputLength * sizeof(float);			//calculate size of vectors		

	wbCheck(cudaMalloc((void **)&deviceInput1, size));		//allocate memory for vector A	on device		
	wbCheck(cudaMalloc((void **)&deviceInput2, size));		//allocate memory for vector B	on device
	wbCheck(cudaMalloc((void **)&deviceOutput, size));		//allocate memory for vector C	on device (result vector)	

	wbTime_stop(GPU, "Allocating GPU memory.");

	wbTime_start(GPU, "Copying input memory to the GPU.");
	//@@ Copy memory to the GPU here

	cudaMemcpy(deviceInput1, hostInput1, size, cudaMemcpyHostToDevice);	//Transfer Vector A from Host to Device	
	cudaMemcpy(deviceInput2, hostInput2, size, cudaMemcpyHostToDevice);	//Transfer Vector A from Host to Device	

	wbTime_stop(GPU, "Copying input memory to the GPU.");

	//@@ Initialize the grid and block dimensions here

	dim3 DimGrid((inputLength - 1) / 64 + 1, 1, 1);
	dim3 DimBlock(64, 1, 1);


	wbTime_start(Compute, "Performing CUDA computation");
	//@@ Launch the GPU Kernel here

	vecAdd << <DimGrid, DimBlock >> >(deviceInput1, deviceInput2, deviceOutput, inputLength);

	//wbCheck(cudaGetLastError());

	cudaThreadSynchronize();
	wbTime_stop(Compute, "Performing CUDA computation");

	wbTime_start(Copy, "Copying output memory to the CPU");
	//@@ Copy the GPU memory back to the CPU here

	cudaMemcpy(hostOutput, deviceOutput, size, cudaMemcpyDeviceToHost);	//Transfer result from Device to Host 

	wbTime_stop(Copy, "Copying output memory to the CPU");

	wbTime_start(GPU, "Freeing GPU Memory");
	//@@ Free the GPU memory here

	cudaFree(deviceInput1);
	cudaFree(deviceInput2);
	cudaFree(deviceOutput);

	wbTime_stop(GPU, "Freeing GPU Memory");

	wbSolution(args, hostOutput, inputLength);

	free(hostInput1);
	free(hostInput2);
	free(hostOutput);

	return 0;
}
