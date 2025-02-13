#include <stdio.h>
#include <iostream>
#include <cstdlib>
#include <functional>
#include <chrono>
#include <string>

#define ARRAY_SIZE 1024
#define NUM_ITERATIONS 1000
#define ARRAY_SIZE_IN_BYTES (sizeof(unsigned int) * (ARRAY_SIZE))

/* Declare  statically two arrays of ARRAY_SIZE each */
unsigned int cpu_input_1[ARRAY_SIZE];
unsigned int cpu_input_2[ARRAY_SIZE];
unsigned int cpu_output[ARRAY_SIZE];

__global__
void calcWithBranch(char c,
	unsigned int* gpu_input_1,
	unsigned int* gpu_input_2,
	unsigned int* gpu_output)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;

	switch (c)
	{
	case '+':
		gpu_output[thread_idx] = gpu_input_1[thread_idx] +
			gpu_input_2[thread_idx];
		break;
	case '-':
		gpu_output[thread_idx] = gpu_input_1[thread_idx] -
			gpu_input_2[thread_idx];
		break;
	case '*':
		gpu_output[thread_idx] = gpu_input_1[thread_idx] *
			gpu_input_2[thread_idx];
		break;
	case '%':
		gpu_output[thread_idx] = gpu_input_1[thread_idx] %
			gpu_input_2[thread_idx];
		break;
	default:
		break;
	}
}

__global__
void add(char c,
	unsigned int* gpu_input_1,
	unsigned int* gpu_input_2,
	unsigned int* gpu_output)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	gpu_output[thread_idx] = gpu_input_1[thread_idx] + gpu_input_2[thread_idx];
}

__global__
void subtract(char c,
	unsigned int* gpu_input_1,
	unsigned int* gpu_input_2,
	unsigned int* gpu_output)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	gpu_output[thread_idx] = gpu_input_1[thread_idx] - gpu_input_2[thread_idx];
}

__global__
void mult(char c,
	unsigned int* gpu_input_1,
	unsigned int* gpu_input_2,
	unsigned int* gpu_output)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	gpu_output[thread_idx] = gpu_input_1[thread_idx] * gpu_input_2[thread_idx];
}

__global__
void mod(char c,
	unsigned int* gpu_input_1,
	unsigned int* gpu_input_2,
	unsigned int* gpu_output)
{
	const unsigned int thread_idx = (blockIdx.x * blockDim.x) + threadIdx.x;
	gpu_output[thread_idx] = gpu_input_1[thread_idx] % gpu_input_2[thread_idx];
}

static unsigned int totalCPUTime = 0, totalGPUTime = 0;

void callWithData(char op,
	unsigned int numBlocks, unsigned int numThreads,
	unsigned int* gpu_input_1, unsigned int* gpu_input_2,
	void (*gpuCall)(char, unsigned int*, unsigned int*, unsigned int*))
{
	using clock = std::chrono::high_resolution_clock;

	for (unsigned int i = 0; i < NUM_ITERATIONS; i++)
	{
		clock::time_point t1_start = clock::now();

		unsigned int* gpu_output;
		cudaMalloc((void**)&gpu_output, ARRAY_SIZE_IN_BYTES);
		cudaMemcpy(gpu_output, cpu_output, ARRAY_SIZE_IN_BYTES,
			cudaMemcpyHostToDevice);

		clock::time_point t2_start = clock::now();

		gpuCall << <numBlocks, numThreads >> > (op,
			gpu_input_1, gpu_input_2, gpu_output);

		cudaDeviceSynchronize();

		clock::time_point t2_end = clock::now();

		cudaMemcpy(cpu_output, gpu_output, ARRAY_SIZE_IN_BYTES,
			cudaMemcpyDeviceToHost);
		cudaFree(gpu_output);

		//for (unsigned int i = 0; i < ARRAY_SIZE; i++)
		//{
		//	printf("Operation: %3u %c %1u = %3u\n",
		//		cpu_input_1[i], op, cpu_input_2[i], cpu_output[i]);
		//}

		clock::time_point t1_end = clock::now();
		totalCPUTime += std::chrono::duration_cast<std::chrono::microseconds>(
			t1_end - t1_start).count();
		totalGPUTime += std::chrono::duration_cast<std::chrono::microseconds>(
			t2_end - t2_start).count();
	}
}

void timedCall(char op,
	unsigned int numBlocks, unsigned int numThreads,
	unsigned int* gpu_input_1, unsigned int* gpu_input_2,
	void (*gpuCall)(char, unsigned int*, unsigned int*, unsigned int*))
{
	callWithData(op, numBlocks, numThreads, gpu_input_1,
		gpu_input_2, gpuCall);

	std::cout << "Host time taken (" << op << ", unbranched, avg 1000): "
		<< (float)totalCPUTime / NUM_ITERATIONS
		<< " microseconds" << std::endl;
	std::cout << "Device time taken (" << op << ", unbranched, avg 1000): "
		<< (float)totalGPUTime / NUM_ITERATIONS
		<< " microseconds" << std::endl;

	totalCPUTime = 0;
	totalGPUTime = 0;

	callWithData(op, numBlocks, numThreads, gpu_input_1, gpu_input_2,
		&calcWithBranch);

	std::cout << "Host time taken (" << op << ", branched, avg 1000): "
		<< (float)totalCPUTime / NUM_ITERATIONS
		<< " microseconds" << std::endl;
	std::cout << "Device time taken (" << op << ", branched, avg 1000): "
		<< (float)totalGPUTime / NUM_ITERATIONS
		<< " microseconds" << std::endl;

	totalCPUTime = 0;
	totalGPUTime = 0;
}

int main(int argc, char* argv[])
{
	// read command line arguments
	int numThreads = (1 << 20);
	int blockSize = 256;

	if (argc >= 2) {
		numThreads = atoi(argv[1]);
	}
	if (argc >= 3) {
		blockSize = atoi(argv[2]);
	}

	int numBlocks = numThreads / blockSize;

	// validate command line arguments
	if (numThreads % blockSize != 0) {
		++numBlocks;
		numThreads = numBlocks * blockSize;

		printf("Warning: Total thread count is not evenly divisible by the block size\n");
		printf("The total number of threads will be rounded up to %d\n", numThreads);
	}

	// Setup
	unsigned int* gpu_input_1;
	unsigned int* gpu_input_2;

	for (unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		cpu_input_1[i] = rand() % (numThreads + 1);
	}

	for (unsigned int i = 0; i < ARRAY_SIZE; i++)
	{
		cpu_input_2[i] = rand() % 4;
	}

	cudaMalloc((void**)&gpu_input_1, ARRAY_SIZE_IN_BYTES);
	cudaMalloc((void**)&gpu_input_2, ARRAY_SIZE_IN_BYTES);
	cudaMemcpy(gpu_input_1, cpu_input_1, ARRAY_SIZE_IN_BYTES,
		cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_input_2, cpu_input_2, ARRAY_SIZE_IN_BYTES,
		cudaMemcpyHostToDevice);
	
	timedCall('+', numBlocks, numThreads, gpu_input_1,
		gpu_input_2, &add);
	timedCall('-', numBlocks, numThreads, gpu_input_1,
		gpu_input_2, &subtract);
	timedCall('*', numBlocks, numThreads, gpu_input_1,
		gpu_input_2, &mult);
	timedCall('%', numBlocks, numThreads, gpu_input_1,
		gpu_input_2, &mod);

	// Teardown
	cudaFree(gpu_input_1);
	cudaFree(gpu_input_2);

	return EXIT_SUCCESS;
}
