#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <fstream>
#include <xmmintrin.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <boost/timer/timer.hpp>
#include <boost/format.hpp>
#include <cub/util_allocator.cuh>
#include <cub/device/device_radix_sort.cuh>
#include <test/test_util.h>
#include "util.h"
#include "cpu_sort.h"


template<typename T>
struct hybridDispatchParams
{
	size_t gpuChunkSize;
	size_t cpuChunkSize;
	size_t mergeBlockSize;
	size_t multiwayBlockSize;
	size_t gpuMergeLen;
	size_t gpuMultiwayLen;
	
	hybridDispatchParams(size_t dataLen)
	{
		int mergeFactor = 3;
		int multiwayFactor = 1;
		int cacheFactor = 2; //what is the most suitable cache size?
		gpuChunkSize = dataLen / (mergeFactor + 1);
		cpuChunkSize = gpuChunkSize / omp_get_max_threads();
		mergeBlockSize = cacheSizeInByte() / (cacheFactor * sizeof(T));
		multiwayBlockSize = 512;
		gpuMergeLen = gpuChunkSize * mergeFactor;
		gpuMultiwayLen = dataLen * multiwayFactor / (multiwayFactor + 1);
	}
};

template<typename T>
struct hybridDispatchParams3
{
	size_t gpuChunkLen;
	size_t cpuChunkLen;
	size_t cpuBlockLen;
	size_t gpuPart;
	size_t cpuPart;
	size_t multiWayUpdate;
	int threads;
	int medianFactor;
	/*size_t multiwayBlockSize;
	  size_t gpuMergeLen;
	  size_t gpuMultiwayLen;*/
	
	//now, dataLen must be power of 2.
	//TODO:more portable and flexible partition method
	hybridDispatchParams3(size_t dataLen)
	{
		threads = omp_get_max_threads();
		size_t baseChunkLen = lastPower2(cacheSizeInByte3() / (2 * sizeof(T)));
		multiWayUpdate = 0;
		medianFactor = 4;
		//TODO: when GPU global memory fewer than the half of dataLen?
		//TODO: other way to cut data lists to more fit in capacity of GPU
		//TODO: to find a more portable solution
		if (dataLen <= baseChunkLen)
		{
			gpuPart = 0;
		}
		else if(dataLen < 1<<23)
			gpuPart = dataLen >> 1;
		else if (dataLen < 1 << 27)
		{
			//gpuPart = dataLen >> 1;
			gpuPart = (dataLen >> 2) * 3;
		}
		// else if (dataLen < 1 << 28)
		// {
		// 	//gpuChunkLen = (dataLen >> 2) * 3;
		// 	gpuPart = (dataLen >> 2) * 3;
		// }
		else if (dataLen == 1 << 28)
		{
			gpuPart = (dataLen >> 3) * 7;
		}
		else if(dataLen < 1 << 30)
		{
			gpuPart = (dataLen >> 2) * 3;
		}
		else
		{
			gpuPart = (dataLen >> 3) * 7;
		}
		gpuChunkLen = std::min(size_t(1 << 27), gpuPart);
		if(gpuChunkLen == 1<<27 && gpuPart < gpuChunkLen << 1) gpuChunkLen = gpuPart >> 1;
		cpuPart = dataLen - gpuPart;
		if(dataLen < 1 << 23)
			multiWayUpdate = gpuPart;
		else if(dataLen == 1<<23)
			multiWayUpdate = (dataLen >> 1);
		else if(dataLen <= 1<<25)
			multiWayUpdate = (dataLen >> 4) * 11;
		else if(dataLen < 1 << 27)
			//multiWayUpdate = dataLen >> 2;
			multiWayUpdate = dataLen >> 1;
		else if(dataLen == 1 << 28)
			multiWayUpdate = (dataLen >> 3) * 5;
		else if(dataLen < 1 << 30)
			multiWayUpdate =  dataLen >> 1;
		else
			multiWayUpdate = (dataLen >> 3) * 5;
		cpuChunkLen =
			gpuChunkLen > 0 ? std::min(gpuChunkLen, baseChunkLen) : dataLen;
		cpuBlockLen = cpuChunkLen / threads;
		std::cout << "gpupart: " << gpuPart << " gpuchunk: " << gpuChunkLen << std::endl;
	}
	
	hybridDispatchParams3(size_t dataLen, size_t gpuPartLen)
	{
		threads = omp_get_max_threads();
		size_t baseChunkLen = lastPower2(cacheSizeInByte3() / (2 * sizeof(T)));
		//std::cout << "baseChunkLen = " << baseChunkLen << std::endl;
		gpuPart = gpuPartLen;
		gpuChunkLen = std::min(gpuPart, size_t(1 << 27));
		cpuPart = dataLen - gpuPartLen;
		cpuChunkLen = std::min(baseChunkLen, cpuPart);
		cpuBlockLen = cpuChunkLen / threads;
		medianFactor = 4;
	}
};

float gpu_sort(DoubleBuffer<float> &data, size_t dataLen, size_t blockLen,
			   int sSelector, int dSelector);
void gpu_sort(DoubleBuffer<float> &data, hybridDispatchParams3<float> &params,
			  int sSelector, int tSelector);
void gpu_sort_test(float *data, rsize_t dataLen);
void hybrid_sort3(float *data, size_t dataLen, double (&results)[2]);
void mergeTest(size_t minLen, size_t maxLen, int seed);
void hybrid_sort(float *data, size_t dataLen);
void hybrid_sort_test(size_t minLen, size_t maxLen, int seed);

int main(int argc, char **argv)
{
	rsize_t dataLen = 1 << 30; //default length of sorted data
	int seed = 1979090303;  //default seed for generate random data sequence
	CommandLineArgs args(argc, argv);
	args.GetCmdLineArgument("l", dataLen);
	args.GetCmdLineArgument("s", seed);
	args.DeviceInit();
	hybrid_sort_test(1<<16, 1<<30, seed);
	//mergeTest(1<<16, 1<<30, seed);
	//multiWayTest(1<<16, 1<<28, seed);
	//multiWayTestMedian(1<<20, 1<<23, seed);
	/*float *data = new float[dataLen];
	  GenerateData(seed, data, dataLen);
	  double times[2];
	  hybrid_sort(data, dataLen, times);*/
	/*gpu_sort_test(data, dataLen);*/
	//gpu_sort_serial(data, dataLen, dataLen);
	//delete [] data;
	/*for (int dlf = 20; dlf < 26; ++dlf)
	  {
	  dataLen = 1 << dlf;
	  std::cout << "data length: " << dataLen << std::endl;
	  float *data = new float[dataLen];
	  GenerateData(seed, data, dataLen);
	  hybrid_sort(data, dataLen);
	  delete [] data;
	  //std::cout << "loop time: " << dlf << std::endl;
	  }*/
	/*dataLen = 1 << 23;
	  float *data = new float[dataLen];
	  GenerateData(seed, data, dataLen);
	  hybrid_sort3(data, dataLen);*/
	//delete [] data;
	std::cout << "test complete." << std::endl;
	//resultTest(cpu_sort_sse_parallel(hdata, dataLen), dataLen);
	//resultTest(mergeSortInBlockParallel(dataIn, dataOut, dataLen), dataLen);
	//gpu_sort(dataIn, dataLen, dataLen >> 2);
	//gpu_sort_serial(dataIn, dataLen, dataLen >>2);
	/*#pragma omp parallel
	  {
	  omp_set_nested(1);
	  #pragma omp single nowait
	  std::cout << "single run" << omp_get_nested() << std::endl;
	  gpu_sort(data, dataLen);
	  #pragma omp single
	  resultTest(data, dataLen);
	  #pragma omp parallel
	  std::cout << omp_get_thread_num();
	  }*/
	return 0;
}

//using stream to overlap kernal excution and data transfer between CPU and GPU.
//all sorting task broken to 2 parts, the first will overlap data upload to GPU,
//the second will overlap data download from CPU.
float gpu_sort(DoubleBuffer<float> &data, size_t dataLen, size_t blockLen,
			   int sSelector, int dSelector)
{
	int blockNum = dataLen / blockLen;
	size_t blockBytes = sizeof(float) * blockLen;
	cudaStream_t *streams = new cudaStream_t[blockNum];
	for (int i = 0; i < blockNum; ++i)
		cudaStreamCreate(&streams[i]);
    cub::DoubleBuffer<float> d_keys;
	//int gSelector = 1;
    cub::CachingDeviceAllocator cda;
    cda.DeviceAllocate((void**) &d_keys.d_buffers[0], sizeof(float) * dataLen);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[1], sizeof(float) * dataLen);
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
								   blockLen);
	cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);
	cudaMemcpyAsync(d_keys.d_buffers[0], data.buffers[sSelector], blockBytes,
					cudaMemcpyHostToDevice, streams[0]);
	int remain_to_upload = blockNum - 1;
	int upload_loop = std::max(1, remain_to_upload >> 1);
	size_t offset = 0;
	size_t up_offset = blockLen;
	for (int i = 0; i < upload_loop; ++i)
	{
		cub::DoubleBuffer<float> chunk(d_keys.d_buffers[0] + offset,
									   d_keys.d_buffers[1] + offset);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
									   chunk, blockLen, 0, sizeof(float) * 8,
									   streams[i]);
		int upload_blocks =
			((remain_to_upload < 2) ? 0 : 2) + (remain_to_upload % 2);
		cudaMemcpyAsync(d_keys.d_buffers[0] + up_offset,
						data.buffers[sSelector] + up_offset,
						upload_blocks * blockBytes, cudaMemcpyHostToDevice,
						streams[i + 1]);
		remain_to_upload -= upload_blocks;
		up_offset += upload_blocks * blockLen;
		offset += blockLen;
	}
	int remain_to_donwload = upload_loop;
	size_t down_offset = 0;
	for (int i = upload_loop; i < blockNum; ++i)
	{
		cub::DoubleBuffer<float> chunk(d_keys.d_buffers[0] + offset,
									   d_keys.d_buffers[1] + offset);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
									   chunk, blockLen, 0, sizeof(float) * 8,
									   streams[i]);
		int dowload_blocks = 1 + (remain_to_donwload > 1);
		cudaMemcpyAsync(data.buffers[dSelector] + down_offset,
						d_keys.d_buffers[1] + down_offset,
						dowload_blocks * blockBytes, cudaMemcpyDeviceToHost,
						streams[i - 1]);
		remain_to_donwload -= (dowload_blocks - 1);
		down_offset += dowload_blocks * blockLen;
		offset += blockLen;
	}
	cudaMemcpyAsync(data.buffers[dSelector] + dataLen - blockLen,
					d_keys.d_buffers[1] + dataLen - blockLen, blockBytes,
					cudaMemcpyDeviceToHost, streams[blockNum - 1]);
	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float sort_time;
	cudaEventElapsedTime(&sort_time, start, stop);
	std::cout << "time used on gpu sort loop: " << sort_time << std::endl;
	for (int i = 0; i < blockNum; ++i)
		cudaStreamDestroy(streams[i]);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cda.DeviceFree(d_temp_storage);
	cda.DeviceFree(d_keys.d_buffers[0]);
	cda.DeviceFree(d_keys.d_buffers[1]);
	return sort_time;
}


//sSelector is source selector that provide data to sort. tSelector is target
//seclector, specify which buffer the result should be copied to.
void gpu_sort(DoubleBuffer<float> &data, hybridDispatchParams3<float> &params,
			  int sSelector, int tSelector)
{
    cub::DoubleBuffer<float> d_keys;
    cub::CachingDeviceAllocator cda(false);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[0],
					   sizeof(float) * params.gpuChunkLen);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[1],
					   sizeof(float) * params.gpuChunkLen);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
								   params.gpuChunkLen);
	cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	float *ptrIn = data.buffers[sSelector] + params.cpuPart;
	float *ptrOut= data.buffers[tSelector] + params.cpuPart;
	for(size_t i = 0; i < params.gpuPart; i += params.gpuChunkLen)
	{
		cudaMemcpyAsync(d_keys.Current(), ptrIn + i,
						sizeof(float) * params.gpuChunkLen,
						cudaMemcpyHostToDevice);
		cudaMemsetAsync(d_keys.d_buffers[d_keys.selector ^ 1], 0,
						sizeof(float) * params.gpuChunkLen);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
									   d_keys, params.gpuChunkLen);
		cudaMemcpyAsync(ptrOut + i, d_keys.Current(),
						sizeof(float) * params.gpuChunkLen,
						cudaMemcpyDeviceToHost);
	}
	cda.DeviceFree(d_temp_storage);
	cda.DeviceFree(d_keys.d_buffers[0]);
	cda.DeviceFree(d_keys.d_buffers[1]);
}

void multiWayMergeGPU(DoubleBuffer<float> &data, size_t *upperBound,
					  int chunkNum, hybridDispatchParams3<float> &params)
{
	if(params.multiWayUpdate == params.gpuPart) return;
	size_t gpuLen = params.gpuPart - params.multiWayUpdate;
	size_t chunkLen = std::min(params.gpuChunkLen, gpuLen);
	//TODO:generalize
	if(gpuLen == 1<<27) chunkLen = 1 << 27;
	//std::cout << gpuLen << " " << chunkLen << std::endl;
	size_t *quantileStart = new size_t[chunkNum];
	size_t *quantileEnd = new size_t[chunkNum];
	size_t *loopUBound = new size_t[chunkNum];
	size_t *loopLBound = new size_t[chunkNum];
	DoubleBuffer<size_t> quantile(quantileStart, quantileEnd);
	DoubleBuffer<size_t> bound(loopLBound, loopUBound);
	quantile.buffers[0][0] = 0;
	std::copy(upperBound, upperBound + chunkNum - 1, quantile.buffers[0] + 1);
	//TODO:in quantilecompute function, whether startoffset is 0 must be
	//checked.
	quantileCompute(data.buffers[data.selector], quantile, bound, upperBound,
					chunkNum, params.cpuPart + params.multiWayUpdate, true);
	std::copy(quantile.buffers[1], quantile.buffers[1] + chunkNum,
			  quantile.buffers[0]);
    cub::DoubleBuffer<float> d_keys;
	//must use false there, because we need actively free all cached memory.
    cub::CachingDeviceAllocator cda(false);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[0], sizeof(float) * chunkLen);
    cda.DeviceAllocate((void**) &d_keys.d_buffers[1], sizeof(float) * chunkLen);
    void *d_temp_storage = NULL;
    size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys,
								   chunkLen);
	cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	float *ptrOut =
		data.buffers[data.selector ^ 1] + params.cpuPart + params.multiWayUpdate;
	//TODO:may need while loop, because chunklen is not equally parted.
	for (size_t i = 0; i < gpuLen; i += chunkLen)
	{
		quantileCompute(data.Current(), quantile, bound, upperBound, chunkNum,
						chunkLen);
		size_t tempLen = 0;
		for (int j = 0; j < chunkNum; ++j)
		{
			size_t len = quantile.buffers[1][j] - quantile.buffers[0][j];
			cudaMemcpyAsync(d_keys.d_buffers[d_keys.selector] + tempLen,
							data.buffers[data.selector] + quantile.buffers[0][j],
							sizeof(float) * len, cudaMemcpyHostToDevice);
			tempLen += len;
		}
		cudaMemsetAsync(d_keys.d_buffers[d_keys.selector ^ 1], 0,
						sizeof(float) * chunkLen);
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
									   d_keys, chunkLen);
		cudaMemcpyAsync(ptrOut + i, d_keys.Current(), sizeof(float) * chunkLen,
						cudaMemcpyDeviceToHost);
		std::copy(quantile.buffers[1], quantile.buffers[1] + chunkNum,
				  quantile.buffers[0]);
	}
	cda.DeviceFree(d_temp_storage);
	cda.DeviceFree(d_keys.d_buffers[0]);
	cda.DeviceFree(d_keys.d_buffers[1]);
	delete [] quantileStart;
	delete [] quantileEnd;
	delete [] loopUBound;
	delete [] loopLBound;
}

void gpu_sort_test(float *data, rsize_t dataLen)
{
	if (dataLen < (1 << 20) || dataLen > (1 << 28))
	{
		std::cout << "data length too short or too long!" << std::endl;
		return;
	}
	std::ofstream rFile("/home/aloneranger/source_code/Hybrid_Sort/result.txt",
						std::ios::app);
	rFile << "gpu kernel and transfer test" << std::endl
		  << boost::format("%1%%|15t|") % "data length"
		  << boost::format("%1%%|15t|") % "transfer time"
		  << boost::format("%1%%|15t|") % "kernel time"
		  << std::endl;
	cub::DoubleBuffer<float> d_keys;
    cub::CachingDeviceAllocator cda(true);
    cda.DeviceAllocate((void**)&d_keys.d_buffers[0], sizeof(float) * dataLen);
    cda.DeviceAllocate((void**)&d_keys.d_buffers[1], sizeof(float) * dataLen);
	void *d_temp_storage = NULL;
	size_t temp_storage_bytes = 0;
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
								   d_keys, dataLen);
	cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
	cudaMemcpy(d_keys.d_buffers[0], data, sizeof(float) * dataLen,
			   cudaMemcpyHostToDevice);
	cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
								   d_keys, dataLen);
	float *temp = new float[dataLen];
	cudaMemcpy(temp, d_keys.Current(), sizeof(float) * dataLen,
			   cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	resultTest(temp, dataLen);
	std::cout << "warm up complete. " << temp[0] << " " << temp[dataLen - 1]
			  << std::endl;
	delete [] temp;
	cda.DeviceFree(d_temp_storage);
	cudaMemset(d_keys.d_buffers[0], 0, sizeof(float) * dataLen);
	cudaMemset(d_keys.d_buffers[1], 0, sizeof(float) * dataLen);
	d_keys.selector = 0;
	int test_time = 50;
	for (size_t chunk_size = 1 << 17; chunk_size <= dataLen; chunk_size *= 2)
	{
		std::cout << chunk_size << std::endl;
		d_temp_storage = NULL;
		temp_storage_bytes = 0;
		cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
									   d_keys, chunk_size);
		cda.DeviceAllocate(&d_temp_storage, temp_storage_bytes);
		float transfer_time = 0.0, kernel_time = 0.0;
		size_t offset = 0;
		for (int i = 0; i < test_time; ++i)
		{
			cudaEvent_t tStart, tStop, sStart, sStop;
			cudaEventCreate(&tStart);
			cudaEventCreate(&tStop);
			cudaEventCreate(&sStart);
			cudaEventCreate(&sStop);
			if (offset == dataLen) {
				offset = 0;
				cudaMemset(d_keys.d_buffers[0], 0, sizeof(float) * dataLen);
				cudaMemset(d_keys.d_buffers[1], 0, sizeof(float) * dataLen);
			}
			cudaEventRecord(tStart, 0);
			cudaMemcpyAsync(d_keys.d_buffers[0] + offset, data + offset,
							sizeof(float) * chunk_size,
							cudaMemcpyHostToDevice, 0);
			cudaEventRecord(tStop, 0);
			cub::DoubleBuffer<float> chunk(d_keys.d_buffers[0] + offset,
										   d_keys.d_buffers[1] + offset);
			cudaEventRecord(sStart, 0);
			cub::DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes,
										   chunk, chunk_size);
			cudaEventRecord(sStop, 0);
			cudaDeviceSynchronize();
			float ttime;
			cudaEventElapsedTime(&ttime, tStart, tStop);
			transfer_time += ttime;
			float ktime;
			cudaEventElapsedTime(&ktime, sStart, sStop);
			kernel_time += ktime;
			offset += chunk_size;
			cudaEventDestroy(tStart);
			cudaEventDestroy(tStop);
			cudaEventDestroy(sStart);
			cudaEventDestroy(sStop);
		}
		rFile << boost::format("%1%%|15t|") % chunk_size
			  << boost::format("%1%%|15t|") % (transfer_time / test_time)
			  << boost::format("%1%%|15t|") % (kernel_time / test_time)
			  << std::endl;
		cda.DeviceFree(d_temp_storage);
	}
	
    /*cudaMemcpy(data, d_keys.Current(), sizeof(float) * dataLen,
	  cudaMemcpyDeviceToHost);*/
	cda.DeviceFree(d_keys.d_buffers[0]);
	cda.DeviceFree(d_keys.d_buffers[1]);
	rFile << std::endl << std::endl;
	rFile.close();
}

void inline mergeStep1(DoubleBuffer<float> &data, size_t startOffset,
					   hybridDispatchParams3<float> &params)
{
#pragma omp parallel for schedule(dynamic)
	for (size_t j = startOffset; j < startOffset + params.cpuChunkLen;
		 j += params.cpuBlockLen)
	{
		DoubleBuffer<float> block(data.buffers[data.selector] + j,
								  data.buffers[data.selector ^ 1] + j);
		singleThreadMerge(block, params.cpuBlockLen);
	}
}

void inline mergeStep2(DoubleBuffer<float> &data, size_t startOffset,
					   hybridDispatchParams3<float> &params)
{
	DoubleBuffer<float> chunk(data.buffers[data.selector] + startOffset,
							  data.buffers[data.selector ^ 1] + startOffset);
	updateSelectorGeneral(chunk.selector, 8, params.cpuBlockLen);
	multiThreadMerge(chunk, params.cpuChunkLen, params.threads,
					 params.cpuBlockLen);
}

void chunkMerge(DoubleBuffer<float> &data, hybridDispatchParams3<float> &params,
				size_t startOffset = 0)
{
	for (size_t i = startOffset; i < params.cpuPart; i += params.cpuChunkLen)
	{
		mergeStep1(data, i, params);
		mergeStep2(data, i, params);
	}
	updateSelectorGeneral(data.selector, 8, params.cpuChunkLen);
}

void medianMerge(DoubleBuffer<float> &data, hybridDispatchParams3<float> &params)
{
	int chunkNum = params.cpuPart / params.cpuChunkLen;
	if(chunkNum > params.medianFactor) return;
	size_t stride = params.cpuChunkLen << 1;
	while (chunkNum > 1)
	{
		for (size_t j = 0; j < params.cpuPart; j += stride)
		{
			DoubleBuffer<float> chunk(data.buffers[data.selector] + j,
									  data.buffers[data.selector ^ 1] + j);
			multiThreadMerge(chunk, stride, 2, params.cpuBlockLen);
		}
		chunkNum >>= 1;
		stride <<= 1;
		data.selector ^= 1;
	}
}

void multiWayMergeSet(DoubleBuffer<float> &data, size_t *upperBound,
					  size_t chunkNum, size_t *quantileSet, size_t blockLen,
					  size_t blockNum, size_t startOffset)
{
	int bufferNum = omp_get_max_threads();
    size_t *loopUBound = new size_t[chunkNum * bufferNum];
    size_t *loopLBound = new size_t[chunkNum * bufferNum];
	DoubleBuffer<size_t> bound(loopLBound, loopUBound);
	quantileSetCompute(data, quantileSet, bound, upperBound, chunkNum,
					   blockLen, blockNum);
	float *mwBuffer =
		(float*)_mm_malloc(blockLen * bufferNum * sizeof(float), 16);
	float **start = new float*[chunkNum * bufferNum];
	float **end   = new float*[chunkNum * bufferNum];
#pragma omp parallel for schedule(dynamic)
	for(size_t i = 0; i < blockNum; ++i)
	{
		int w = omp_get_thread_num();
		std::vector<float> unalignVec;
		DoubleBuffer<size_t> quantile(quantileSet + i * chunkNum,
									  quantileSet + i * chunkNum + chunkNum);
		multiWayMergeBitonic(data, chunkNum, mwBuffer + w * blockLen,
							 startOffset + i * blockLen, quantile, unalignVec,
							 start + w * chunkNum, end + w * chunkNum);
	}
	delete [] loopLBound;
	delete [] loopUBound;
	_mm_free(mwBuffer);
	delete [] start;
	delete [] end;
}

void multiWayMergeRecursion(DoubleBuffer<float> &data, size_t chunkLen,
							size_t chunkNum, size_t blockLen, int wayLen = 16)
{
	if(chunkNum == 1) return;
	int step = std::min(chunkNum, size_t(wayLen));
	size_t *upperBound = new size_t[step];
	size_t stride = step * chunkLen;
	size_t blockNum = stride / blockLen;
	size_t *quantileSet = new size_t[chunkNum * (blockNum + 1)];
	size_t offset = 0;
	for(int i = 0; i < chunkNum; i += step)
	{
		std::fill(upperBound, upperBound + step, chunkLen);
		upperBound[0] += offset;
		std::partial_sum(upperBound, upperBound + step, upperBound);
		//TODO: initial of first array of quantile must all move into quantile
		//compute functions.
		quantileSet[0] = offset;
		std::copy(upperBound, upperBound + chunkNum - 1, quantileSet + 1);
		multiWayMergeSet(data, upperBound, step, quantileSet, blockLen,
						 blockNum, offset);
		std::copy(quantileSet + chunkNum * blockNum,
				  quantileSet + chunkNum * blockNum + chunkNum, quantileSet);
		offset += stride;
	}
	delete [] upperBound;
	delete [] quantileSet;
	data.selector ^= 1;
	multiWayMergeRecursion(data, stride, chunkNum / step, blockLen, wayLen);
}

void mergeStep3(DoubleBuffer<float> &data, hybridDispatchParams3<float> &params,
				int wayLen = 8)
{
	size_t chunkNum = params.cpuPart / params.cpuChunkLen;
	if(chunkNum <= params.medianFactor) return;
	multiWayMergeRecursion(data, params.cpuChunkLen, chunkNum,
						   params.cpuBlockLen);
}

void hybrid_sort3(float *data, size_t dataLen, double (&results)[2])
{
	float* dataIn = (float*)_mm_malloc(dataLen * sizeof(float), 16);
	float* dataOut= (float*)_mm_malloc(dataLen * sizeof(float), 16);
	std::copy(data, data + dataLen, dataIn);
	DoubleBuffer<float> hdata(dataIn, dataOut);
	
	hybridDispatchParams3<float> params(dataLen, 0);
	chunkMerge(hdata, params);
	medianMerge(hdata, params);
	mergeStep3(hdata, params);
	resultTest(hdata.Current(), dataLen);
	
	const int test_time = 1;
	double cmerge = 0.0, mmerge = 0.0;
	for (int i = 0; i < test_time; ++i)
	{
		std::copy(data, data + dataLen, dataIn);
		std::fill(dataOut, dataOut + dataLen, 0);
		hdata.selector = 0;
		double start, end;
		start = omp_get_wtime();
		chunkMerge(hdata, params);
		medianMerge(hdata, params);
		end = omp_get_wtime();
		cmerge += (end - start);
		start = omp_get_wtime();
		mergeStep3(hdata, params);
		end = omp_get_wtime();
		mmerge += (end - start);
	}
	results[0] = cmerge / test_time, results[1] = mmerge / test_time;
	_mm_free(dataIn);
	_mm_free(dataOut);
}

//if use same buffer store partial sorted data and run multi-way merge, then
//multi-way merge may overwrite data that is not merged yet, result to a wrong
//data list.
//TODO:does use task generation process can improve performance?
void multiWayMergeCPU(DoubleBuffer<float> &data, size_t *upperBound,
					  size_t chunkNum, hybridDispatchParams3<float> params)
{
	size_t cpumulti = params.cpuPart + params.multiWayUpdate;
	size_t blockNum = cpumulti / params.cpuBlockLen;
	//std::cout << cpumulti << " " << blockNum << std::endl;
    size_t *loopUBound = new size_t[chunkNum * params.threads];
    size_t *loopLBound = new size_t[chunkNum * params.threads];
	DoubleBuffer<size_t> bound(loopLBound, loopUBound);
	size_t *quantileSet = new size_t[chunkNum * (blockNum + 1)];
	//TODO: initial of first array of quantile must all move into quantile
	//compute functions.
	quantileSet[0] = 0;
	std::copy(upperBound, upperBound + chunkNum - 1, quantileSet + 1);
	quantileSetCompute(data, quantileSet, bound, upperBound, chunkNum,
					   params.cpuBlockLen, blockNum);
	float *mwBuffer = (float*)_mm_malloc(params.cpuChunkLen * sizeof(float), 16);
	float **start = new float*[chunkNum * params.threads];
	float **end   = new float*[chunkNum * params.threads];
	/*for(size_t i = 0; i < (blockNum + 1); ++i)
	  {
	  size_t index = i * chunkNum;
	  bool outofb = false;
	  for(size_t j = 0; j < chunkNum; ++j)
	  {
	  if(quantileSet[index + j] > (params.cpuPart + params.gpuPart))
	  {
	  outofb = true;
	  break;
	  }
	  }
	  if(outofb)
	  {
	  for(size_t j = 0; j < chunkNum; ++j)
	  std::cout << quantileSet[index + j] << " ";
	  }
	  }*/
	//std::cout << "quantile set compute complete.\n";
	//synchronize problem is the reason that parallel for loop cannot be
	//used. otherwise multi-thread may sort data in same position. this
	//version use static temp buffer for each thread to solve the problem,
	//which may not be best performance.
	//TODO: try circular buffer and/or parallel task to get the best
	//perfomance solution.
#pragma omp parallel for schedule(dynamic)
	for(size_t j = 0; j < blockNum; ++j)
	{
		int w = omp_get_thread_num();
		std::vector<float> unalignVec;
		DoubleBuffer<size_t> quantile(quantileSet + j * chunkNum,
									  quantileSet + j * chunkNum + chunkNum);
		//std::cout << j << std::endl;
		/*std::cout << quantile.buffers[0][0] << " " << quantile.buffers[0][1]
				  << " " << quantile.buffers[0][2] << std::endl
				  << quantile.buffers[1][0] << " " << quantile.buffers[1][1]
				  << " " << quantile.buffers[1][2] << std::endl;*/
		multiWayMergeBitonic(data, chunkNum, mwBuffer + w * params.cpuBlockLen,
							 j * params.cpuBlockLen, quantile, unalignVec,
							 start + w * chunkNum, end + w * chunkNum);
	}
	/*std::copy(quantileSet + chunkNum * params.threads,
	  quantileSet + chunkNum * (params.threads + 1), quantileSet);*/
	delete [] loopUBound;
	delete [] loopLBound;
	delete [] quantileSet;
	_mm_free(mwBuffer);
	delete [] start;
	delete [] end;
}

void mergeTest(size_t minLen, size_t maxLen, int seed)
{
	std::ofstream rFile("/home/aloneranger/source_code/Hybrid_Sort/result.txt",
						std::ios::app);
	if (rFile.is_open()) 
		rFile << "cpu merge test results\n"
			  << boost::format("%1%%|15t|") % "data length"
			  << boost::format("%1%%|15t|") % "chunk merge"
			  << boost::format("%1%%|15t|") % "multiway merge"
			  << std::endl;
	
	float *data = new float[maxLen];
	GenerateData(seed, data, maxLen);
	//Now, all length of data lists must be power of 2.
	for (size_t dataLen = minLen; dataLen <= maxLen; dataLen <<= 1)
	{
		std::cout << "data length: " << dataLen << std::endl;
		double results[2];
		hybrid_sort3(data, dataLen, results);
		/*rFile << boost::format("%1%%|15t|") % dataLen
			  << boost::format("%1%%|15t|") % results[0]
			  << boost::format("%1%%|15t|") % results[1]
			  << std::endl;*/
		std::cout << "merge test function result: " << results[0] << " "
				  << results[1] << std::endl;
	}
	delete [] data;
	/*rFile << std::endl << std::endl;
	  rFile.close();*/
}

void hybridMergeStep(DoubleBuffer<float> &data,
					 hybridDispatchParams3<float> &params)
{
	mergeStep1(data, 0, params);
	
#pragma omp parallel 
	{
		omp_set_nested(2);
#pragma omp sections
		{
#pragma omp section
			{
				if (params.gpuPart)
				{
					/*double x, y;
					  x = omp_get_wtime();*/
					int selector = 0;
					updateSelectorMultiWay(selector, 8, params.cpuChunkLen,
										   params.cpuPart);
					gpu_sort(data, params, 0, selector);
					/*y = omp_get_wtime();
					std::cout << "gpu in hybrid merge: " << (y - x)
					<< std::endl;*/
				}
			}
#pragma omp section
			{
				mergeStep2(data, 0, params);
				chunkMerge(data, params, params.cpuChunkLen);
				medianMerge(data, params);
				mergeStep3(data, params);
			}
		}
	}
	if(params.gpuPart) cudaDeviceSynchronize();
}

//TODO: cpu may not sort to one part, it may have several small parts.
//this can be decided by test GPU and CPU perfomance. how to guarantee
//portable?
//or is there a method to notify CPU, let it terminate sort work, though
//it may produce a more irregular upperbound, it does not matter to
//multiwaymerge.
void hybridMultiWayStep(DoubleBuffer<float> &data, size_t *upperBound,
						size_t chunkNum, hybridDispatchParams3<float> &params)
{
	//std::cout << "hybrid multiway step begin\n";
#pragma omp parallel 
	{
		omp_set_nested(1);
#pragma omp sections
		{
#pragma omp section
			{
				/*double x, y;
				  x = omp_get_wtime();*/
				multiWayMergeGPU(data, upperBound, chunkNum, params);
				/*y = omp_get_wtime();
				std::cout << "gpu in hybrid multiway: " << (y - x) << std::endl;*/
			}
#pragma omp section
			{
				/*double x, y;
				  x = omp_get_wtime();*/
				multiWayMergeCPU(data, upperBound, chunkNum, params);
				/*y = omp_get_wtime();
				  std::cout << "cpu in hybrid multiway: " << (y - x) << std::endl;*/
			}
		}
	}
	cudaDeviceSynchronize();
	data.selector ^= 1;
}

void hybrid_sort(float *data, size_t dataLen)
{
	float* dataIn = (float*)_mm_malloc(dataLen * sizeof(float), 16);
	float* dataOut= (float*)_mm_malloc(dataLen * sizeof(float), 16);
	std::copy(data, data + dataLen, dataIn);
	DoubleBuffer<float> hdata(dataIn, dataOut);
	hybridDispatchParams3<float> params(dataLen);
	hybridMergeStep(hdata, params);
	if (params.gpuPart)
	{
		int chunkNum = params.gpuPart / params.gpuChunkLen + 1;
		size_t *upperBound = new size_t[chunkNum];
		upperBound[0] = params.cpuPart;
		std::fill(upperBound + 1, upperBound + chunkNum, params.gpuChunkLen);
		std::partial_sum(upperBound, upperBound + chunkNum, upperBound);
		hybridMultiWayStep(hdata, upperBound, chunkNum, params);
		delete [] upperBound;
	} 
	resultTest(hdata.Current(), dataLen);
	_mm_free(dataIn);
	_mm_free(dataOut);
}

void hybrid_sort_test(size_t minLen, size_t maxLen, int seed)
{
	float *data = new float[maxLen];
	GenerateData(seed, data, maxLen);
	float* dataIn = (float*)_mm_malloc(maxLen * sizeof(float), 16);
	float* dataOut= (float*)_mm_malloc(maxLen * sizeof(float), 16);
	std::copy(data, data + maxLen, dataIn);
	DoubleBuffer<float> hdata(dataIn, dataOut);
	hybridDispatchParams3<float> params(maxLen);
	double x, y;
	x = omp_get_wtime();
	hybridMergeStep(hdata, params);
	y = omp_get_wtime();
	std::cout << "hybrid merge: " << (y - x) << std::endl;
	/*resultTest(hdata.Current(), params.cpuPart);
	resultTest(hdata.Current() + params.cpuPart, params.gpuChunkLen);
	resultTest(hdata.Current() + params.cpuPart + params.gpuChunkLen, params.gpuChunkLen);
	resultTest(hdata.buffers[hdata.selector ^ 1] + params.cpuPart, params.gpuChunkLen);
	resultTest(hdata.buffers[hdata.selector ^ 1] + params.cpuPart + params.gpuChunkLen, params.gpuChunkLen);*/
	if(params.gpuPart)
	{
		int chunkNum = params.gpuPart / params.gpuChunkLen + 1;
		std::cout << "chunk num: " << chunkNum << std::endl;
		size_t *upperBound = new size_t[chunkNum];
		upperBound[0] = params.cpuPart;
		std::fill(upperBound + 1, upperBound + chunkNum, params.gpuChunkLen);
		std::partial_sum(upperBound, upperBound + chunkNum, upperBound);
		x = omp_get_wtime();
		hybridMultiWayStep(hdata, upperBound, chunkNum, params);
		y = omp_get_wtime();
		std::cout << "hybrid multiway: " << (y - x) << std::endl;
		delete [] upperBound;
	}
	resultTest(hdata.Current(), maxLen);
	std::cout << "cpu multiway result: ";
	resultTest(hdata.Current(), params.cpuPart + params.multiWayUpdate);
	
	std::ofstream rFile("/home/aloneranger/source_code/Hybrid_Sort/result.txt",
						std::ios::app);
	if (rFile.is_open()) 
		rFile << "hybrid sort results: " << std::endl
			  << boost::format("%1%%|15t|") % "data length"
			  << boost::format("%1%%|15t|") % "merge step"
			  << boost::format("%1%%|15t|") % "multiway step"
			  << std::endl;
	
	for(size_t dataLen = minLen; dataLen <= maxLen; dataLen <<= 1)
	{
		int test_time = 50;
		double hmerge = 0.0, hmulti = 0.0;
		hybridDispatchParams3<float> pm(dataLen);
		int cnum = pm.gpuPart ? (pm.gpuPart / pm.gpuChunkLen + 1) : 1;
		size_t *uBound = new size_t[cnum];
		uBound[0] = pm.cpuPart;
		std::fill(uBound + 1, uBound + cnum, pm.gpuChunkLen);
		std::partial_sum(uBound, uBound + cnum, uBound);
		for(int i = 0; i < test_time; ++i)
		{
			std::copy(data, data + dataLen, dataIn);
			std::fill(dataOut, dataOut + dataLen, 0);
			hdata.selector = 0;
			double start, end;
			start = omp_get_wtime();
			hybridMergeStep(hdata, pm);
			end = omp_get_wtime();
			hmerge += (end - start);
			start = omp_get_wtime();
			if(pm.gpuPart) hybridMultiWayStep(hdata, uBound, cnum, pm);
			end = omp_get_wtime();
			hmulti += (end - start);
		}
		//cudaDeviceReset();
		rFile << boost::format("%1%%|15t|") % dataLen
			  << boost::format("%1%%|15t|") % (hmerge / test_time)
			  << boost::format("%1%%|15t|") % (hmulti / test_time)
			  << std::endl;
		delete [] uBound;
		//std::cout << "hmerge: " << hmerge << " hmulti: " << hmulti << std::endl;
		std::cout << dataLen << " test complete." << std::endl;
	}
	rFile << std::endl << std::endl;
	rFile.close();
	_mm_free(dataIn);
	_mm_free(dataOut);
	delete [] data;
}
