/*
 * cpu_sort.h
 *
 *  Created on: 2017-04-13
 *      Author: starkantoden
 */

#ifndef CPU_SORT_H_
#define CPU_SORT_H_

#ifndef _Windows
typedef size_t rsize_t;
#endif

#include <vector>

template <typename T>
struct DoubleBuffer
{
    T *buffers[2];
	
    int selector;
	
    inline DoubleBuffer()
    {
        selector = 0;
        buffers[0] = NULL;
        buffers[1] = NULL;
    }
	
    inline DoubleBuffer(T *current, T *alternate)
    {
        selector = 0;
        buffers[0] = current;
        buffers[1] = alternate;
    }
	
    inline T* Current()
    {
        return buffers[selector];
    }
};

void updateMergeSelector(int &selector, rsize_t dataLen);
void updateSelectorGeneral(int &selector, size_t startLen, size_t dataLen);
void updateSelectorMultiWay(int &selector, size_t startLen, size_t chunkLen,
							size_t dataLen);
size_t lastPower2(size_t a);
void mergeSort(DoubleBuffer<float> &data, rsize_t dataLen);
void mergeSortGeneral(DoubleBuffer<float> &data, rsize_t dataLen);
void singleThreadMerge(DoubleBuffer<float> &data, size_t dataLen);
void multiThreadMerge(DoubleBuffer<float> &data, size_t dataLen, int chunkNum,
					  size_t blockLen);
void registerSortIteration(DoubleBuffer<float> &data, rsize_t minStride,
						   rsize_t dataLen);
void multiWayMerge(DoubleBuffer<float> &data, rsize_t dataLen,
				   rsize_t sortedBlockLen, rsize_t mergeStride,
				   rsize_t startIndex, rsize_t endIndex);
void multiWayMergeGeneral(DoubleBuffer<float> &data, size_t dataLen,
						  size_t sortedChunkLen, size_t mergeStride,
						  size_t startOffset, size_t endOffset);
void multiWayMergeHybrid(DoubleBuffer<float> &data, size_t dataLen,
						 size_t *upperBound, size_t chunkNum, size_t mergeStride,
						 size_t startOffset, size_t endOffset);
void multiWayMergeMedian(DoubleBuffer<float> &data, size_t dataLen,
						 size_t *upperBound, size_t chunkNum,
						 size_t mergeStride, size_t startOffset,
						 size_t endOffset);
void multiWayMergeMedian(DoubleBuffer<float> &data, size_t dataLen,
						 size_t *upperBound, size_t chunkNum, float *tempBuffer,
						 size_t mergeStride, size_t startOffset,
						 size_t uaArrayLen);
void multiWayMergeMedianParallel(DoubleBuffer<float> &data, size_t dataLen,
								 size_t blockLen, size_t chunkLen);
void quantileSetCompute(DoubleBuffer<float> &data, size_t *quantileSet,
						DoubleBuffer<size_t> &bound, const size_t *upperBound,
						size_t chunkNum, size_t mergeStride, int setLen);
void multiWayMergeBitonic(DoubleBuffer<float> &data, size_t chunkNum,
								 float *tempBuffer, size_t startOffset,
								 DoubleBuffer<size_t> &quantile,
								 std::vector<float> &unalignVec, float **start,
								 float **end);
//Default argument for a given parameter has to be specified no more than once.
//Specifying it more than once (even with the same default value) is illegal.
void quantileCompute(float *data, DoubleBuffer<rsize_t> &quantile,
					 DoubleBuffer<rsize_t> &bound, const rsize_t *upperBound,
					 rsize_t chunkNum, rsize_t quantileLen,
					 bool initial = false);

#endif /* CPU_SORT_H_ */
