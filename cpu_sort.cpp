#include <iostream>
#include <algorithm>
#include <omp.h>

#include "cpu_sort.h"
#include "sse_sort.h"

void mergeInRegister(float *dataIn, float *dataOut, size_t dataLen)
{
    const size_t halfDataLen = dataLen >> 1;
    const int halfArrayLen = rArrayLen >> 1;
    __m128 *rData = new __m128[rArrayLen];
    float *ptrOut = dataOut;
    float *ptrLeftEnd = dataIn + halfDataLen, *ptrRightEnd = dataIn + dataLen;
    size_t lRemain = halfDataLen, rRemain = halfDataLen;
    loadData(ptrRightEnd - rRemain, rData + halfArrayLen, halfArrayLen);
    rRemain -= sortUnitLen;
    while (lRemain || rRemain)
    {
        bool useLeft;
        if (lRemain && rRemain)
            useLeft = *(ptrLeftEnd - lRemain) < *(ptrRightEnd - rRemain);
        else
            useLeft = lRemain > rRemain;
        if (useLeft)
        {
            loadData(ptrLeftEnd - lRemain, rData, halfArrayLen);
            lRemain -= sortUnitLen;
        }
        else
        {
            loadData(ptrRightEnd - rRemain, rData, halfArrayLen);
            rRemain -= sortUnitLen;
        }
		
        bitonicSort16232(rData);
        storeData(ptrOut, rData, halfArrayLen);
        ptrOut += sortUnitLen;
    }
    storeData(ptrOut, rData + halfArrayLen, halfArrayLen);
    delete [] rData;
}

//return value: flipped or not. If flipped, then the next sort must use
//dataOut as input.
bool registerSortIteration(float *dataIn, float *dataOut,
						   rsize_t minStride, rsize_t maxStride, rsize_t dataLen)
{
    rsize_t sortStride = minStride;
    float *sInput = dataIn, *sOutput = dataOut;
    while (sortStride <= maxStride)
    {
        for (rsize_t j = 0; j < dataLen; j += sortStride)
            mergeInRegister(sInput + j, sOutput + j, sortStride);
        float *temp = sInput;
        sInput = sOutput;
        sOutput = temp;
        sortStride *= 2;
    }
    return (sInput == dataOut);
}

void quantileInitialByPred(const rsize_t *quantileStart, rsize_t *quantileEnd,
						   const rsize_t *upperBound, rsize_t *loopUBound,
						   rsize_t sortedBlockNum, rsize_t mergeStride)
{
    for (rsize_t j = 0; j < sortedBlockNum; ++j)
        loopUBound[j] = std::min(quantileStart[j] + mergeStride, upperBound[j]);
    rsize_t average = mergeStride / sortedBlockNum;
    rsize_t n = mergeStride, row = 0;
    while (n)
    {
        if (row == sortedBlockNum)
            row = 0;
        rsize_t toBeAdd = std::min(std::max(average, (rsize_t)1), n);
        rsize_t canBeAdd = loopUBound[row] - quantileEnd[row];
        quantileEnd[row] += std::min(toBeAdd, canBeAdd);
        n -= std::min(toBeAdd, canBeAdd);
        ++row;
    }
}

void quantileCompute(float *data, const rsize_t *quantileStart,
					 rsize_t *quantileEnd, const rsize_t *upperBound,
					 rsize_t *loopUBound, rsize_t *loopLBound,
					 rsize_t sortedBlockNum, rsize_t mergeStride,
					 rsize_t quantileLen, bool initial = false)
{
    std::copy(quantileStart, quantileStart + sortedBlockNum, loopLBound);
    if (initial)
    {
        for (rsize_t j = 0; j < sortedBlockNum; ++j)
            loopUBound[j] = std::min(quantileStart[j] + quantileLen,
									 upperBound[j]);
        rsize_t average = quantileLen / sortedBlockNum;
        rsize_t residue = quantileLen % sortedBlockNum;
        for (rsize_t j = 0; j < sortedBlockNum; ++j)
            quantileEnd[j] = loopLBound[j] + average + (j < residue);
    }
    else
        quantileInitialByPred(quantileStart, quantileEnd, upperBound,
							  loopUBound, sortedBlockNum, mergeStride);
    while (true)
    {
        const float *lmax = NULL, *rmin = NULL;
        rsize_t lmaxRow, rminRow;
        for (rsize_t j = 0; j < sortedBlockNum; j++)
        {
            rsize_t testIndex = quantileEnd[j];
            if (testIndex > loopLBound[j] &&
                (!lmax || *lmax < data[testIndex - 1]))
            {
                lmax = data + testIndex - 1;
                lmaxRow = j;
            }
            if (testIndex < loopUBound[j] &&
                (!rmin || *rmin > data[testIndex]))
            {
                rmin = data + testIndex;
                rminRow = j;
            }
        }
        if (!lmax || !rmin || lmaxRow == rminRow || *lmax < *rmin ||
            (*lmax == *rmin && lmaxRow < rminRow))
            break;
        loopUBound[lmaxRow] = quantileEnd[lmaxRow] - 1;
        loopLBound[rminRow] = quantileEnd[rminRow] + 1;
        rsize_t deltaMax = (loopUBound[lmaxRow] - loopLBound[lmaxRow]) >> 1;
        rsize_t deltaMin = (loopUBound[rminRow] - loopLBound[rminRow]) >> 1;
        rsize_t delta = std::min(deltaMin, deltaMax);
        quantileEnd[lmaxRow] = loopUBound[lmaxRow] - delta;
        quantileEnd[rminRow] = loopLBound[rminRow] + delta;
    }
}

void multiWayMerge(float *dataIn, float *dataOut, rsize_t dataLen,
				   rsize_t sortedBlockLen, rsize_t mergeStride,
				   rsize_t startIndex, rsize_t endIndex)
{
    rsize_t sortedBlockNum = dataLen / sortedBlockLen;
    rsize_t *quantileStart = new rsize_t[sortedBlockNum];
    rsize_t *quantileEnd = new rsize_t[sortedBlockNum];
    rsize_t *upperBound = new rsize_t[sortedBlockNum];
    rsize_t *loopUBound = new rsize_t[sortedBlockNum];
    rsize_t *loopLBound = new rsize_t[sortedBlockNum];
    for (rsize_t j = 0; j < sortedBlockNum; ++j)
        quantileStart[j] = sortedBlockLen * j;
    for (rsize_t j = 0; j < sortedBlockNum; ++j)
        upperBound[j] = quantileStart[j] + sortedBlockLen;
    rsize_t i = startIndex;
    float *ptrOut = dataOut;
    if (startIndex)
    {
        ptrOut += i * mergeStride;
        quantileCompute(dataIn, quantileStart, quantileEnd, upperBound,
						loopUBound, loopLBound, sortedBlockNum, mergeStride,
						mergeStride * i, true);
        std::copy(quantileEnd, quantileEnd + sortedBlockNum, quantileStart);
    }
    else
    {
        std::copy(quantileStart, quantileStart + sortedBlockNum, quantileEnd);
    }
    for (; i < endIndex - 1; ++i)
    {
        quantileCompute(dataIn, quantileStart, quantileEnd, upperBound,
						loopUBound, loopLBound, sortedBlockNum, mergeStride, mergeStride);
        for (rsize_t j = 0; j < sortedBlockNum; ++j)
            for (rsize_t k = quantileStart[j]; k < quantileEnd[j]; ++k)
                *ptrOut++ = dataIn[k];
        std::copy(quantileEnd, quantileEnd + sortedBlockNum, quantileStart);
    }
    if (endIndex < dataLen / mergeStride)
    {
        quantileCompute(dataIn, quantileStart, quantileEnd, upperBound,
						loopUBound, loopLBound, sortedBlockNum, mergeStride, mergeStride);
        for (rsize_t j = 0; j < sortedBlockNum; j++)
            for (rsize_t k = quantileStart[j]; k < quantileEnd[j]; ++k)
                *ptrOut++ = dataIn[k];
        std::copy(quantileEnd, quantileEnd + sortedBlockNum, quantileStart);
    }
    else
        for (rsize_t j = 0; j < sortedBlockNum; j++)
            for (rsize_t k = quantileStart[j]; k < upperBound[j]; ++k)
                *ptrOut++ = dataIn[k];
    delete [] quantileStart;
    delete [] quantileEnd;
    delete [] upperBound;
    delete [] loopUBound;
    delete [] loopLBound;
}

//return value: flipped or not.
bool mergeSort(float *dataIn, float *dataOut, rsize_t dataLen)
{
    for (rsize_t offset = 0; offset < dataLen; offset += blockUnitLen)
    {
        sortInRegister(dataIn + offset);
    }
    return registerSortIteration(dataIn, dataOut, blockUnitLen * 2, dataLen,
								 dataLen);
}

/*float *mergeSortInBlockParallel(float *dataIn, float *dataOut,
  rsize_t dataLen)
  {
  const rsize_t blockSize = cacheSizeInByte() / (cacheFactor * sizeof(float));
  std::cout << "selected block size: " << blockSize << std::endl;
  bool flipped, flip2;
  if (blockSize)
  {
  int w;
  rsize_t blockNum = (dataLen / blockSize);

  //merge every block in cache, then merge blocks until the length of
  //sorted elements is chunkSize.
  #pragma omp parallel private(w)
  {
  int threads = omp_get_num_threads();
  w = omp_get_thread_num();
  rsize_t chunkSize = dataLen / threads;
  rsize_t chunkStart = w * chunkSize;
  for (rsize_t offset = chunkStart; offset < chunkStart + chunkSize;
  offset += blockSize)
  {
  flipped = mergeSort(dataIn + offset, dataOut + offset,
  blockSize);
  }
  if (flipped)
  flipped = registerSortIteration(dataOut + chunkStart,
  dataIn + chunkStart, blockSize * 2, chunkSize,
  chunkSize);
  else
  flipped = registerSortIteration(dataIn + chunkStart,
  dataOut + chunkStart, blockSize * 2, chunkSize,
  chunkSize);
  }

  //every thread will need compute quantile for every block in the
  //chunk assigned to it.
  #pragma omp parallel private(w)
  {
  int threads = omp_get_num_threads();
  rsize_t blocksPerThread = blockNum / threads;
  w = omp_get_thread_num();
  rsize_t quantileStart = w * blocksPerThread;
  rsize_t chunkSize = blockSize * blocksPerThread;
  rsize_t chunkStart = quantileStart * blockSize;
  if (flipped)
  {
  multiWayMerge(dataOut, dataIn, dataLen, chunkSize,
  blockSize, quantileStart,
  quantileStart + blocksPerThread);
  for (rsize_t offset = chunkStart;
  offset < chunkStart + chunkSize; offset += blockSize)
  flip2 = mergeSort(dataIn + offset, dataOut + offset,
  blockSize);
  }
  else
  {
  multiWayMerge(dataIn, dataOut, dataLen, chunkSize,
  blockSize, quantileStart,
  quantileStart + blocksPerThread);
  for (rsize_t offset = chunkStart;
  offset < chunkStart + chunkSize; offset += blockSize)
  flip2 = mergeSort(dataOut + offset, dataIn + offset,
  blockSize);
  }
  }
  }
  if(!(flip2 ^ flipped))
  return dataOut;
  return dataIn;
  }*/
