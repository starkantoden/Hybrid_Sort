/* sort functions using double buffer class */

#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <omp.h>
#include <xmmintrin.h>
#include <immintrin.h>
#include "cpu_sort.h"
#include "sse_sort.h"


int getMergeStartBuffer(size_t chunkNum, int endBuffer);

	// insert sort, and search process use binary search.

void insertBinarySort(float *data, size_t dataLen)
{
	for (size_t j = 1; j < dataLen; ++j)
	{
		if (data[j] > data[j - 1])
		{
			continue;
		}
		else
		{
			size_t low = 0, high = j - 1, medium;
			while (low != high)
			{
				medium = (low + high) / 2;
				if (data[medium] > data[j])
					--high;
				else
					++low;
			}
			if (data[j] > data[low])
				++low;
			float temp = data[j];
			//In std::copy function, pointers or iterators cannot
			//overlap, in this case must use copy_backward.
			//there is no exception be showed, but result may not
			//be correct.
			std::copy_backward(data + low, data + j, data + j + 1);
			data[low] = temp;
		}
	}
	
}

	// TODO:modify to a version that compliant with STL style. replace above pointer
	// funtion.

void insertBinarySortVec(std::vector<float> &data)
{
	std::vector<float>::iterator it = data.begin();
	for (size_t j = 1; j < data.size(); ++j)
	{
		if (data[j] > data[j - 1])
		{
			continue;
		}
		else
		{
			size_t low = 0, high = j - 1, medium;
			while (low != high)
			{
				medium = (low + high) / 2;
				if (data[medium] > data[j])
					--high;
				else
					++low;
			}
			if (data[j] > data[low])
				++low;
			float temp = data[j];
			
				// In std::copy function, pointers or iterators cannot
				// overlap, in this case must use copy_backward.
				// there is no exception be showed, but result may not
				// be correct.
			
			std::copy_backward(it + low, it + j, it + j + 1);
			data[low] = temp;
		}
	}
	
}

void mergeInRegister(DoubleBuffer<float> &data, size_t dataLen)
{
	const size_t halfDataLen = dataLen >> 1;
	const int halfArrayLen = rArrayLen >> 1;
	__m128 rData[rArrayLen];
	float *ptrOut = data.buffers[data.selector ^ 1];
	float *ptrIn[2], *end[2];
	end[0] = data.Current() + halfDataLen;
	end[1] = data.Current() + dataLen;
	ptrIn[0] = data.Current();
	ptrIn[1] = end[0];
	loadData(ptrIn[1], rData + halfArrayLen, halfArrayLen);
	ptrIn[1] += sortUnitLen;
	while ((ptrIn[0] < end[0]) && (ptrIn[1] < end[1]))
	{
		int index = ((*ptrIn[0]) >= (*ptrIn[1]));
		loadData(ptrIn[index], rData, halfArrayLen);
		ptrIn[index] += sortUnitLen;
		
		_mm_prefetch(ptrIn[0], _MM_HINT_T0);
		_mm_prefetch(ptrIn[1], _MM_HINT_T0);
		bitonicSort16232(rData);
		storeData(ptrOut, rData, halfArrayLen);
		ptrOut += sortUnitLen;
	}
	int index = (ptrIn[0] == end[0]);
	for (; ptrIn[index] < end[index]; ptrIn[index] += sortUnitLen)
	{
		loadData(ptrIn[index], rData, halfArrayLen);
		
		_mm_prefetch(ptrIn[index] + sortUnitLen, _MM_HINT_T0);
		bitonicSort16232(rData);
		storeData(ptrOut, rData, halfArrayLen);
		ptrOut += sortUnitLen;
	}
	storeData(ptrOut, rData + halfArrayLen, halfArrayLen);
}

	// blockNum must be power of 2 and cannot great than 8.
	// len indicate how many simd register lanes will be use per block.

void inline loadSimdDataInitial(__m128 *rData, float **blocks, int blockNum,
								int len)
{
	int offset = len;
	for (int i = 1; i < blockNum; i += 2)
	{
		loadData(blocks + i, rData + offset, len);
		offset += 2 * len;
	}
}

void inline loadSimdData(__m128 *rData, float **blocks, float **blockBound,
						 int blockNum, int len)
{
	int second, offset = 0;
	for (int i = 0; i < blockNum; i += 2)
	{
		if (blocks[i] != blockBound[i] && blocks[i + 1] != blockBound[i + 1])
			second = (*(blocks[i]) >= *(blocks[i + 1]));
		else
			second = (blocks[i] == blockBound[i]);
		loadData(blocks + i + second, rData + offset, len);
		offset += 2 * len;
	}
}

	// TODO: recall the means of blockNum, may be there are several blocks of output?

void inline storeSimdData(__m128 *rData, float **output, int blockNum, int len,
						  int start)
{
	int offset = start;
	for (int i = 0; i < blockNum; i += 2)
	{
		storeData(output + (i >> 1), rData + offset, len);
		offset += 2 * len;
	}
}

	// data length and block length both must be power of 2.

void mergeInRegister(DoubleBuffer<float> &data, size_t dataLen, size_t blockLen)
{
	size_t blockNum = dataLen / blockLen;
	size_t bLen = blockLen;
	__m128 rData[rArrayLen];
	for (; blockNum >= rArrayLen; blockNum >>= 1, bLen <<= 1)
	{
		float *blocks[rArrayLen];
		float *blockBound[rArrayLen];
		float *output[rArrayLen >> 1];
		for (size_t i = 0; i < blockNum; i += rArrayLen)
		{
			size_t startIndex = i * bLen;
			blocks[0] = data.buffers[data.selector] + startIndex;
			//pointers cannot be added.
			for (int j = 1; j < rArrayLen; ++j) blocks[j] = blocks[j-1] + bLen;
			std::copy(blocks + 1, blocks + rArrayLen, blockBound);
			blockBound[rArrayLen - 1] = blocks[rArrayLen - 1] + bLen;
			output[0] = data.buffers[data.selector ^ 1] + startIndex;
			for (int j = 1; j < (rArrayLen >> 1); ++j)
				output[j] = output[j - 1] + (bLen << 1);
			
			//1 is actually rArrayLen / rArrayLen
			loadSimdDataInitial(rData, blocks, rArrayLen, 1);
			//simdLen actually is rArrayLen * simdLen / rArrayLen
			size_t loop = bLen * 2 / simdLen - 1;
			for (size_t i = 0; i < loop; ++i)
			{
				loadSimdData(rData, blocks, blockBound, rArrayLen, 1);
				bitonicSort428<4>(rData, true);
				storeSimdData(rData, output, rArrayLen, 1, 0);
			}
			storeSimdData(rData, output, rArrayLen, 1, 1);
		}
		data.selector ^= 1;
	}
	for (; blockNum > 1; blockNum >>= 1, bLen <<= 1)
	{
		float **blocks = new float *[blockNum];
		float **blockBound = new float *[blockNum];
		float **output = new float *[blockNum >> 1];
		blocks[0] = data.buffers[data.selector];
		for (int i = 1; i < blockNum; ++i) blocks[i] = blocks[i - 1] + bLen;
		std::copy(blocks + 1, blocks + blockNum, blockBound);
		blockBound[blockNum - 1] = blocks[blockNum - 1] + bLen;
		output[0] = data.buffers[data.selector ^ 1];
		for (int i = 1; i < (blockNum >> 1); ++i)
			output[i] = output[i - 1] + (bLen << 1);
		
		int len = rArrayLen / blockNum;
		loadSimdDataInitial(rData, blocks, blockNum, len);
		size_t loop = bLen * 2 / (len * simdLen) - 1;
		for (size_t i = 0; i < loop; ++i)
		{
			loadSimdData(rData, blocks, blockBound, blockNum, len);
			if (blockNum == 4) bitonicSort8216<2>(rData, true);
			if (blockNum == 2) bitonicSort16232(rData);
			storeSimdData(rData, output, blockNum, len, 0);
		}
		storeSimdData(rData, output, blockNum, len, len);
		delete [] blocks;
		delete [] blockBound;
		delete [] output;
		data.selector ^= 1;
	}
}

bool inline multipleOf(size_t offset, int factor)
{
	return ((offset & ~(factor - 1)) == offset);
}

void inline swapFloat(float &a, float &b)
{
	float temp = a;
	a = b;
	b = temp;
}

	// TODO: modify to general mode, namely two array pointer, not only one data.

int inline loadUnalignData(float *data, size_t &offsetA, size_t &offsetB,
						   float *unalignData, int factor, bool start)
{
	size_t begin[2], end[2];
	size_t factornot = ~(factor - 1);
	if (start)
	{
		begin[0] = offsetA;
		end[0] = (offsetA + factor) & factornot;
		begin[1] = offsetB;
		end[1] = (offsetB + factor) & factornot;
		offsetA = end[0];
		offsetB = end[1];
	}
	else
	{
		begin[0] = offsetA & factornot;
		end[0] = offsetA;
		begin[1] = offsetB & factornot;
		end[1] = offsetB;
		offsetA = begin[0];
		offsetB = begin[1];
	}
	int n = 0, selector = (data[begin[0]] > data[begin[1]]);
	//remember that the value of unalignData cannot be changed here.
	for (int i = begin[selector]; i < end[selector]; ++i)
		unalignData[n++] = data[i];
	for (int i = begin[selector ^ 1]; i < end[selector ^ 1]; ++i)
		unalignData[n++] = data[i];
	int lenA = end[selector] - begin[selector];
	if (lenA % simdLen)
	{
		int laneIndex = lenA / simdLen;
		float *ptr = unalignData + laneIndex * simdLen;
		if (ptr[0] > ptr[1]) swapFloat(ptr[0], ptr[1]);
		if (ptr[2] > ptr[3]) swapFloat(ptr[2], ptr[3]);
		if (ptr[0] > ptr[2]) swapFloat(ptr[0], ptr[2]);
		if (ptr[1] > ptr[3]) swapFloat(ptr[1], ptr[3]);
		if (ptr[1] > ptr[2]) swapFloat(ptr[1], ptr[2]);
	}
	return selector;
}

	// If the trail of two lists is unalign because of median computation,
	// we need know where to "insert" the unalign data, to sort them correctly.
	// comparation is only done in one list, because the uValue comes from it and
	// must larger than all the previous keys in same list. offset is the trail of
	// the other list, the return value is number of elements that must be loaded
	// after unaligned data loaded.

size_t getTrailPosition(float *data, size_t bOffset, size_t eOffset, float uValue,
						int unitLen)
{
	size_t n = 0;
	size_t i = eOffset - unitLen;
	while (i >= bOffset) {
		if (data[i] > uValue)
		{
			n += unitLen;
			i -= unitLen;
		}
		else
			break;
	}
	return n;
}

	// only used by 16 to 32 merge loop, namely only two lists merge to one list.
	// must be used as a mediate process, rData must be copied a half, and must be
	// emplified after this process complete.

inline void simdMergeLoop2(__m128 *rData, float **dataOut, float **blocks,
						   float **blockBound, int lanes, int unitLen)
{
	//std::cout << "simd merge loop2 begin... " << blockBound[0] - blocks[0] << " " << blockBound[1] - blocks[1] << " " << unitLen << " " << lanes << std::endl;
	if (blocks[0] != blockBound[0] && blocks[1] != blockBound[1])
	{
		int selector = (*(blockBound[0] - unitLen) > *(blockBound[1] - unitLen));
		//std::cout << *(blockBound[0] - unitLen) << " " << *(blockBound[1] - unitLen) << " " << selector << std::endl;
		int xors = selector ^ 1;
		//selected list have higher priority to load data to simd registers.
		int zeros = 0;
		while (blocks[selector] != blockBound[selector])
		{
			int temps = (*blocks[selector] > *blocks[xors]) ? xors : selector;
			/*if(blocks[0] == (blockBound[0] - unitLen))
			  std::cout << *blocks[0] << " " << *blocks[1] << std::endl;*/
			if(temps == 0) ++zeros;
			loadData(&blocks[temps], rData, lanes);
			bitonicSort16232(rData);
			storeData(dataOut, rData, lanes);
		}
		//std::cout << "first block is selected " << zeros << std::endl;
	}
	//std::cout << "one list is out..\n";
	//std::cout << blockBound[0] - blocks[0] << " " << blockBound[1] - blocks[1] << std::endl;
	int selector = (blocks[0] == blockBound[0]);
	while (blocks[selector] != blockBound[selector])
	{
		loadData(&blocks[selector], rData, lanes);
		bitonicSort16232(rData);
		storeData(dataOut, rData, lanes);
	}
}

	// merge two lists into one list. the two list both reside in dataIn, the
	// elements that will be merged is bounded by offset arrays.
	// TODO: rename to simdMergeUnalign

void simdMergeGeneral(float *dataIn, float *dataOut, size_t offsetA[2],
					  size_t offsetB[2]) 
{
	int unitLen = (rArrayLen >> 1) * simdLen;
	int halfArrayLen = rArrayLen >> 1;
	__m128 rData[rArrayLen];
	float **output = new float*;
	*output = dataOut;
	if (!multipleOf(offsetA[0], unitLen))
	{
		float *unalignStart = (float*)_mm_malloc(unitLen * sizeof(float), 16);
		loadUnalignData(dataIn, offsetA[0], offsetB[0], unalignStart, unitLen,
						true);
		loadData(unalignStart, rData + halfArrayLen, halfArrayLen);
		bitonicSort428<2>(rData + halfArrayLen, true);
		bitonicSort8216<1>(rData + halfArrayLen, true);
		_mm_free(unalignStart);
	}
	else
	{
		loadData(dataIn + offsetB[0], rData + halfArrayLen, halfArrayLen);
		offsetB[0] += unitLen;
	}
	int tKeys = 0, selector = 0; 
	float *unalignEnd = NULL;
	if (!multipleOf(offsetA[1], unitLen))
	{
		unalignEnd = (float*)_mm_malloc(unitLen * sizeof(float), 16);
		selector = loadUnalignData(dataIn, offsetA[1], offsetB[1],
								   unalignEnd, unitLen, false);
		if (selector)
			tKeys = getTrailPosition(dataIn, offsetA[0], offsetA[1],
									 unalignEnd[0], unitLen);
		else
			tKeys = getTrailPosition(dataIn, offsetB[0], offsetB[1],
									 unalignEnd[0], unitLen);
	}
	float *blocks[2], *blockBound[2];
	blocks[0] = dataIn + offsetA[0], blocks[1] = dataIn + offsetB[0];
	blockBound[0] = dataIn + offsetA[1], blockBound[1] = dataIn + offsetB[1];
	//std::cout << selector << std::endl;
	blockBound[selector ^ 1] -= tKeys;
	simdMergeLoop2(rData, output, blocks, blockBound, halfArrayLen, unitLen);
	blockBound[selector ^ 1] += tKeys;
	if (unalignEnd != NULL)
	{
		loadData(unalignEnd, rData, halfArrayLen);
		bitonicSort428<2>(rData, true);
		bitonicSort8216<1>(rData, true);
		bitonicSort16232(rData);
		storeData(output, rData, halfArrayLen);
		_mm_free(unalignEnd);
		simdMergeLoop2(rData, output, blocks, blockBound, halfArrayLen, unitLen);
	}
	storeData(output, rData + halfArrayLen, halfArrayLen);
	delete output;
}

void simdMergeAlign(float *dataOut, float **start, float **end)
{
	int unitLen = (rArrayLen >> 1) * simdLen;
	int halfArrayLen = rArrayLen >> 1;
	__m128 rData[rArrayLen];
	float **output = new float*;
	*output = dataOut;
	loadData(&start[1], rData + halfArrayLen, halfArrayLen);
	simdMergeLoop2(rData, output, start, end, halfArrayLen, unitLen);
	storeData(output, rData + halfArrayLen, halfArrayLen);
	delete output;
}

void mergeInRegisterUnalignBuffer(DoubleBuffer<float> &data,
								  DoubleBuffer<float> &buffer,
								  size_t *offsetA, size_t *offsetB,
								  size_t outputOffset)
{
	simdMergeGeneral(data.buffers[data.selector],
					 buffer.buffers[buffer.selector] + outputOffset,
					 offsetA, offsetB);
}

void mergeInRegisterUnalign(DoubleBuffer<float> &data, size_t offsetA[2],
							size_t offsetB[2], size_t outputOffset)
{
	simdMergeGeneral(data.buffers[data.selector],
					 data.buffers[data.selector ^ 1] + outputOffset,
					 offsetA, offsetB);
}

void mergeInRegisterUnalign(DoubleBuffer<float> &data, size_t offsetA[2],
							size_t offsetB[2], float *outputOffset)
{
	int unitLen = (rArrayLen >> 1) * simdLen;
	int halfArrayLen = rArrayLen >> 1;
	__m128 rData[rArrayLen];
	float **output = new float*;
	*output = outputOffset;
	if (!multipleOf(offsetA[0], unitLen))
	{
		float *unalignStart = new float[unitLen];
		loadUnalignData(data.Current(), offsetA[0], offsetB[0], unalignStart,
						unitLen, true);
		loadData(unalignStart, rData + halfArrayLen, halfArrayLen);
		bitonicSort428<2>(rData + halfArrayLen, true);
		bitonicSort8216<1>(rData + halfArrayLen, true);
		delete [] unalignStart;
	}
	else
	{
		loadData(data.Current() + offsetB[0], rData + halfArrayLen,
				 halfArrayLen);
		offsetB[0] += unitLen;
	}
	size_t loop = (offsetA[1] - offsetA[0] + offsetB[1] - offsetB[0]) / unitLen;
	float *unalignEnd = NULL;
	if (!multipleOf(offsetA[1], unitLen))
	{
		unalignEnd = new float[unitLen];
		loadUnalignData(data.Current(), offsetA[1], offsetB[1], unalignEnd,
						unitLen, false);
		--loop;
	}
	float *blocks[2], *blockBound[2];
	blocks[0] = data.Current() + offsetA[0];
	blocks[1] = data.Current() + offsetB[0];
	blockBound[0] = data.Current() + offsetA[1];
	blockBound[1] = data.Current() + offsetB[1];
	for (size_t i = 0; i < loop; ++i)
	{
		loadSimdData(rData, blocks, blockBound, 2, halfArrayLen);
		bitonicSort16232(rData);
		//storeSimdData(rData, output, 2, halfArrayLen, 0);
		storeData(output, rData, halfArrayLen);
	}
	if (unalignEnd != NULL)
	{
		loadData(unalignEnd, rData, halfArrayLen);
		bitonicSort428<2>(rData, true);
		bitonicSort8216<1>(rData, true);
		bitonicSort16232(rData);
		//storeSimdData(rData, output, 2, halfArrayLen, 0);
		storeData(output, rData, rArrayLen);
		delete [] unalignEnd;
	}
	else
		storeData(output, rData + halfArrayLen, halfArrayLen);
	//storeSimdData(rData, output, 2, halfArrayLen, halfArrayLen);
	delete output;
}

void registerSortIteration(DoubleBuffer<float> &data, rsize_t minStride,
						   rsize_t dataLen)
{
	for (size_t i = minStride; i <= dataLen; i *= 2)
	{
		for (rsize_t j = 0; j < dataLen; j += i)
		{
			DoubleBuffer<float> chunk(data.buffers[data.selector] + j,
									  data.buffers[data.selector ^ 1] + j);
			mergeInRegister(chunk, i);
		}
		data.selector ^= 1;
	}
}

void quantileInitial(DoubleBuffer<rsize_t> &quantile, const rsize_t *upperBound,
					 DoubleBuffer<rsize_t> bound, rsize_t chunkNum,
					 rsize_t quantileLen, bool initial)
{
	for (rsize_t j = 0; j < chunkNum; ++j)
		bound.buffers[1][j] =
			std::min(quantile.buffers[0][j] + quantileLen, upperBound[j]);
	size_t n = quantileLen;
	if (initial)
	{
		std::copy(bound.buffers[0], bound.buffers[0] + chunkNum,
				  quantile.buffers[1]);
	}
	int *remain = new int[chunkNum];
	std::fill(remain, remain + chunkNum, 1);
	do
	{
		size_t average = n / std::accumulate(remain, remain + chunkNum, 0);
		for (size_t i = 0; i < chunkNum; ++i)
		{
			if (remain[i])
			{
				size_t toBeAdd = std::min(std::max(average, size_t(1)), n);
				size_t canBeAdd = bound.buffers[1][i] - quantile.buffers[1][i];
				if (toBeAdd < canBeAdd)
				{
					quantile.buffers[1][i] += toBeAdd;
					n -= toBeAdd;
				}
				else
				{
					quantile.buffers[1][i] = bound.buffers[1][i];
					n -= canBeAdd;
					remain[i] = 0;
				}
			}
		}
	}while(n);
	delete [] remain;
}

 // TODO: think of the relation of quantile initial and quantile compute, to
 // simplify the program.

void quantileCompute(float *data, DoubleBuffer<rsize_t> &quantile,
					 DoubleBuffer<rsize_t> &bound, const rsize_t *upperBound,
					 rsize_t chunkNum, rsize_t quantileLen,
					 bool initial)
{
	std::copy(quantile.buffers[0], quantile.buffers[0] + chunkNum,
			  bound.buffers[0]);
	quantileInitial(quantile, upperBound, bound, chunkNum,
					quantileLen, initial);
	while (true)
	{
		const float *lmax = NULL, *rmin = NULL;
		rsize_t lmaxRow, rminRow;
		for (rsize_t j = 0; j < chunkNum; ++j)
		{
			rsize_t testIndex = quantile.buffers[1][j];
			if (testIndex > bound.buffers[0][j] &&
				(!lmax || *lmax < data[testIndex - 1]))
			{
				lmax = data + testIndex - 1;
				lmaxRow = j;
			}
			if (testIndex < bound.buffers[1][j] &&
				(!rmin || *rmin > data[testIndex]))
			{
				rmin = data + testIndex;
				rminRow = j;
			}
		}
		if (!lmax || !rmin || lmaxRow == rminRow || *lmax < *rmin ||
			(*lmax == *rmin && lmaxRow < rminRow))
            break;
		bound.buffers[1][lmaxRow] = quantile.buffers[1][lmaxRow] - 1;
		bound.buffers[0][rminRow] = quantile.buffers[1][rminRow] + 1;
		rsize_t deltaMin = (bound.buffers[1][rminRow] -
							bound.buffers[0][rminRow]) >> 1;
		rsize_t deltaMax =
			(bound.buffers[1][lmaxRow] - bound.buffers[0][lmaxRow]) >> 1;
		rsize_t delta = std::min(deltaMin, deltaMax);
		quantile.buffers[1][lmaxRow] =
			bound.buffers[1][lmaxRow] - delta;
		quantile.buffers[1][rminRow] =
			bound.buffers[0][rminRow] + delta;
	}
}

	// until now, the elements of first chunk in quantileset is correctly
	// initialized, no seperate quantile compute needed.

void quantileSetCompute(DoubleBuffer<float> &data, size_t *quantileSet,
						DoubleBuffer<size_t> &bound, const size_t *upperBound,
						size_t chunkNum, size_t mergeStride, int setLen)
{
	/*#pragma omp parallel
	  {
	  #pragma omp single
	  {
	  int bulk = 16, n = 0;
	  while(n < setLen)
	  {
	  int temp = std::min(bulk, setLen - n);
	  #pragma omp task
	  {
	  int w = omp_get_thread_num();
	  DoubleBuffer<size_t> bnd(bound.buffers[0] + w * chunkNum,
	  bound.buffers[1] + w * chunkNum);
	  if(n)
	  {
	  DoubleBuffer<size_t> qtl(quantileSet,
	  quantileSet + n * chunkNum);
	  quantileCompute(data.Current(), qtl, bnd, upperBound,
	  chunkNum, n * mergeStride, true);
	  }
	  int x = temp - (n < (setLen - temp));
	  for (int i = 0, j = n * chunkNum; i < x; ++i, j += chunkNum)
	  {
	  DoubleBuffer<size_t> quantile(quantileSet + j,
	  quantileSet + j+chunkNum);
	  //TODO:may move these initialization into quantile
	  //compute function?
	  std::copy(quantile.buffers[0],
	  quantile.buffers[0] + chunkNum,
	  quantile.buffers[1]);
	  quantileCompute(data.Current(), quantile, bnd,
	  upperBound, chunkNum, mergeStride);
	  }
	  }
	  n += temp;
	  }
	  }
	  }*/
	
		// int bulk = 16;
	
	int bulk = setLen / 16;
	if(setLen < 160) bulk = setLen / 8;
#pragma omp parallel for schedule(dynamic)
	for(int i = 0; i < setLen; i += bulk)
	{
		int w = omp_get_thread_num();
		DoubleBuffer<size_t> bnd(bound.buffers[0] + w * chunkNum,
								 bound.buffers[1] + w * chunkNum);
		if(i)
		{
			DoubleBuffer<size_t> qtl(quantileSet, quantileSet + i * chunkNum);
			quantileCompute(data.Current(), qtl, bnd, upperBound, chunkNum,
							i * mergeStride, true);
		}
		int x = bulk - (i < (setLen - bulk));
		for(int j = 0, k = i * chunkNum; j < x; ++j, k += chunkNum)
		{
			DoubleBuffer<size_t> quantile(quantileSet + k,
										  quantileSet + k + chunkNum);
			std::copy(quantile.buffers[0], quantile.buffers[0] + chunkNum,
					  quantile.buffers[1]);
			quantileCompute(data.Current(), quantile, bnd, upperBound, chunkNum,
							mergeStride);
		}
	}
}

void moveBaseQuantile(DoubleBuffer<float> &data, DoubleBuffer<rsize_t> &quantile,
					  DoubleBuffer<rsize_t> bound, const rsize_t *upperBound,
					  rsize_t chunkNum, rsize_t mergeStride, float **ptrOut)
{
	quantileCompute(data.Current(), quantile, bound, upperBound, chunkNum,
					mergeStride);
	for (rsize_t j = 0; j < chunkNum; ++j)
	{
		std::copy(data.buffers[data.selector] + quantile.buffers[0][j],
				  data.buffers[data.selector] + quantile.buffers[1][j],
				  *ptrOut);
		(*ptrOut) += (quantile.buffers[1][j] - quantile.buffers[0][j]);
	}
	std::copy(quantile.buffers[1], quantile.buffers[1] + chunkNum,
			  quantile.buffers[0]);
}

void multiWayMerge(DoubleBuffer<float> &data, rsize_t dataLen,
				   rsize_t sortedBlockLen, rsize_t mergeStride,
				   rsize_t startIndex, rsize_t endIndex)
{
	rsize_t sortedBlockNum = dataLen / sortedBlockLen;
    rsize_t *quantileStart = new rsize_t[sortedBlockNum];
    rsize_t *quantileEnd = new rsize_t[sortedBlockNum];
    rsize_t *upperBound = new rsize_t[sortedBlockNum];
    rsize_t *loopUBound = new rsize_t[sortedBlockNum];
    rsize_t *loopLBound = new rsize_t[sortedBlockNum];
	DoubleBuffer<rsize_t> quantile(quantileStart, quantileEnd);
	DoubleBuffer<rsize_t> bound(loopLBound, loopUBound);
	for (rsize_t j = 0; j < sortedBlockNum; ++j)
		quantile.buffers[0][j] = sortedBlockLen * j;
	for (rsize_t j = 0; j < sortedBlockNum; ++j)
		upperBound[j] = quantile.buffers[0][j] + sortedBlockLen;
	rsize_t i = startIndex;
	float *ptrOut = data.buffers[data.selector ^ 1];
	if (startIndex)
	{
		ptrOut += i * mergeStride;
		quantileCompute(data.buffers[data.selector], quantile, bound, upperBound,
						sortedBlockNum, mergeStride * i, true);
		std::copy(quantile.buffers[1], quantile.buffers[1] + sortedBlockNum,
				  quantile.buffers[0]);
	}
	else
	{
		std::copy(quantile.buffers[0], quantile.buffers[0] + sortedBlockNum,
				  quantile.buffers[1]);
	}
	for (; i < endIndex - 1; ++i)
		moveBaseQuantile(data, quantile, bound, upperBound, sortedBlockNum,
						 mergeStride, &ptrOut);
	if (endIndex < dataLen / mergeStride)
		moveBaseQuantile(data, quantile, bound, upperBound, sortedBlockNum,
						 mergeStride, &ptrOut);
	else
		for (rsize_t j = 0; j < sortedBlockNum; ++j)
			for (rsize_t k = quantile.buffers[0][j]; k < upperBound[j];
				 ++k)
				*ptrOut++ = data.buffers[data.selector][k];
	delete [] quantileStart;
    delete [] quantileEnd;
    delete [] upperBound;
    delete [] loopUBound;
    delete [] loopLBound;
}

void multiWayMergeGeneral(DoubleBuffer<float> &data, size_t dataLen,
						  size_t sortedChunkLen, size_t mergeStride,
						  size_t startOffset, size_t endOffset)
{
	size_t sortedChunkNum = dataLen / sortedChunkLen;
	size_t *quantileStart = new size_t[sortedChunkNum];
	size_t *quantileEnd = new size_t[sortedChunkNum];
	size_t *upperBound = new size_t[sortedChunkNum];
	size_t *loopUBound = new size_t[sortedChunkNum];
	size_t *loopLBound = new size_t[sortedChunkNum];
	DoubleBuffer<size_t> quantile(quantileStart, quantileEnd);
	DoubleBuffer<size_t> bound(loopLBound, loopUBound);
	for (size_t j = 0; j < sortedChunkNum; ++j)
		quantile.buffers[0][j] = sortedChunkLen * j;
	for (size_t j = 0; j < sortedChunkNum; ++j)
		upperBound[j] = quantile.buffers[0][j] + sortedChunkLen;
	float *ptrOut = data.buffers[data.selector ^ 1] + startOffset;
	if (startOffset)
	{
		quantileCompute(data.buffers[data.selector], quantile, bound, upperBound,
						sortedChunkNum, startOffset, true);
		std::copy(quantile.buffers[1], quantile.buffers[1] + sortedChunkNum,
				  quantile.buffers[0]);
	}
	else
	{
		std::copy(quantile.buffers[0], quantile.buffers[0] + sortedChunkNum,
				  quantile.buffers[1]);
	}
	for (size_t offset = startOffset; offset < endOffset - mergeStride;
		 offset += mergeStride)
	{
		moveBaseQuantile(data, quantile, bound, upperBound, sortedChunkNum,
						 mergeStride, &ptrOut);
	}
	if (endOffset < dataLen)
		moveBaseQuantile(data, quantile, bound, upperBound, sortedChunkNum,
						 mergeStride, &ptrOut);
	else
		for (size_t j = 0; j < sortedChunkNum; ++j)
			for (size_t k = quantile.buffers[0][j]; k < upperBound[j]; ++k)
				*ptrOut++ = data.buffers[data.selector][k];
	delete [] quantileStart;
	delete [] quantileEnd;
	delete [] upperBound;
	delete [] loopUBound;
	delete [] loopLBound;
}

void multiWayMergeHybrid(DoubleBuffer<float> &data, size_t dataLen,
						 size_t *upperBound, size_t chunkNum, size_t mergeStride,
						 size_t startOffset, size_t endOffset)
{
	size_t *quantileStart = new size_t[chunkNum];
	size_t *quantileEnd = new size_t[chunkNum];
	size_t *loopUBound = new size_t[chunkNum];
	size_t *loopLBound = new size_t[chunkNum];
	DoubleBuffer<size_t> quantile(quantileStart, quantileEnd);
	DoubleBuffer<size_t> bound(loopLBound, loopUBound);
	quantile.buffers[0][0] = 0;
	std::copy(upperBound, upperBound + chunkNum - 1, quantile.buffers[0] + 1);
	float *ptrOut = data.buffers[data.selector ^ 1] + startOffset;
	if (startOffset)
	{
		quantileCompute(data.buffers[data.selector], quantile, bound, upperBound,
						chunkNum, startOffset, true);
		std::copy(quantile.buffers[1], quantile.buffers[1] + chunkNum,
				  quantile.buffers[0]);
	}
	else
	{
		std::copy(quantile.buffers[0], quantile.buffers[0] + chunkNum,
				  quantile.buffers[1]);
	}
	size_t end = std::min(endOffset, dataLen - mergeStride);
	for (size_t offset = startOffset; offset < end; offset += mergeStride)
	{
		moveBaseQuantile(data, quantile, bound, upperBound, chunkNum,
						 mergeStride, &ptrOut);
	}
	if (endOffset == dataLen)
		for (size_t j = 0; j < chunkNum; ++j)
		{
			std::copy(data.buffers[data.selector] + quantile.buffers[0][j],
					  data.buffers[data.selector] + upperBound[j], ptrOut);
			ptrOut += (upperBound[j] - quantile.buffers[0][j]);
		}
	delete [] quantileStart;
	delete [] quantileEnd;
	delete [] loopUBound;
	delete [] loopLBound;
}

void multiWayMergeMedian(DoubleBuffer<float> &data, size_t dataLen,
						 size_t *upperBound, size_t chunkNum,
						 size_t mergeStride, size_t startOffset,
						 size_t endOffset)
{
	int unitLen = (rArrayLen >> 1) * simdLen;
	size_t *quantileStart = new size_t[chunkNum];
	size_t *quantileEnd = new size_t[chunkNum];
	DoubleBuffer<size_t> quantile(quantileStart, quantileEnd);
	size_t *loopUBound = new size_t[chunkNum];
	size_t *loopLBound = new size_t[chunkNum];
	DoubleBuffer<size_t> bound(loopLBound, loopUBound);
	quantile.buffers[0][0] = 0;
	std::copy(upperBound, upperBound + chunkNum - 1, quantile.buffers[0] + 1);
	if (startOffset)
	{
		quantileCompute(data.Current(), quantile, bound, upperBound, chunkNum,
						startOffset, true);
		std::copy(quantile.buffers[1], quantile.buffers[1] + chunkNum,
				  quantile.buffers[0]);
	}
	else
	{
		std::copy(quantile.buffers[0], quantile.buffers[0] + chunkNum,
				  quantile.buffers[1]);
	}
	float *singleBuffer = (float*)_mm_malloc(mergeStride * sizeof(float), 16);
	for (size_t i = startOffset, m = 0; i < endOffset; i += mergeStride, ++m)
	{
		quantileCompute(data.Current(), quantile, bound, upperBound, chunkNum,
						mergeStride);
		std::vector<float> unalignVec;
		float **start = new float*[chunkNum];
		float **end = new float*[chunkNum];
		int factornot = ~(unitLen - 1);
		for (size_t j = 0; j < chunkNum; ++j)
		{
			size_t listLen = quantile.buffers[1][j] - quantile.buffers[0][j];
			size_t uLen = listLen & factornot;
			size_t aLen = listLen - uLen;
			//Proved that insert new elements in a vector the end of its
			//iterator can be used.
			if (aLen)
				unalignVec.insert(unalignVec.end(),
								  data.Current()+quantile.buffers[0][j]+uLen,
								  data.Current()+quantile.buffers[1][j]);
			start[j] = data.Current() + quantile.buffers[0][j];
			end[j] = data.Current() + quantile.buffers[0][j] + uLen;
		}
		std::copy(quantile.buffers[1], quantile.buffers[1] + chunkNum,
				  quantile.buffers[0]);
		if (!unalignVec.empty())
			std::sort(unalignVec.begin(), unalignVec.end());
		DoubleBuffer<float> block(singleBuffer,
								  data.buffers[data.selector ^ 1] + i);
		int selector = getMergeStartBuffer(chunkNum, 1);
		float *ptrIn = block.buffers[selector];
		for (size_t j = 0; j < chunkNum; j += 2)
		{
			for (int k = 0; k < 2; ++k)
			{
				float *temp = ptrIn;
				std::copy(start[j + k], end[j + k], ptrIn);
				ptrIn += (end[j + k] - start[j + k]);
				if (!unalignVec.empty() && (unalignVec[0] >= *(ptrIn - 1)))
				{
					std::copy(unalignVec.begin(), unalignVec.end(), ptrIn);
					ptrIn += unalignVec.size();
					unalignVec.clear();
				}
				start[j + k] = temp;
				end[j + k] = ptrIn;
			}
			size_t startOffset = start[j] - block.buffers[selector];
			size_t endOffset = end[j + 1] - block.buffers[selector];
			simdMergeAlign(block.buffers[selector ^ 1] + startOffset,
						   &start[j], &end[j]);
			start[j / 2] = block.buffers[selector ^ 1] + startOffset;
			end[j / 2] = block.buffers[selector ^ 1] + endOffset;
		}
		for (size_t j = chunkNum >> 1; j > 1; j >>= 1)
		{
			selector ^= 1;
			for (size_t k = 0; k < j; k += 2)
			{
				size_t startOffset = start[k] - block.buffers[selector];
				size_t endOffset = end[k + 1] - block.buffers[selector];
				simdMergeAlign(block.buffers[selector ^ 1] + startOffset,
							   &start[k], &end[k]);
				start[k / 2] = block.buffers[selector ^ 1] + startOffset;
				end[k / 2] = block.buffers[selector ^ 1] + endOffset;
			}
		}
		delete [] start;
		delete [] end;
	}
	data.selector ^= 1;
	_mm_free(singleBuffer);
	delete [] quantileStart;
	delete [] quantileEnd;
	delete [] loopLBound;
	delete [] loopUBound;
}

inline void multiWayMergeMedian(DoubleBuffer<float> &data, size_t dataLen,
								size_t *upperBound, size_t chunkNum,
								float *tempBuffer,
								DoubleBuffer<size_t> &quantile,
								DoubleBuffer<size_t> &bound, size_t mergeStride,
								size_t startOffset,
								std::vector<float> &unalignVec, int factornot,
								float **start, float **end)
{
	if (startOffset)
	{
		quantileCompute(data.Current(), quantile, bound, upperBound, chunkNum,
						startOffset, true);
		std::copy(quantile.buffers[1], quantile.buffers[1] + chunkNum,
				  quantile.buffers[0]);
	}
	else
	{
		std::copy(quantile.buffers[0], quantile.buffers[0] + chunkNum,
				  quantile.buffers[1]);
	}
	quantileCompute(data.Current(), quantile, bound, upperBound, chunkNum,
					mergeStride);
	for (size_t j = 0; j < chunkNum; ++j)
	{
		size_t listLen = quantile.buffers[1][j] - quantile.buffers[0][j];
		size_t uLen = listLen & factornot;
		size_t aLen = listLen - uLen;
		if (aLen)
			unalignVec.insert(unalignVec.end(),
							  data.Current() + quantile.buffers[0][j] + uLen,
							  data.Current() + quantile.buffers[1][j]);
		start[j] = data.Current() + quantile.buffers[0][j];
		end[j] = data.Current() + quantile.buffers[0][j] + uLen;
	}
	if (!unalignVec.empty()) insertBinarySortVec(unalignVec);
	//if (!unalignVec.empty()) std::sort(unalignVec.begin(), unalignVec.end());
	DoubleBuffer<float> block(tempBuffer,
							  data.buffers[data.selector ^ 1] + startOffset);
	int selector = getMergeStartBuffer(chunkNum, 1);
	float *ptrIn = block.buffers[selector];
	for (size_t j = 0; j < chunkNum; j += 2)
	{
		for (int k = 0; k < 2; ++k)
		{
			float *temp = ptrIn;
			std::copy(start[j + k], end[j + k], ptrIn);
			ptrIn += (end[j + k] - start[j + k]);
			if (!unalignVec.empty() && (unalignVec[0] >= *(ptrIn - 1)))
			{
				std::copy(unalignVec.begin(), unalignVec.end(), ptrIn);
				ptrIn += unalignVec.size();
				unalignVec.clear();
			}
			start[j + k] = temp;
			end[j + k] = ptrIn;
		}
		size_t startOffset = start[j] - block.buffers[selector];
		size_t endOffset = end[j + 1] - block.buffers[selector];
		simdMergeAlign(block.buffers[selector ^ 1] + startOffset,
					   &start[j], &end[j]);
		start[j / 2] = block.buffers[selector ^ 1] + startOffset;
		end[j / 2] = block.buffers[selector ^ 1] + endOffset;
	}
	for (size_t j = chunkNum >> 1; j > 1; j >>= 1)
	{
		selector ^= 1;
		for (size_t k = 0; k < j; k += 2)
		{
			size_t startOffset = start[k] - block.buffers[selector];
			size_t endOffset = end[k + 1] - block.buffers[selector];
			simdMergeAlign(block.buffers[selector ^ 1] + startOffset,
						   &start[k], &end[k]);
			start[k / 2] = block.buffers[selector ^ 1] + startOffset;
			end[k / 2] = block.buffers[selector ^ 1] + endOffset;
		}
	}
}

void inline unalignMove(std::vector<float> &unalignVec, float *&output,
						float **start, float **end, size_t index)
{
	float *temp = output;
	output = std::copy(start[index], end[index], output);
	if(!unalignVec.empty() && (unalignVec[0] >= output[-1]))
	{
		output = std::copy(unalignVec.begin(), unalignVec.end(), output);
		unalignVec.clear();
	}
	start[index] = temp;
	end[index] = output;
}

void inline simdMergeUnit(DoubleBuffer<float> &block, float **start, float **end,
						  size_t sIdx, size_t &tIdx, int selector)
{
	size_t startOffset = start[sIdx] - block.buffers[selector];
	size_t endOffset = end[sIdx + 1] - block.buffers[selector];
	//std::cout << "offsets: " << startOffset << " " << endOffset << std::endl;
	simdMergeAlign(block.buffers[selector ^ 1] + startOffset,
				   &start[sIdx], &end[sIdx]);
	start[tIdx] = block.buffers[selector ^ 1] + startOffset;
	end[tIdx] = block.buffers[selector ^ 1] + endOffset;
	++tIdx;
	//std::cout << "simd merge unit complete.\n";
}

void multiWayMergeBitonic(DoubleBuffer<float> &data, size_t chunkNum,
						  float *tempBuffer, size_t startOffset,
						  DoubleBuffer<size_t> &quantile,
						  std::vector<float> &unalignVec, float **start,
						  float **end)
{
	int unitLen = (rArrayLen >> 1) * simdLen;
	int factornot = ~(unitLen - 1);
	size_t cLen = 0;
	for (size_t j = 0; j < chunkNum; ++j)
	{
		size_t listLen = quantile.buffers[1][j] - quantile.buffers[0][j];
		if(listLen)
		{
			size_t uLen = listLen & factornot;
			if (uLen < listLen)
				unalignVec.insert(unalignVec.end(),
								  data.Current() + quantile.buffers[0][j] + uLen,
								  data.Current() + quantile.buffers[1][j]);
			if(uLen)
			{
				start[cLen] = data.Current() + quantile.buffers[0][j];
				end[cLen] = data.Current() + quantile.buffers[0][j] + uLen;
				//std::cout << end[cLen] - start[cLen] << " ";
				++cLen;
			}
		}
	}
	//std::cout << "\ncLen: " << cLen << std::endl;
	if (!unalignVec.empty()) insertBinarySortVec(unalignVec);
	if(cLen == 1)
	{
		float *ptrOut = data.buffers[data.selector ^ 1] + startOffset;
		ptrOut = std::copy(start[0], end[0], ptrOut);
		if(!unalignVec.empty())
		{
			std::copy(unalignVec.begin(), unalignVec.end(), ptrOut);
			unalignVec.clear();
		}
		//std::cout << "copied!\n";
		return;
	}
	DoubleBuffer<float> block(tempBuffer,
							  data.buffers[data.selector ^ 1] + startOffset);
	int selector = getMergeStartBuffer(chunkNum, 1);
	float *ptrIn = block.buffers[selector];
	size_t cIdx = 0;
	if(cLen & 1)
	{
		//In the next loop, other lists will be moved twice. once for load
		//data, once for sort and move. So data In the odd one must be
		//copied to correct address, equivalent to move twice.
		float *tempIn = block.buffers[selector ^ 1];
		unalignMove(unalignVec, tempIn, start, end, cIdx);
		++cIdx;
		ptrIn += (tempIn - block.buffers[selector ^ 1]);
	}
	for (size_t j = cIdx; j < cLen; j += 2)
	{
		for (int k = 0; k < 2; ++k)
			unalignMove(unalignVec, ptrIn, start, end, j + k);
		simdMergeUnit(block, start, end, j, cIdx, selector);
	}
	cLen = cIdx;
	cIdx = 0;
	while (cLen > 1)
	{
		selector ^= 1;
		if(cLen & 1)
		{
			float *output = block.buffers[selector ^ 1];
			unalignMove(unalignVec, output, start, end, cIdx);
			++cIdx;
		}
		for(size_t j = cIdx; j < cLen; j += 2)
			simdMergeUnit(block, start, end, j, cIdx, selector);
		cLen = cIdx;
		cIdx = 0;
	}
}

void multiWayMergeMedianParallel(DoubleBuffer<float> &data, size_t dataLen,
								 size_t blockLen, size_t chunkLen)
{
	size_t chunkNum = dataLen / chunkLen;
	size_t *upperBound = new size_t[chunkNum];
	std::fill(upperBound, upperBound + chunkNum, chunkLen);
	std::partial_sum(upperBound, upperBound + chunkNum, upperBound);
	size_t stride = omp_get_max_threads() * blockLen;
	float *mwBuffer =
		(float*)_mm_malloc(stride * omp_get_max_threads() *sizeof(float), 16);
	int unitLen = (rArrayLen >> 1) * simdLen;
	int factornot = ~(unitLen - 1);
#pragma omp parallel
	{
		size_t *quantileStart = new size_t[chunkNum];
		size_t *quantileEnd = new size_t[chunkNum];
		DoubleBuffer<size_t> quantile(quantileStart, quantileEnd);
		size_t *loopUBound = new size_t[chunkNum];
		size_t *loopLBound = new size_t[chunkNum];
		DoubleBuffer<size_t> bound(loopLBound, loopUBound);
		std::vector<float> unalignVec;
		size_t offset = omp_get_thread_num() * blockLen;
		float **start = new float*[chunkNum];
		float **end = new float*[chunkNum];
		for (size_t i = offset; i < dataLen; i += stride)
		{
			quantile.buffers[0][0] = 0;
			std::copy(upperBound, upperBound + chunkNum - 1,
					  quantile.buffers[0] + 1);
			multiWayMergeMedian(data, dataLen, upperBound, chunkNum,
								mwBuffer + offset, quantile, bound, blockLen,
								i, unalignVec, factornot, start, end);
		}
		delete [] start;
		delete [] end;
		delete [] quantileStart;
		delete [] quantileEnd;
		delete [] loopLBound;
		delete [] loopUBound;
	}
	data.selector ^= 1;
	delete [] upperBound;
	_mm_free(mwBuffer);
}

void mergeSort(DoubleBuffer<float> &data, rsize_t dataLen)
{
	for (rsize_t offset = 0; offset < dataLen; offset += blockUnitLen)
		sortInRegister(data.Current() + offset);
	registerSortIteration(data, blockUnitLen * 2, dataLen);
}

void singleThreadMerge(DoubleBuffer<float> &data, size_t dataLen)
{
	__m128 rData[rArrayLen];
	float *ptr = data.buffers[data.selector];
	size_t loop = dataLen >> logBlockUnitLen; 
	//cannot use pointer to pointer, because load data and store data coexist.
	for (size_t i = 0; i < loop; ++i)
	{
		loadData(ptr, rData, rArrayLen);
		simdOddEvenSort(rData);
		bitonicSort428<4>(rData, true);
		storeData(ptr, rData, rArrayLen);
		ptr += blockUnitLen;
	}
	mergeInRegister(data, dataLen, rArrayLen);
}

void updateMergeSelector(int &selector, rsize_t dataLen)
{
	rsize_t blocks = dataLen / blockUnitLen;
	if (_tzcnt_u64(blocks) & 1) selector ^= 1;
}

void updateSelectorGeneral(int &selector, size_t startLen, size_t dataLen)
{
	size_t blocks = dataLen / startLen;
	if (_tzcnt_u64(blocks) & 1) selector ^= 1;
}

//TODO:may be generalized. now div must be power of 2.
void updateSelectorMultiWay(int &selector, size_t startLen, size_t chunkLen,
							size_t dataLen)
{
	size_t div = dataLen / chunkLen;
	if(div <= 4)
	{
		updateSelectorGeneral(selector, startLen, dataLen);
	}
	else
	{
		updateSelectorGeneral(selector, startLen, chunkLen);
		//int multit = div / 16 + (div % 16 != 0);
		/*int multit = div, n = 0;
		do
		{
			if(multit <= 16) multit = 0;
			else multit /= 16;
			++n;
		} while (multit > 0);
		if(n & 1) selector ^= 1;*/
		size_t temp = _tzcnt_u64(div);
		size_t n = temp / 4 + (temp % 4 != 0);
	}
}

size_t lastPower2(size_t a)
{
	return 1 << (64 - _lzcnt_u64(a) - 1);
}

int getMergeStartBuffer(size_t chunkNum, int endBuffer)
{
	//this expression check whether chunknum is power of 2 or not,
	//if not, then we need get the minimum number which is power of 2 and
	// greater than it.
	if(chunkNum & (chunkNum - 1))
	{
		if((64 - _lzcnt_u64(chunkNum)) & 1)
			return endBuffer ^ 1;
		return endBuffer;
	}
	else
	{
		if (_tzcnt_u64(chunkNum) & 1)
			return endBuffer ^ 1;
		return endBuffer;
	}
}

void mergeSortGeneral(DoubleBuffer<float> &data, size_t dataLen)
{
	for (rsize_t offset = 0; offset < dataLen; offset += blockUnitLen)
		sortInRegister(data.Current() + offset);
	size_t sortedBlockLen = blockUnitLen;
	size_t sortedBlockNum = dataLen / blockUnitLen;
	do
	{
		size_t stride = std::min(sortedBlockNum, size_t(16));
		size_t strideLen = stride * sortedBlockLen;
		for (size_t j = 0; j < dataLen; j += strideLen)
		{
			multiWayMergeGeneral(data, dataLen, sortedBlockLen, blockUnitLen,
								 j, j + strideLen);
		}
		data.selector ^= 1;
		for (rsize_t offset = 0; offset < dataLen; offset += blockUnitLen)
			sortInRegister(data.Current() + offset);
		sortedBlockLen = strideLen;
		sortedBlockNum = dataLen / sortedBlockLen;
	}while(sortedBlockNum > 1);
}

//medianA and medianB are both relative offset.
void getMedian(float *data, size_t mergeLen, size_t chunkLen,
			   size_t baseOffset, size_t &medianA, size_t &medianB)
{
	size_t startA = medianA, startB = medianB;
	size_t minA, maxA;
	//if chunkA is "shorter" then everything is ok, but if chunkA is "longer",
	//then care must be taken to prevent medianB get a value that bigger
	//than the bound of chunkB.
	//the key is: is mergeLen - (chunkLen - medianB) positive or negative?
	maxA = std::min(medianA + mergeLen, chunkLen);
	minA = medianA + mergeLen - std::min(mergeLen, (chunkLen - medianB));
	float *blockA = data + baseOffset;
	float *blockB = data + baseOffset + chunkLen;
	while(minA + 1 != maxA)
	{
		size_t median = (minA + maxA) >> 1;
		if(blockA[median] <= blockB[mergeLen - median])
			minA = median;
		else
			maxA = median;
	}
	size_t resultA = (blockA[minA] <= blockB[mergeLen - maxA])? maxA : minA;
	medianA = resultA;
	medianB = mergeLen - resultA;
}

void multiThreadMergeGeneral(float *dataIn, float* dataOut, size_t dataLen,
							 int chunkNum, size_t blockLen)
{
	int blockNum = dataLen / blockLen;
	//std::cout << "blockNum: " << blockNum << std::endl;
	size_t chunkLen = dataLen / chunkNum;
	int pairNum = chunkNum >> 1;
	int medianNum =  blockNum - pairNum;
	int mdNumPerPair = medianNum / pairNum;
	int thdNumPerPair = blockNum / pairNum;
	size_t *medianA = new size_t[medianNum];
	size_t *medianB = new size_t[medianNum];
	//may read much fewer elements than sort, so can compute all medians.
	//#pragma omp parallel for schedule(dynamic, 8)
	for (int j = 0; j < medianNum; ++j)
	{
		//length can greater than the length of a chunk.
		size_t length = (j % mdNumPerPair + 1) * blockLen;
		int pairIndex = j / mdNumPerPair;
		size_t baseOffset = pairIndex * chunkLen * 2;
		size_t mA = 0, mB = 0;
		getMedian(dataIn, length, chunkLen, baseOffset, mA, mB);
		medianA[j] = baseOffset + mA;
		medianB[j] = baseOffset + chunkLen + mB;
	}
#pragma omp parallel for schedule(dynamic)
	for (int j = 0; j < blockNum; ++j)
	{
		size_t offsetA[2], offsetB[2];
		int pairIndex = j / thdNumPerPair;
		int offset = j % thdNumPerPair;
		int preThreadNum = pairIndex * mdNumPerPair;
		size_t baseOffset = pairIndex * chunkLen * 2;
		int baseIndex = preThreadNum + offset;
		if (offset)
		{
			offsetA[0] = medianA[baseIndex - 1];
			offsetB[0] = medianB[baseIndex - 1];
		}
		else
		{
			offsetA[0] = baseOffset;
			offsetB[0] = baseOffset + chunkLen;
		}
		if (offset == (thdNumPerPair - 1))
		{
			offsetA[1] = baseOffset + chunkLen;
			offsetB[1] = offsetA[1] + chunkLen;
		}
		else
		{
			offsetA[1] = medianA[baseIndex];
			offsetB[1] = medianB[baseIndex];
		}
		simdMergeGeneral(dataIn, dataOut + j * blockLen, offsetA, offsetB);
	}
	delete [] medianA;
	delete [] medianB;
}

//chunkNum should be exponent of 2, all chunks must be equal of length. 
void multiThreadMerge(DoubleBuffer<float> &data, size_t dataLen, int chunkNum,
					  size_t blockLen)
{
	for (int i = chunkNum; i > 1; i >>= 1)
	{
		multiThreadMergeGeneral(data.buffers[data.selector],
								data.buffers[data.selector ^ 1], dataLen, i,
								blockLen);
		data.selector ^= 1;
	}
}

 
