#include "sse_sort.h"

	// data length must be 32, this function will produce 8 sorted sequence of 
	// length 4

void simdOddEvenSort(__m128 *rData)
{   	 //	odd even sort lanes, then transpose them
	
    const int pairSize = rArrayLen >> 1;
    __m128 temp[pairSize];
    for (int i = 0; i < pairSize; ++i) temp[i] = rData[2 * i];
    for (int i = 0; i < rArrayLen; i += 2)
        rData[i] = _mm_min_ps(rData[i], rData[i + 1]);
    for (int i = 1; i < rArrayLen; i += 2)
        rData[i] = _mm_max_ps(rData[i], temp[i >> 1]);
	
    for (int i = 0; i < pairSize; i += 2)
    {
        temp[i] = rData[i * 2];
        temp[i + 1] = rData[i * 2 + 1];
    }
	
    for (int i = 0; i < rArrayLen; i += 4)
    {
        rData[i] = _mm_min_ps(rData[i], rData[i + 2]);
        rData[i + 1] = _mm_min_ps(rData[i + 1], rData[i + 3]);
    }
	
    for (int i = 2; i < rArrayLen; i += 4)
    {
        rData[i] = _mm_max_ps(rData[i], temp[(i >> 1) - 1]);
        rData[i + 1] = _mm_max_ps(rData[i + 1], temp[i >> 1]);
    }
	
    	// TODO:portability?
	
    for (int i = 0; i < rArrayLen >> 2; ++i) temp[i] = rData[i * 4 + 1];
    for (int i = 1; i < rArrayLen; i += 4)
        rData[i] = _mm_min_ps(rData[i], rData[i + 1]);
    for (int i = 2; i < rArrayLen; i += 4)
        rData[i] = _mm_max_ps(rData[i], temp[i >> 2]);
	
    	// temp,0,1,2,3
	
    for (int i = 0; i < rArrayLen; i += 2)
        temp[i >> 1] = _mm_shuffle_ps(rData[i], rData[i + 1], 0x44);
    //rdata,1,3,5,7
    for (int i = 1; i < rArrayLen; i += 2)
        rData[i] = _mm_shuffle_ps(rData[i], rData[i - 1], 0xee);
    //rdata,0,4
    for (int i = 0; i < pairSize; i += 2)
        rData[i * 2] = _mm_shuffle_ps(temp[i], temp[i + 1], 0x88);
    //rdata,2,6,depend,1,3,5,7
    for (int i = 2; i < rArrayLen; i += 4)
        rData[i] = _mm_shuffle_ps(rData[i - 1], rData[i + 1], 0x22);
    //rdata,3,7,depend,1,5
    for (int i = 3; i < rArrayLen; i += 4)
        rData[i] = _mm_shuffle_ps(rData[i - 2], rData[i], 0x77);
    //rdata,1,5
    for (int i = 0; i < pairSize; i += 2)
        rData[i * 2 + 1] = _mm_shuffle_ps(temp[i], temp[i + 1], 0xdd);
	
}

void sortInRegister(float *data)
	
{
    __m128 rData[rArrayLen];
    loadData(data, rData, rArrayLen);
	
	simdOddEvenSort(rData);
	
    bitonicSort428<4>(rData, true);
	
    bitonicSort8216<2>(rData, true);
	
    bitonicSort16232(rData);
	
    storeData(data, rData, rArrayLen);
}
