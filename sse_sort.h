#ifndef SSE_SORT_H
#define SSE_SORT_H

#include <smmintrin.h>
#include <xmmintrin.h>

const int simdLen = 4; //One Simd slot can hold 4 float number
const int rArrayLen = 8; //the length of simd register array in merge sort
//when use sort network, elements sorted in simd registers
//cannot more than 16 or 4 simd slots
const int logSortUnitLen = 4, sortUnitLen = 1 << logSortUnitLen;
const int logBlockUnitLen = 5, blockUnitLen = 1 << logBlockUnitLen;

inline void reverseData(__m128 *data, int n)
{
    for (int i = 0; i < n; i++)
    {
        data[i] = _mm_shuffle_ps(data[i], data[i], 0x1b);
    }
}

template<int pairSize>
void bitonicSort428(__m128 *data, bool reverse = false)
{
    __m128 temp[pairSize];
    const int simds = pairSize * 2;
    if (reverse)
        for (int i = 1; i < simds; i += 2)
            data[i] = _mm_shuffle_ps(data[i], data[i], 0x1b);
	
    for (int i = 0; i < simds; i += 2)
        temp[i >> 1] = _mm_max_ps(data[i], data[i+1]);
    for (int i = 0; i < simds; i += 2)
        data[i] = _mm_min_ps(data[i], data[i+1]);
    for (int i = 0; i < simds; i += 2)
        data[i+1] = _mm_shuffle_ps(data[i], temp[i>>1], 0xee);
    for (int i = 0; i < simds; i += 2)
        data[i] = _mm_shuffle_ps(data[i], temp[i>>1], 0x44);
	
    for (int i = 0; i < simds; i += 2)
        temp[i>>1] = _mm_min_ps(data[i], data[i+1]);
    for (int i = 1; i < simds; i += 2)
        data[i] = _mm_max_ps(data[i], data[i-1]);
    for (int i = 0; i < simds; i += 2)
        data[i] = _mm_blend_ps(temp[i>>1], data[i+1], 0x6);
    for (int i = 1; i < simds; i += 2)
        data[i] = _mm_blend_ps(data[i], temp[i>>1], 0x6);
    for (int i = 1; i < simds; i += 2)
        data[i] = _mm_shuffle_ps(data[i], data[i], 0xb1);
	
    for (int i = 0; i < simds; i += 2)
        temp[i>>1] = _mm_min_ps(data[i], data[i+1]);
    for (int i = 1; i < simds; i += 2)
        data[i] = _mm_max_ps(data[i], data[i-1]);
    for (int i = 0; i < simds; i += 2)
        data[i] = _mm_shuffle_ps(temp[i>>1], data[i+1], 0x44);
    for (int i = 0; i < simds; i += 2)
        data[i] = _mm_shuffle_ps(data[i], data[i], 0xd8);
    for (int i = 1; i < simds; i += 2)
        data[i] = _mm_shuffle_ps(data[i], temp[i>>1], 0xbb);
    for (int i = 1; i < simds; i += 2)
        data[i] = _mm_shuffle_ps(data[i], data[i], 0x72);
}

template<>
inline void bitonicSort428<1>(__m128 *data, bool reverse)
{
    __m128 temp;
    if (reverse) data[1] = _mm_shuffle_ps(data[1], data[1], 0x1b);
    temp = _mm_max_ps(data[0], data[1]);
    data[0] = _mm_min_ps(data[0], data[1]);
    data[1] = _mm_shuffle_ps(data[0], temp, 0xee);
    data[0] = _mm_shuffle_ps(data[0], temp, 0x44);
	
	
    temp = _mm_min_ps(data[0], data[1]);
    data[1] = _mm_max_ps(data[1], data[0]);
    data[0] = _mm_blend_ps(temp, data[1], 0x6);
    data[1] = _mm_blend_ps(data[1], temp, 0x6);
    data[1] = _mm_shuffle_ps(data[1], data[1], 0xb1);
	
    temp = _mm_min_ps(data[0], data[1]);
    data[1] = _mm_max_ps(data[1], data[0]);
    data[0] = _mm_shuffle_ps(temp, data[1], 0x44);
    data[0] = _mm_shuffle_ps(data[0], data[0], 0xd8);
    data[1] = _mm_shuffle_ps(data[1], temp, 0xbb);
    data[1] = _mm_shuffle_ps(data[1], data[1], 0x72);
}

template<int pairSize>
void bitonicSort8216(__m128 *data, bool reverse = false)
{
    const int stride = 4;
    const int size = pairSize * 2;
    __m128 temp[size];
    if (reverse)
        for (int i = 0; i < pairSize; ++i)
            reverseData(data + stride * i + 2, 2);
    for (int i = 0; i < size; i += 2) //布尔量直接和整型量加减，分别代表0和1.
    {
        temp[i] = data[i * 2 + 2 + reverse];
        temp[i + 1] = data[i * 2 + 3 - reverse];
    }
    for (int i = 0; i < pairSize; ++i)
    {
        __m128 *ptr = data + i * stride, *ptrTemp = temp + 2 * i;
        *(ptr + 3) = _mm_max_ps(*ptr, *ptrTemp);
        *(ptr + 2) = _mm_max_ps(*(ptr + 1), *(ptrTemp + 1));
    }
    for (int i = 0; i < pairSize; ++i)
    {
        __m128 *ptr = data + i * stride, *ptrTemp = temp + 2 * i;
        *ptr = _mm_min_ps(*ptr, *ptrTemp);
        *(ptr + 1) = _mm_min_ps(*(ptr + 1), *(ptrTemp + 1));
    }
    bitonicSort428<pairSize * 2>(data);
}

template<>
inline void bitonicSort8216<1>(__m128 *data, bool reverse)
{
    __m128 temp[2];
    //TODO: data dependence and performance?
    if (reverse)
    {
        reverseData(data + 2, 2);
    }
	
    temp[0] = data[2 + reverse];
    temp[1] = data[3 - reverse];
    data[3] = _mm_max_ps(data[0], temp[0]);
    data[2] = _mm_max_ps(data[1], temp[1]);
    data[0] = _mm_min_ps(data[0], temp[0]);
    data[1] = _mm_min_ps(data[1], temp[1]);
	
    bitonicSort428<2>(data);
}


//TODO: recursive use template
inline void bitonicSort16232(__m128 *data)
{
    const int stride = 4, indexLen = 2 * stride - 1;
    __m128 temp[stride];
    reverseData(data + stride, stride);
    for (int i = 0; i < stride; i++) temp[i] = data[i];
    for (int i = 0; i < stride; i++)
        data[i] = _mm_min_ps(temp[i], data[indexLen - i]);
    for (int i = 0; i < stride; i++)
        data[indexLen - i] = _mm_max_ps(data[indexLen - i], temp[i]);
    bitonicSort8216<2>(data);
}

//this function is also necessary, because in some cases the pointer cannot be
//changed.
inline void loadData(float *dataIn, __m128 *registers, int len)
{
    float *ptr = dataIn;
    for (int i = 0; i < len; i++)
    {
        registers[i] = _mm_load_ps(ptr);
        ptr += simdLen;
    }
}

inline void loadData(float **dataIn, __m128 *rData, int simdLanes)
{
	for (int i = 0; i < simdLanes; ++i)
	{
		rData[i] = _mm_load_ps(*dataIn);
		(*dataIn) += simdLen;
	}
}

inline void storeData(float *dataOut, __m128 *registers, int len)
{
    float *ptr = dataOut;
    for (int i = 0; i < len; i++)
    {
        //TODO: compare the performance difference
        _mm_store_ps(ptr, registers[i]);
        //_mm_stream_ps(ptr, registers[i]);
        ptr += simdLen;
    }
}

inline void storeData(float **dataOut, __m128 *rData, int simdLanes)
{
	for (int i = 0; i < simdLanes; ++i)
	{
		_mm_store_ps(*dataOut, rData[i]);
		(*dataOut) += simdLen;
	}
}

inline void streamData(float **dataOut, __m128 *rData, int simdLanes)
{
	for (int i = 0; i < simdLanes; ++i)
	{
		_mm_stream_ps(*dataOut, rData[i]);
		(*dataOut) += simdLen;
	}
}

void simdOddEvenSort(__m128 *rData);
void sortInRegister(float *data);
#endif //SSE_SORT_H
