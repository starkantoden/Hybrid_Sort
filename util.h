#ifndef _Windows
#include <stdio.h>
#endif

#include <algorithm>
#include <boost/random/linear_congruential.hpp>
#include <boost/random/uniform_real.hpp>

/*#ifndef _Windows
typedef size_t rsize_t;
#endif*/

void GenerateData(int seed, float *data, size_t N)
{
    boost::minstd_rand generator(seed);
    boost::uniform_real<float> floatDist(1.0, 6.0);
    for (size_t i = 0; i < N; i++)
    {
        data[i] = floatDist(generator);
    }
}

template<typename T>bool resultTest(T *data, size_t N)
{
    for (size_t i = 0; i < N - 1; i++)
    {
        //if (data[i] > data[i + 1] || data[i] == 0)
		if (data[i] > data[i + 1])
        {
            std::cout << "unsorted at fucking " << i << "! " << std::endl;
			std::cout << "fail value: " << data[i] << " " << data[i + 1]
					  << std::endl;
            return false;
        }
    }
    std::cout << "successfully sorted!" << std::endl;
    return true;
}

template<typename T>void resultCheck(T* dataA, T* dataB, size_t N)
{
	std::sort(dataA, dataA + N);
	for (size_t i = 0; i < N; ++i)
	{
		if(dataA[i] != dataB[i])
			std::cout << "inconsistent at " << i << " " << dataA[i] << " "
					  << dataB[i] << std::endl;
	}
}

inline size_t cacheSizeInByte()
{
    size_t size = 0;
#ifndef _Windows
    FILE *fptr = 0;
    fptr = fopen("/sys/devices/system/cpu/cpu0/cache/index0/size", "r");
    if(fptr)
    {
        char size_type;
        if (fscanf(fptr, "%lu%c", &size, &size_type) != 2)
            size = 0;
        if(size_type == 'K') size *= 1024;
        fclose(fptr);
    }
#endif
    //std::cout << "cache size selected: " << size << std::endl;
    return size;
}

inline size_t cacheSizeInByte3()
{
    size_t size = 0;
#ifndef _Windows
    FILE *fptr = 0;
    fptr = fopen("/sys/devices/system/cpu/cpu0/cache/index3/size", "r");
    if(fptr)
    {
        char size_type;
        if (fscanf(fptr, "%lu%c", &size, &size_type) != 2)
            size = 0;
        if(size_type == 'K') size *= 1024;
        fclose(fptr);
    }
#endif
    //std::cout << "cache size selected: " << size << std::endl;
    return size;
}
