#ifndef _NEON_H
#define _NEON_H
typedef unsigned char uchar;
#include <immintrin.h>

class NEON
{

private:
	int width;
	int height;

public:
	NEON(int width, int height);
	~NEON();
	void yuyvToRgbaHostSIMD(uchar *yuyv, uchar *rgba);
};

#endif