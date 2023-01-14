#pragma once
#include "orb_structures.h"
#include "cuda_utils.h"



namespace orb
{
#define MAX_OCTAVE		5
#define FAST_PATTERN	16
#define HARRIS_SIZE		7
#define MAX_PATCH		31


	/* Set the maximum number of keypoints. */
	void setMaxNumPoints(const int num);

	/* Set scale factor for harris score. */
	void setScaleSqSq();

	/* Get the address of point counter */
	void getPointCounter(void** addr);

	/* Make offsets for FAST keypoints detection */
	void makeOffsets(int3* owhps, int noctaves);

	/* Compute FAST threshold LUT */
	void setFastThresholdLUT(int fast_threshold);

	/* Compute umax for angle computation */
	void setUmax(const int patch_size);

	/* Set pattern for feature computation */
	void setPattern(const int patch_size, const int wta_k);

	/* Set Gaussain kernel. Size 7, sigma 2 */
	void setGaussianKernel();

	/* Set the hamming distance LUT */
	//void setHammingTable();


	/* Create pyramid */
	__global__ void doubleImage(unsigned char* src, unsigned char* dst, int3 swhp, int3 dwhp);
	__global__ void scaleDown(unsigned char* src, unsigned char* dst, int3 swhp, int3 dwhp, int factor);
	void cuCreatePyramid(unsigned char* src, unsigned char* dst, int3 iwhp, int3* owhps, int* osizes, int* moffsets, int noctaves, bool doubled);
	void cuCreatePyramidAsync(unsigned char* src, unsigned char* dst, int3 iwhp, int3* owhps, int* osizes, int* moffsets, int noctaves, bool doubled);

	/* Find extreme by FAST */
	__device__ float fastScore(const unsigned char* ptr, const int* pixel, int threshold);
	__device__ float harrisScore(const unsigned char* ptr, const int pitch);
	__global__ void fastDetect(unsigned char* image, OrbPoint* points, int3 iwhp, int3 owhp, int threshold, int border, int octave);
	__global__ void fastDetectWithNMS(unsigned char* image, float* vmap, int3 owhp, int threshold, int octave, bool harris_score);
	__global__ void nms(float* vmap, int3 vwhp, OrbPoint* points, int border, int octave);
	__global__ void bitonicSort(OrbPoint* points, int p, int n, bool descending);
	void cuFastDectect(unsigned char* image, unsigned char* octave_images, OrbData& result, int3 iwhp, int3* owhps, 
		int* osizes, int* moffsets, int noctaves, bool doubled, int threshold, int border);
	void cuFastDectectWithNMS(unsigned char* image, unsigned char* octave_images, float* octave_vmaps, OrbData& result, int3 iwhp, int3* owhps,
		int* osizes, int* moffsets, int noctaves, bool doubled, int threshold, int border, bool harris_score);
	void cuFastDectectAsync(unsigned char* image, unsigned char* octave_images, OrbData& result, int3 iwhp, int3* owhps,
		int* osizes, int* moffsets, int noctaves, bool doubled, int threshold, int border);
	void cuFastDectectWithNMSAsync(unsigned char* image, unsigned char* octave_images, float* octave_vmaps, OrbData& result, int3 iwhp, int3* owhps,
		int* osizes, int* moffsets, int noctaves, bool doubled, int threshold, int border, bool harris_score);
	void cuRetainTopN(OrbData& result, const int n);

	/* Compute orientation */
	__global__ void angleIC(unsigned char* image, OrbPoint* points, int3 iwhp, int half_k, int npts);
	__global__ void angleIC(unsigned char* images, OrbPoint* points, int half_k, int npts);
	void cuComputeAngle(unsigned char* image, OrbData& result, int3 iwhp, int patch_size);
	void cuComputeAngle(unsigned char* octave_images, OrbData& result, int3* owhps, int* moffsets, int noctaves, int patch_size);

	/* Gassian blurring */
	__global__ void gaussFilter(unsigned char* src, unsigned char* dst, int3 whp);
	void cuGassianBlur(unsigned char* src, unsigned char* dst, int3 whp);
	void cuGassianBlur(unsigned char* octave_images, int3* owhps, int* moffsets, int noctaves);

	/* Compute descriptors */
	__inline__ __device__ unsigned char getValue(const unsigned char* center, const int2* pattern, float sine, float cose, int pitch, int idx);
	__device__ unsigned char feature2(const unsigned char* center, const int2* pattern, float sine, float cose, int pitch);
	__device__ unsigned char feature3(const unsigned char* center, const int2* pattern, float sine, float cose, int pitch);
	__device__ unsigned char feature4(const unsigned char* center, const int2* pattern, float sine, float cose, int pitch);
	__global__ void describle(unsigned char* image, OrbPoint* points, int3 iwhp, int wta_k, int npts);
	__global__ void describle(unsigned char* images, OrbPoint* points, int wta_k, int npts);
	void cuDescribe(unsigned char* image, OrbData& result, int3 iwhp, int wta_k);
	void cuDescribe(unsigned char* octave_images, OrbData& result, int wta_k);

	/* Rescale coordinates */
	__global__ void rescale(OrbPoint* points, int npts);
	void cuRescale(OrbData& result, int* scales, int noctaves);

	/* Match */
	inline __device__ int hammingDistance2(unsigned char* f1, unsigned char* f2);
	__global__ void hammingMatch(OrbPoint* points1, OrbPoint* points2, int n1, int n2);
	void cuHammingMatch(OrbData& result1, OrbData& result2);

}
