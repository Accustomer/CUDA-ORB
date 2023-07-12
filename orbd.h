#pragma once
#include "orb_structures.h"
#include "cuda_utils.h"


namespace orb
{

	/* Set the maximum number of keypoints. */
	void setMaxNumPoints(const int num);

	/* Get the address of point counter */
	void getPointCounter(void** addr);

	/* Compute FAST threshold LUT */
	void setFastThresholdLUT(int fast_threshold);

	/* Compute umax for angle computation */
	void setUmax(const int patch_size);

	/* Set pattern for feature computation */
	void setPattern(const int patch_size, const int wta_k);

	/* Set Gaussain kernel. Size 7, sigma 2 */
	void setGaussianKernel();

	/* Set scale factor for harris score. */
	void setScaleSqSq();

	/* Make offsets for FAST keypoints detection */
	void makeOffsets(int* pitchs, int noctaves);


	/* Find extreme by FAST */
	void hFastDectectWithNMS(unsigned char* image, unsigned char* octave_images, float* vmem, OrbData& result, int* oszp,
		int noctaves, int threshold, int border, bool harris_score);

	/* Compute orientation */
	void hComputeAngle(unsigned char* octave_images, OrbData& result, int* oszp, int noctaves, int patch_size);

	/* Gassian blurring */
	void hGassianBlur(unsigned char* octave_images, int* oszp, int noctaves);

	/* Compute descriptors */
	void hDescribe(unsigned char* octave_images, OrbData& result, unsigned char* desc, int wta_k, int noctaves);

	/* Match descriptors */
	void hMatch(OrbData& result1, OrbData& result2, unsigned char* desc1, unsigned char* desc2);

}