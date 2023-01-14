#pragma once
#include "orb_structures.h"
#include "cuda_utils.h"


namespace orb
{

	/* Initialize memory for ORB data */
	void initOrbData(OrbData& data, const int max_pts, const bool host, const bool dev);

	/* Free memory for ORB data */
	void freeOrbData(OrbData& datav);

	/* Match keypoints by descriptors */
	void match(OrbData& data1, OrbData& data2);
	void hMatch(OrbData& data1, OrbData& data2);





	class Orbor
	{
	public:
		Orbor();
		~Orbor();

		/* Initialization */
		void init(int _noctaves = 5, int _edge_threshold = 31, bool _doubled = false, int _wta_k = 2,
			ScoreType _score_type = ScoreType::HARRIS_SCORE, int _patch_size = 31, int _fast_threshold = 20, 
			bool _nonmax_suppression = true, int _retain_topn = -1, int _width = -1, int _height = -1);

		/* Detect keypoints and compute descriptors */
		void detectAndCompute(unsigned char* image, OrbData& result, int3 whp0, const bool desc = true);

	private:
		//static const int kbytes = 32;
		// The number of pyramid levels.
		int noctaves = 5;	
		// This is size of the border where the features are not detected.
		int edge_threshold = 31;
		// Zoom in image to double size
		bool doubled = false;
		// The number of points that produce each element of the oriented BRIEF descriptor.
		int wta_k = 2;
		// The default HARRIS_SCORE means that Harris algorithm is used to rank features
		// (the score is written to KeyPoint::scoreand is used to retain best nfeatures features);
		// FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
		// but it is a little faster to compute.
		ScoreType score_type = ScoreType::HARRIS_SCORE;
		// The size of the patch used by the oriented BRIEF descriptor.
		int patch_size = 31;
		// The fast threshold
		int fast_threshold = 20;
		// Apply NMS for keypoints
		bool nonmax_suppression = true;
		// The top-n best keypoints retained for matching. If less and equal to 0, retain all
		int retain_topn = -1;
		// Size of image for reusing
		int width = -1;
		int height = -1;
		// Memory for reusing
		unsigned char* smem = NULL;		// Shared memory of scaled images
		float* vmem = NULL;				// Shared memory of score maps for non-max suppression
		int3* swhps = NULL;
		int* ssizes = NULL;
		int* scales = NULL;
		int* soffsets = NULL;
		int total_size = 0;

		/* Calculate size */
		int calcSize(int w, int h, int3* owhps, int* osizes, int* oscales, int* moffsets);


	};

}