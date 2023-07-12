#pragma once


namespace orb
{

	/* Structure of ORB point */
	struct OrbPoint
	{
		int x;
		int y;
		int octave;
		float score;
		float angle;
		int match;
		int distance;
	};


	/* Structure of ORB matching data */
	struct OrbData
	{
		int num_pts;         // Number of available ORB points
		//int max_pts;         // Number of allocated ORB points
		OrbPoint* h_data;    // Host (CPU) data
		OrbPoint* d_data;    // Device (GPU) data
	};


	/* Score type */
	enum ScoreType
	{
		HARRIS_SCORE = 0,
		FAST_SCORE = 1
	};

}