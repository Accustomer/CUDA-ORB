#include "orb.h"
#include "orbd.h"
// #include "cuda_image.h"




namespace orb
{


	void initOrbData(OrbData& data, const int max_pts, const bool host, const bool dev)
	{
		data.num_pts = 0;
		data.max_pts = max_pts;
		const size_t size = sizeof(OrbPoint) * max_pts;
		data.h_data = host ? (OrbPoint*)malloc(size) : NULL;
		data.d_data = NULL;
		if (dev)
		{
			CHECK(cudaMalloc((void**)&data.d_data, size));
		}
	}


	void freeOrbData(OrbData& data)
	{
		if (data.d_data != NULL)
		{
			CHECK(cudaFree(data.d_data));
		}
		if (data.h_data != NULL)
		{
			free(data.h_data);
		}
		data.num_pts = 0;
		data.max_pts = 0;
	}


	void match(OrbData& data1, OrbData& data2)
	{
		cuHammingMatch(data1, data2);
		//cuFindMaxCorr(data1, data2);
		if (data1.h_data != NULL)
		{
			int* h_ptr = &data1.h_data[0].match;
			int* d_ptr = &data1.d_data[0].match;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(OrbPoint), d_ptr, sizeof(OrbPoint), 4 * sizeof(int), data1.num_pts, cudaMemcpyDeviceToHost));
		}
	}


	void hMatch(OrbData& data1, OrbData& data2)
	{
		for (int i = 0; i < data1.num_pts; i++)
		{
			int min_dist = INT_MAX, snd_dist = INT_MAX;
			int min_idx = -1;
			const unsigned char* f1 = data1.h_data[i].features;
			for (int j = 0; j < data2.num_pts; j++)
			{
				const unsigned char* f2 = data2.h_data[j].features;
				int dist = 0;
				for (int k = 0; k < 32; k++)
				{
					dist += __builtin_popcount(f1[k] ^ f2[k]);  // __popcnt
				}

				if (dist < min_dist)
				{
					snd_dist = min_dist;
					min_dist = dist;
					min_idx = j;
				}
				else if (dist >= min_dist && dist < snd_dist)
				{
					snd_dist = min_dist;
				}
			}

			if (min_dist < snd_dist)
			{
				data1.h_data[i].match = min_idx;
				data1.h_data[i].match_x = data2.h_data[min_idx].x;
				data1.h_data[i].match_y = data2.h_data[min_idx].y;
				data1.h_data[i].distance = min_dist;
			}
			else
			{
				data1.h_data[i].match = -1;
				data1.h_data[i].match_x = -1;
				data1.h_data[i].match_y = -1;
				data1.h_data[i].distance = -1;
			}
		}
	}





	Orbor::Orbor()
	{
	}


	Orbor::~Orbor()
	{
		if (smem)
		{
			CHECK(cudaFree(smem));
		}
		if (vmem)
		{
			CHECK(cudaFree(vmem));
		}
		if (swhps)
		{
			delete[] swhps;
			swhps = nullptr;
		}
		if (ssizes)
		{
			delete[] ssizes;
			ssizes = nullptr;
		}
		if (scales)
		{
			delete[] scales;
			scales = nullptr;
		}
		if (soffsets)
		{
			delete[] soffsets;
			soffsets = nullptr;
		}
	}


	void Orbor::init(int _noctaves, int _edge_threshold, bool _doubled, int _wta_k, ScoreType _score_type, 
		int _patch_size, int _fast_threshold, bool _nonmax_suppression, int _retain_topn, int _width, int _height)
	{
		noctaves = _noctaves;
		edge_threshold = _edge_threshold;
		doubled = _doubled;
		wta_k = _wta_k;
		score_type = _score_type;
		patch_size = _patch_size;
		fast_threshold = _fast_threshold;
		nonmax_suppression = _nonmax_suppression;
		retain_topn = _retain_topn;
		width = _width;
		height = _height;
		setFastThresholdLUT(fast_threshold);
		setUmax(patch_size);
		setPattern(patch_size, wta_k);
		setGaussianKernel();
		//setHammingTable();
		if (score_type == HARRIS_SCORE)
		{
			setScaleSqSq();
		}
	}


	void Orbor::detectAndCompute(unsigned char* image, OrbData& result, int3 whp0, const bool desc)
	{
		// Get address of point counter
		unsigned int* d_point_counter_addr;
		getPointCounter((void**)&d_point_counter_addr);
		CHECK(cudaMemset(d_point_counter_addr, 0, sizeof(unsigned int)));
		setMaxNumPoints(result.max_pts);

		// Allocate memory for pyramid
		unsigned char* tmem = NULL;
		float* wmem = NULL;
		int3* owhps = NULL;
		int* osizes = NULL;
		int* oscales = NULL;
		int* moffsets = NULL;
		static bool last_reused = false;
		const bool reused = whp0.x == width && whp0.y == height;
		if (reused)
		{
			if (smem == NULL)
			{
				swhps = new int3[noctaves];
				ssizes = new int[noctaves];
				scales = new int[noctaves];
				soffsets = new int[noctaves];
				total_size = this->calcSize(width, height, swhps, ssizes, scales, soffsets);
				CHECK(cudaMalloc((void**)&smem, total_size * sizeof(unsigned char)));
				if (nonmax_suppression)
				{
					//CHECK(cudaMalloc((void**)&vmem, whp0.y * whp0.z * sizeof(float)));
					CHECK(cudaMalloc((void**)&vmem, total_size * sizeof(float)));
				}
				
			}
			if (!last_reused)
			{
				makeOffsets(swhps, noctaves);
			}
			tmem = smem; wmem = vmem; owhps = swhps; osizes = ssizes; oscales = scales; moffsets = soffsets;
		}
		else
		{
			owhps = new int3[noctaves];
			osizes = new int[noctaves];
			oscales = new int[noctaves];
			moffsets = new int[noctaves];
			const int tosize = this->calcSize(whp0.x, whp0.y, owhps, osizes, oscales, moffsets);
			CHECK(cudaMalloc((void**)&tmem, tosize * sizeof(unsigned char)));
			if (nonmax_suppression)
			{
				//CHECK(cudaMalloc((void**)&wmem, whp0.y * whp0.z * sizeof(float)));
				CHECK(cudaMalloc((void**)&wmem, tosize * sizeof(float)));
			}
			makeOffsets(owhps, noctaves);
		}
		last_reused = reused;

		// Detect keypoints and compute angles
		if (nonmax_suppression)
		{
			cuFastDectectWithNMS(image, tmem, wmem, result, whp0, owhps, osizes, moffsets, noctaves, doubled, fast_threshold, edge_threshold, score_type == HARRIS_SCORE);
			//cuFastDectectWithNMSAsync(image, tmem, wmem, result, whp0, owhps, osizes, oscales, noctaves, doubled, fast_threshold, edge_threshold, score_type == HARRIS_SCORE);
		}
		else
		{
			//cuCreatePyramid(image, tmem, whp0, owhps, osizes, noctaves, doubled);
			//cuCreatePyramidAsync(image, tmem, whp0, owhps, osizes, noctaves, doubled);
			cuFastDectect(image, tmem, result, whp0, owhps, osizes, moffsets, noctaves, doubled, fast_threshold, edge_threshold);		
			//cuFastDectectAsync(image, tmem, result, whp0, owhps, osizes, oscales, noctaves, doubled, fast_threshold, edge_threshold);
		}
		CHECK(cudaMemcpy(&result.num_pts, d_point_counter_addr, sizeof(unsigned int), cudaMemcpyDeviceToHost));
		result.num_pts = std::min<int>(result.num_pts, result.max_pts);

		// Retain top-n
		if (retain_topn > 0 && retain_topn < result.num_pts)
		{
			cuRetainTopN(result, retain_topn);
		}

		// Compute descriptors
		if (desc)
		{
			// Compute orientation
			//cuComputeAngle(image, result, whp0, patch_size);
			cuComputeAngle(tmem, result, owhps, moffsets, noctaves, patch_size);

			// Blurring
			//unsigned char* bmem = tmem + (doubled ? osizes[0] : 0);
			//cuGassianBlur(image, bmem, whp0);
			cuGassianBlur(tmem, owhps, moffsets, noctaves);

			// Compute descriptors
			//cuDescribe(bmem, result, whp0, wta_k);
			cuDescribe(tmem, result, wta_k);
		}

		// Rescale coordinates
		cuRescale(result, oscales, noctaves);

		// Copy point data to host
		if (result.h_data != NULL)
		{
			int* h_ptr = &result.h_data[0].x;
			int* d_ptr = &result.d_data[0].x;
			CHECK(cudaMemcpy2D(h_ptr, sizeof(OrbPoint), d_ptr, sizeof(OrbPoint), (desc ? 13 : 4) * sizeof(int), result.num_pts, cudaMemcpyDeviceToHost));
		}
		
		// Post-processing
		if (reused)
		{
			CHECK(cudaMemset(smem, 0, total_size * sizeof(unsigned char)));
			if (nonmax_suppression)
			{
				CHECK(cudaMemset(vmem, 0, total_size * sizeof(float)));
			}
		}
		else
		{
			if (tmem)
			{
				CHECK(cudaFree(tmem));
			}
			if (wmem)
			{
				CHECK(cudaFree(wmem));
			}			
			delete[] osizes; osizes = nullptr;
			delete[] owhps; owhps = nullptr;
			delete[] oscales; oscales = nullptr;
			delete[] moffsets; moffsets = nullptr;
		}
	}


	int Orbor::calcSize(int w, int h, int3* owhps, int* osizes, int* oscales, int* moffsets)
	{
		// The size of first octave
		owhps[0].x = w;
		owhps[0].y = h;
		oscales[0] = 0;
		if (doubled)
		{
			owhps[0].x += w;
			owhps[0].y += h;
			oscales[0] = -1;
		}
		owhps[0].z = iAlignUp(owhps[0].x, 128);
		osizes[0] = owhps[0].y * owhps[0].z;
		moffsets[0] = 0;

		// Size of residual octaves
		for (int i = 1, j = 0; i < noctaves; i++, j++)
		{
			owhps[i].x = owhps[j].x >> 1;
			owhps[i].y = owhps[j].y >> 1;
			owhps[i].z = iAlignUp(owhps[i].x, 128);
			osizes[i] = owhps[i].y * owhps[i].z;
			oscales[i] = oscales[j] + 1;
			moffsets[i] = moffsets[j] + osizes[j];
		}
			
		//total_size = tosize * sizeof(unsigned char);
		return moffsets[noctaves - 1] + osizes[noctaves - 1];
	}



}

