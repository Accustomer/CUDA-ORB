#include "orbd.h"
#include <device_launch_parameters.h>
// #include <opencv2/core.hpp>


namespace orb
{
#define X1			1024
#define X2			32
#define K			(FAST_PATTERN / 2)
#define N			(FAST_PATTERN + K + 1)
#define HARRIS_K	0.04f
#define GAUSS_K		7
#define HGK			(GAUSS_K / 2)
#define MAX_DIST	64
#define FLT_MAX     3.402823466e+38F


	__constant__ int d_max_num_points;
	__constant__ float d_scale_sq_sq;
	__device__ unsigned int d_point_counter;
	__constant__ int dpixel[25 * MAX_OCTAVE];
	__constant__ unsigned char dthresh_table[512];
	__constant__ int d_umax[MAX_PATCH / 2 + 2];
	__constant__ int2 d_pattern[512];
	__constant__ float d_gauss[HGK + 1];
	__constant__ int ofs[HARRIS_SIZE * HARRIS_SIZE];
	__constant__ int angle_param[MAX_OCTAVE * 2];
	//__constant__ int hamming_table[256];


	static int bit_pattern_31_[256 * 4] = {
	8,-3, 9,5/*mean (0), correlation (0)*/,
	4,2, 7,-12/*mean (1.12461e-05), correlation (0.0437584)*/,
	-11,9, -8,2/*mean (3.37382e-05), correlation (0.0617409)*/,
	7,-12, 12,-13/*mean (5.62303e-05), correlation (0.0636977)*/,
	2,-13, 2,12/*mean (0.000134953), correlation (0.085099)*/,
	1,-7, 1,6/*mean (0.000528565), correlation (0.0857175)*/,
	-2,-10, -2,-4/*mean (0.0188821), correlation (0.0985774)*/,
	-13,-13, -11,-8/*mean (0.0363135), correlation (0.0899616)*/,
	-13,-3, -12,-9/*mean (0.121806), correlation (0.099849)*/,
	10,4, 11,9/*mean (0.122065), correlation (0.093285)*/,
	-13,-8, -8,-9/*mean (0.162787), correlation (0.0942748)*/,
	-11,7, -9,12/*mean (0.21561), correlation (0.0974438)*/,
	7,7, 12,6/*mean (0.160583), correlation (0.130064)*/,
	-4,-5, -3,0/*mean (0.228171), correlation (0.132998)*/,
	-13,2, -12,-3/*mean (0.00997526), correlation (0.145926)*/,
	-9,0, -7,5/*mean (0.198234), correlation (0.143636)*/,
	12,-6, 12,-1/*mean (0.0676226), correlation (0.16689)*/,
	-3,6, -2,12/*mean (0.166847), correlation (0.171682)*/,
	-6,-13, -4,-8/*mean (0.101215), correlation (0.179716)*/,
	11,-13, 12,-8/*mean (0.200641), correlation (0.192279)*/,
	4,7, 5,1/*mean (0.205106), correlation (0.186848)*/,
	5,-3, 10,-3/*mean (0.234908), correlation (0.192319)*/,
	3,-7, 6,12/*mean (0.0709964), correlation (0.210872)*/,
	-8,-7, -6,-2/*mean (0.0939834), correlation (0.212589)*/,
	-2,11, -1,-10/*mean (0.127778), correlation (0.20866)*/,
	-13,12, -8,10/*mean (0.14783), correlation (0.206356)*/,
	-7,3, -5,-3/*mean (0.182141), correlation (0.198942)*/,
	-4,2, -3,7/*mean (0.188237), correlation (0.21384)*/,
	-10,-12, -6,11/*mean (0.14865), correlation (0.23571)*/,
	5,-12, 6,-7/*mean (0.222312), correlation (0.23324)*/,
	5,-6, 7,-1/*mean (0.229082), correlation (0.23389)*/,
	1,0, 4,-5/*mean (0.241577), correlation (0.215286)*/,
	9,11, 11,-13/*mean (0.00338507), correlation (0.251373)*/,
	4,7, 4,12/*mean (0.131005), correlation (0.257622)*/,
	2,-1, 4,4/*mean (0.152755), correlation (0.255205)*/,
	-4,-12, -2,7/*mean (0.182771), correlation (0.244867)*/,
	-8,-5, -7,-10/*mean (0.186898), correlation (0.23901)*/,
	4,11, 9,12/*mean (0.226226), correlation (0.258255)*/,
	0,-8, 1,-13/*mean (0.0897886), correlation (0.274827)*/,
	-13,-2, -8,2/*mean (0.148774), correlation (0.28065)*/,
	-3,-2, -2,3/*mean (0.153048), correlation (0.283063)*/,
	-6,9, -4,-9/*mean (0.169523), correlation (0.278248)*/,
	8,12, 10,7/*mean (0.225337), correlation (0.282851)*/,
	0,9, 1,3/*mean (0.226687), correlation (0.278734)*/,
	7,-5, 11,-10/*mean (0.00693882), correlation (0.305161)*/,
	-13,-6, -11,0/*mean (0.0227283), correlation (0.300181)*/,
	10,7, 12,1/*mean (0.125517), correlation (0.31089)*/,
	-6,-3, -6,12/*mean (0.131748), correlation (0.312779)*/,
	10,-9, 12,-4/*mean (0.144827), correlation (0.292797)*/,
	-13,8, -8,-12/*mean (0.149202), correlation (0.308918)*/,
	-13,0, -8,-4/*mean (0.160909), correlation (0.310013)*/,
	3,3, 7,8/*mean (0.177755), correlation (0.309394)*/,
	5,7, 10,-7/*mean (0.212337), correlation (0.310315)*/,
	-1,7, 1,-12/*mean (0.214429), correlation (0.311933)*/,
	3,-10, 5,6/*mean (0.235807), correlation (0.313104)*/,
	2,-4, 3,-10/*mean (0.00494827), correlation (0.344948)*/,
	-13,0, -13,5/*mean (0.0549145), correlation (0.344675)*/,
	-13,-7, -12,12/*mean (0.103385), correlation (0.342715)*/,
	-13,3, -11,8/*mean (0.134222), correlation (0.322922)*/,
	-7,12, -4,7/*mean (0.153284), correlation (0.337061)*/,
	6,-10, 12,8/*mean (0.154881), correlation (0.329257)*/,
	-9,-1, -7,-6/*mean (0.200967), correlation (0.33312)*/,
	-2,-5, 0,12/*mean (0.201518), correlation (0.340635)*/,
	-12,5, -7,5/*mean (0.207805), correlation (0.335631)*/,
	3,-10, 8,-13/*mean (0.224438), correlation (0.34504)*/,
	-7,-7, -4,5/*mean (0.239361), correlation (0.338053)*/,
	-3,-2, -1,-7/*mean (0.240744), correlation (0.344322)*/,
	2,9, 5,-11/*mean (0.242949), correlation (0.34145)*/,
	-11,-13, -5,-13/*mean (0.244028), correlation (0.336861)*/,
	-1,6, 0,-1/*mean (0.247571), correlation (0.343684)*/,
	5,-3, 5,2/*mean (0.000697256), correlation (0.357265)*/,
	-4,-13, -4,12/*mean (0.00213675), correlation (0.373827)*/,
	-9,-6, -9,6/*mean (0.0126856), correlation (0.373938)*/,
	-12,-10, -8,-4/*mean (0.0152497), correlation (0.364237)*/,
	10,2, 12,-3/*mean (0.0299933), correlation (0.345292)*/,
	7,12, 12,12/*mean (0.0307242), correlation (0.366299)*/,
	-7,-13, -6,5/*mean (0.0534975), correlation (0.368357)*/,
	-4,9, -3,4/*mean (0.099865), correlation (0.372276)*/,
	7,-1, 12,2/*mean (0.117083), correlation (0.364529)*/,
	-7,6, -5,1/*mean (0.126125), correlation (0.369606)*/,
	-13,11, -12,5/*mean (0.130364), correlation (0.358502)*/,
	-3,7, -2,-6/*mean (0.131691), correlation (0.375531)*/,
	7,-8, 12,-7/*mean (0.160166), correlation (0.379508)*/,
	-13,-7, -11,-12/*mean (0.167848), correlation (0.353343)*/,
	1,-3, 12,12/*mean (0.183378), correlation (0.371916)*/,
	2,-6, 3,0/*mean (0.228711), correlation (0.371761)*/,
	-4,3, -2,-13/*mean (0.247211), correlation (0.364063)*/,
	-1,-13, 1,9/*mean (0.249325), correlation (0.378139)*/,
	7,1, 8,-6/*mean (0.000652272), correlation (0.411682)*/,
	1,-1, 3,12/*mean (0.00248538), correlation (0.392988)*/,
	9,1, 12,6/*mean (0.0206815), correlation (0.386106)*/,
	-1,-9, -1,3/*mean (0.0364485), correlation (0.410752)*/,
	-13,-13, -10,5/*mean (0.0376068), correlation (0.398374)*/,
	7,7, 10,12/*mean (0.0424202), correlation (0.405663)*/,
	12,-5, 12,9/*mean (0.0942645), correlation (0.410422)*/,
	6,3, 7,11/*mean (0.1074), correlation (0.413224)*/,
	5,-13, 6,10/*mean (0.109256), correlation (0.408646)*/,
	2,-12, 2,3/*mean (0.131691), correlation (0.416076)*/,
	3,8, 4,-6/*mean (0.165081), correlation (0.417569)*/,
	2,6, 12,-13/*mean (0.171874), correlation (0.408471)*/,
	9,-12, 10,3/*mean (0.175146), correlation (0.41296)*/,
	-8,4, -7,9/*mean (0.183682), correlation (0.402956)*/,
	-11,12, -4,-6/*mean (0.184672), correlation (0.416125)*/,
	1,12, 2,-8/*mean (0.191487), correlation (0.386696)*/,
	6,-9, 7,-4/*mean (0.192668), correlation (0.394771)*/,
	2,3, 3,-2/*mean (0.200157), correlation (0.408303)*/,
	6,3, 11,0/*mean (0.204588), correlation (0.411762)*/,
	3,-3, 8,-8/*mean (0.205904), correlation (0.416294)*/,
	7,8, 9,3/*mean (0.213237), correlation (0.409306)*/,
	-11,-5, -6,-4/*mean (0.243444), correlation (0.395069)*/,
	-10,11, -5,10/*mean (0.247672), correlation (0.413392)*/,
	-5,-8, -3,12/*mean (0.24774), correlation (0.411416)*/,
	-10,5, -9,0/*mean (0.00213675), correlation (0.454003)*/,
	8,-1, 12,-6/*mean (0.0293635), correlation (0.455368)*/,
	4,-6, 6,-11/*mean (0.0404971), correlation (0.457393)*/,
	-10,12, -8,7/*mean (0.0481107), correlation (0.448364)*/,
	4,-2, 6,7/*mean (0.050641), correlation (0.455019)*/,
	-2,0, -2,12/*mean (0.0525978), correlation (0.44338)*/,
	-5,-8, -5,2/*mean (0.0629667), correlation (0.457096)*/,
	7,-6, 10,12/*mean (0.0653846), correlation (0.445623)*/,
	-9,-13, -8,-8/*mean (0.0858749), correlation (0.449789)*/,
	-5,-13, -5,-2/*mean (0.122402), correlation (0.450201)*/,
	8,-8, 9,-13/*mean (0.125416), correlation (0.453224)*/,
	-9,-11, -9,0/*mean (0.130128), correlation (0.458724)*/,
	1,-8, 1,-2/*mean (0.132467), correlation (0.440133)*/,
	7,-4, 9,1/*mean (0.132692), correlation (0.454)*/,
	-2,1, -1,-4/*mean (0.135695), correlation (0.455739)*/,
	11,-6, 12,-11/*mean (0.142904), correlation (0.446114)*/,
	-12,-9, -6,4/*mean (0.146165), correlation (0.451473)*/,
	3,7, 7,12/*mean (0.147627), correlation (0.456643)*/,
	5,5, 10,8/*mean (0.152901), correlation (0.455036)*/,
	0,-4, 2,8/*mean (0.167083), correlation (0.459315)*/,
	-9,12, -5,-13/*mean (0.173234), correlation (0.454706)*/,
	0,7, 2,12/*mean (0.18312), correlation (0.433855)*/,
	-1,2, 1,7/*mean (0.185504), correlation (0.443838)*/,
	5,11, 7,-9/*mean (0.185706), correlation (0.451123)*/,
	3,5, 6,-8/*mean (0.188968), correlation (0.455808)*/,
	-13,-4, -8,9/*mean (0.191667), correlation (0.459128)*/,
	-5,9, -3,-3/*mean (0.193196), correlation (0.458364)*/,
	-4,-7, -3,-12/*mean (0.196536), correlation (0.455782)*/,
	6,5, 8,0/*mean (0.1972), correlation (0.450481)*/,
	-7,6, -6,12/*mean (0.199438), correlation (0.458156)*/,
	-13,6, -5,-2/*mean (0.211224), correlation (0.449548)*/,
	1,-10, 3,10/*mean (0.211718), correlation (0.440606)*/,
	4,1, 8,-4/*mean (0.213034), correlation (0.443177)*/,
	-2,-2, 2,-13/*mean (0.234334), correlation (0.455304)*/,
	2,-12, 12,12/*mean (0.235684), correlation (0.443436)*/,
	-2,-13, 0,-6/*mean (0.237674), correlation (0.452525)*/,
	4,1, 9,3/*mean (0.23962), correlation (0.444824)*/,
	-6,-10, -3,-5/*mean (0.248459), correlation (0.439621)*/,
	-3,-13, -1,1/*mean (0.249505), correlation (0.456666)*/,
	7,5, 12,-11/*mean (0.00119208), correlation (0.495466)*/,
	4,-2, 5,-7/*mean (0.00372245), correlation (0.484214)*/,
	-13,9, -9,-5/*mean (0.00741116), correlation (0.499854)*/,
	7,1, 8,6/*mean (0.0208952), correlation (0.499773)*/,
	7,-8, 7,6/*mean (0.0220085), correlation (0.501609)*/,
	-7,-4, -7,1/*mean (0.0233806), correlation (0.496568)*/,
	-8,11, -7,-8/*mean (0.0236505), correlation (0.489719)*/,
	-13,6, -12,-8/*mean (0.0268781), correlation (0.503487)*/,
	2,4, 3,9/*mean (0.0323324), correlation (0.501938)*/,
	10,-5, 12,3/*mean (0.0399235), correlation (0.494029)*/,
	-6,-5, -6,7/*mean (0.0420153), correlation (0.486579)*/,
	8,-3, 9,-8/*mean (0.0548021), correlation (0.484237)*/,
	2,-12, 2,8/*mean (0.0616622), correlation (0.496642)*/,
	-11,-2, -10,3/*mean (0.0627755), correlation (0.498563)*/,
	-12,-13, -7,-9/*mean (0.0829622), correlation (0.495491)*/,
	-11,0, -10,-5/*mean (0.0843342), correlation (0.487146)*/,
	5,-3, 11,8/*mean (0.0929937), correlation (0.502315)*/,
	-2,-13, -1,12/*mean (0.113327), correlation (0.48941)*/,
	-1,-8, 0,9/*mean (0.132119), correlation (0.467268)*/,
	-13,-11, -12,-5/*mean (0.136269), correlation (0.498771)*/,
	-10,-2, -10,11/*mean (0.142173), correlation (0.498714)*/,
	-3,9, -2,-13/*mean (0.144141), correlation (0.491973)*/,
	2,-3, 3,2/*mean (0.14892), correlation (0.500782)*/,
	-9,-13, -4,0/*mean (0.150371), correlation (0.498211)*/,
	-4,6, -3,-10/*mean (0.152159), correlation (0.495547)*/,
	-4,12, -2,-7/*mean (0.156152), correlation (0.496925)*/,
	-6,-11, -4,9/*mean (0.15749), correlation (0.499222)*/,
	6,-3, 6,11/*mean (0.159211), correlation (0.503821)*/,
	-13,11, -5,5/*mean (0.162427), correlation (0.501907)*/,
	11,11, 12,6/*mean (0.16652), correlation (0.497632)*/,
	7,-5, 12,-2/*mean (0.169141), correlation (0.484474)*/,
	-1,12, 0,7/*mean (0.169456), correlation (0.495339)*/,
	-4,-8, -3,-2/*mean (0.171457), correlation (0.487251)*/,
	-7,1, -6,7/*mean (0.175), correlation (0.500024)*/,
	-13,-12, -8,-13/*mean (0.175866), correlation (0.497523)*/,
	-7,-2, -6,-8/*mean (0.178273), correlation (0.501854)*/,
	-8,5, -6,-9/*mean (0.181107), correlation (0.494888)*/,
	-5,-1, -4,5/*mean (0.190227), correlation (0.482557)*/,
	-13,7, -8,10/*mean (0.196739), correlation (0.496503)*/,
	1,5, 5,-13/*mean (0.19973), correlation (0.499759)*/,
	1,0, 10,-13/*mean (0.204465), correlation (0.49873)*/,
	9,12, 10,-1/*mean (0.209334), correlation (0.49063)*/,
	5,-8, 10,-9/*mean (0.211134), correlation (0.503011)*/,
	-1,11, 1,-13/*mean (0.212), correlation (0.499414)*/,
	-9,-3, -6,2/*mean (0.212168), correlation (0.480739)*/,
	-1,-10, 1,12/*mean (0.212731), correlation (0.502523)*/,
	-13,1, -8,-10/*mean (0.21327), correlation (0.489786)*/,
	8,-11, 10,-6/*mean (0.214159), correlation (0.488246)*/,
	2,-13, 3,-6/*mean (0.216993), correlation (0.50287)*/,
	7,-13, 12,-9/*mean (0.223639), correlation (0.470502)*/,
	-10,-10, -5,-7/*mean (0.224089), correlation (0.500852)*/,
	-10,-8, -8,-13/*mean (0.228666), correlation (0.502629)*/,
	4,-6, 8,5/*mean (0.22906), correlation (0.498305)*/,
	3,12, 8,-13/*mean (0.233378), correlation (0.503825)*/,
	-4,2, -3,-3/*mean (0.234323), correlation (0.476692)*/,
	5,-13, 10,-12/*mean (0.236392), correlation (0.475462)*/,
	4,-13, 5,-1/*mean (0.236842), correlation (0.504132)*/,
	-9,9, -4,3/*mean (0.236977), correlation (0.497739)*/,
	0,3, 3,-9/*mean (0.24314), correlation (0.499398)*/,
	-12,1, -6,1/*mean (0.243297), correlation (0.489447)*/,
	3,2, 4,-8/*mean (0.00155196), correlation (0.553496)*/,
	-10,-10, -10,9/*mean (0.00239541), correlation (0.54297)*/,
	8,-13, 12,12/*mean (0.0034413), correlation (0.544361)*/,
	-8,-12, -6,-5/*mean (0.003565), correlation (0.551225)*/,
	2,2, 3,7/*mean (0.00835583), correlation (0.55285)*/,
	10,6, 11,-8/*mean (0.00885065), correlation (0.540913)*/,
	6,8, 8,-12/*mean (0.0101552), correlation (0.551085)*/,
	-7,10, -6,5/*mean (0.0102227), correlation (0.533635)*/,
	-3,-9, -3,9/*mean (0.0110211), correlation (0.543121)*/,
	-1,-13, -1,5/*mean (0.0113473), correlation (0.550173)*/,
	-3,-7, -3,4/*mean (0.0140913), correlation (0.554774)*/,
	-8,-2, -8,3/*mean (0.017049), correlation (0.55461)*/,
	4,2, 12,12/*mean (0.01778), correlation (0.546921)*/,
	2,-5, 3,11/*mean (0.0224022), correlation (0.549667)*/,
	6,-9, 11,-13/*mean (0.029161), correlation (0.546295)*/,
	3,-1, 7,12/*mean (0.0303081), correlation (0.548599)*/,
	11,-1, 12,4/*mean (0.0355151), correlation (0.523943)*/,
	-3,0, -3,6/*mean (0.0417904), correlation (0.543395)*/,
	4,-11, 4,12/*mean (0.0487292), correlation (0.542818)*/,
	2,-4, 2,1/*mean (0.0575124), correlation (0.554888)*/,
	-10,-6, -8,1/*mean (0.0594242), correlation (0.544026)*/,
	-13,7, -11,1/*mean (0.0597391), correlation (0.550524)*/,
	-13,12, -11,-13/*mean (0.0608974), correlation (0.55383)*/,
	6,0, 11,-13/*mean (0.065126), correlation (0.552006)*/,
	0,-1, 1,4/*mean (0.074224), correlation (0.546372)*/,
	-13,3, -9,-2/*mean (0.0808592), correlation (0.554875)*/,
	-9,8, -6,-3/*mean (0.0883378), correlation (0.551178)*/,
	-13,-6, -8,-2/*mean (0.0901035), correlation (0.548446)*/,
	5,-9, 8,10/*mean (0.0949843), correlation (0.554694)*/,
	2,7, 3,-9/*mean (0.0994152), correlation (0.550979)*/,
	-1,-6, -1,-1/*mean (0.10045), correlation (0.552714)*/,
	9,5, 11,-2/*mean (0.100686), correlation (0.552594)*/,
	11,-3, 12,-8/*mean (0.101091), correlation (0.532394)*/,
	3,0, 3,5/*mean (0.101147), correlation (0.525576)*/,
	-1,4, 0,10/*mean (0.105263), correlation (0.531498)*/,
	3,-6, 4,5/*mean (0.110785), correlation (0.540491)*/,
	-13,0, -10,5/*mean (0.112798), correlation (0.536582)*/,
	5,8, 12,11/*mean (0.114181), correlation (0.555793)*/,
	8,9, 9,-6/*mean (0.117431), correlation (0.553763)*/,
	7,-4, 8,-12/*mean (0.118522), correlation (0.553452)*/,
	-10,4, -10,9/*mean (0.12094), correlation (0.554785)*/,
	7,3, 12,4/*mean (0.122582), correlation (0.555825)*/,
	9,-7, 10,-2/*mean (0.124978), correlation (0.549846)*/,
	7,0, 12,-2/*mean (0.127002), correlation (0.537452)*/,
	-1,-6, 0,-11/*mean (0.127148), correlation (0.547401)*/
	};


	void setMaxNumPoints(const int num)
	{
		CHECK(cudaMemcpyToSymbol(d_max_num_points, &num, sizeof(int), 0, cudaMemcpyHostToDevice));
	}


	void setScaleSqSq()
	{
		float scale = 1.f / (4 * HARRIS_SIZE * 255.f);	//  * 255.f
		float scale_sq_sq = scale * scale * scale * scale;
		CHECK(cudaMemcpyToSymbol(d_scale_sq_sq, &scale_sq_sq, sizeof(float), 0, cudaMemcpyHostToDevice));
	}


	void setHarrisOffsets(const int pitch)
	{
		static int p = -1;
		if (p != pitch)
		{
			int hofs[HARRIS_SIZE * HARRIS_SIZE];
			for (int i = 0; i < HARRIS_SIZE; i++)
			{
				for (int j = 0; j < HARRIS_SIZE; j++)
				{
					hofs[i * HARRIS_SIZE + j] = i * pitch + j;
				}
			}
			CHECK(cudaMemcpyToSymbol(ofs, hofs, HARRIS_SIZE * HARRIS_SIZE * sizeof(int), 0, cudaMemcpyHostToDevice));
			p = pitch;
		}
	}


	void setUmax(const int patch_size)
	{
		int half_patch = patch_size / 2;
		int* h_umax = new int[half_patch + 2];
		h_umax[half_patch + 1] = 0;

		float v = half_patch * sqrtf(2.f) / 2;
		int vmax = (int)floorf(v + 1);
		int vmin = (int)ceilf(v);
		for (int i = 0; i <= vmax; i++)
		{
			h_umax[i] = (int)roundf(sqrtf(half_patch * half_patch - i * i));
		}

		// Make sure we are symmetric
		for (int i = half_patch, v0 = 0; i >= vmin; --i)
		{
			while (h_umax[v0] == h_umax[v0 + 1])
				++v0;
			h_umax[i] = v0;
			++v0;
		}

		// Copy to device
		CHECK(cudaMemcpyToSymbol(d_umax, h_umax, sizeof(int) * (half_patch + 2), 0, cudaMemcpyHostToDevice));
		delete[] h_umax; h_umax = nullptr;
	}


	void getPointCounter(void** addr)
	{
		CHECK(cudaGetSymbolAddress(addr, d_point_counter));
	}


	void makeOffsets(int3* owhps, int noctaves)
	{
#if (FAST_PATTERN == 16)
		const int offsets[16][2] = {
			{0,  3}, { 1,  3}, { 2,  2}, { 3,  1}, { 3, 0}, { 3, -1}, { 2, -2}, { 1, -3}, 
			{0, -3}, {-1, -3}, {-2, -2}, {-3, -1}, {-3, 0}, {-3,  1}, {-2,  2}, {-1,  3}
		};
#elif (FAST_PATTERN == 12)
		const int offsets[12][2] = {
			{0,  2}, { 1,  2}, { 2,  1}, { 2, 0}, { 2, -1}, { 1, -2},
			{0, -2}, {-1, -2}, {-2, -1}, {-2, 0}, {-2,  1}, {-1,  2}
		};
#elif (FAST_PATTERN == 8)
		const int offsets[8][2] = {
			{0,  1}, { 1,  1}, { 1, 0}, { 1, -1},
			{0, -1}, {-1, -1}, {-1, 0}, {-1,  1}
		};
#endif // FAST_PATTERN

		int* hpixel = new int[25 * noctaves];
		int* temp_pixel = hpixel;
		for (int i = 0; i < noctaves; i++)
		{
			int k = 0;
			for (; k < FAST_PATTERN; k++)
				temp_pixel[k] = offsets[k][0] + offsets[k][1] * owhps[i].z;
			for (; k < 25; k++)
				temp_pixel[k] = temp_pixel[k - FAST_PATTERN];
			temp_pixel += 25;
		}

		CHECK(cudaMemcpyToSymbol(dpixel, hpixel, 25 * noctaves * sizeof(int), 0, cudaMemcpyHostToDevice));
		delete[] hpixel; hpixel = nullptr;
	}


	void setFastThresholdLUT(int fast_threshold)
	{
		unsigned char hthreshold_tab[512];
		for (int i = -255, j = 0; i <= 255; i++, j++)
			hthreshold_tab[j] = (unsigned char)(i < -fast_threshold ? 1 : i > fast_threshold ? 2 : 0);
		CHECK(cudaMemcpyToSymbol(dthresh_table, hthreshold_tab, 512 * sizeof(unsigned char), 0, cudaMemcpyHostToDevice));
	}


	void setPattern(const int patch_size, const int wta_k)
	{
		const int npoints = 512;
		int2 patternbuf[npoints];
		const int2* pattern0 = (const int2*)bit_pattern_31_;
		if (patch_size != 31)
		{
			pattern0 = patternbuf;
			// we always start with a fixed seed, to make patterns the same on each run
			srand(0x34985739);
			for (int i = 0; i < npoints; i++)
			{
				patternbuf[i].x = rand() % patch_size - -patch_size / 2;
				patternbuf[i].y = rand() % patch_size - -patch_size / 2;
			}
		}

		if (wta_k == 2)
		{
			//pattern = new int2[npoints];
			//memcpy(pattern, pattern0, npoints * sizeof(int2));
			CHECK(cudaMemcpyToSymbol(d_pattern, pattern0, npoints * sizeof(int2), 0, cudaMemcpyHostToDevice));
		}
		else
		{
			//initializeOrbPattern(pattern0, pattern, ntuples, wta_k, npoints);
			srand(0x12345678);
			int i, k, k1;
			int ntuples = 32 * 4;			
			int2* pattern = new int2[ntuples * wta_k];
			for (i = 0; i < ntuples; i++)
			{
				for (k = 0; k < wta_k; k++)
				{
					while (true)
					{
						int idx = rand() % npoints;
						int2 pt = pattern0[idx];
						for (k1 = 0; k1 < k; k1++)
						{
							int2 pt1 = pattern[wta_k * i + k1];
							if (pt.x == pt1.x && pt.y == pt1.y)
								break;
						}
						if (k1 == k)
						{
							pattern[wta_k * i + k] = pt;
							break;
						}
					}
				}
			}

			CHECK(cudaMemcpyToSymbol(d_pattern, pattern, ntuples * wta_k * sizeof(int2), 0, cudaMemcpyHostToDevice));
			delete[] pattern; pattern = nullptr;
		}
	}


	void setGaussianKernel()
	{
		const float sigma = 2;
		const float svar = -1.f / (2 * sigma * sigma);

		float kernel[HGK + 1];
		float kersum = 0;
		for (int i = 0, j = HGK; i <= HGK; i++, j--)
		{
			kernel[i] = expf(j * j * svar);
			kersum += kernel[i] + (j == 0 ? 0 : kernel[i]);
		}

		kersum = 1.f / kersum;
		for (int i = 0; i <= HGK; i++)
		{
			kernel[i] *= kersum;
		}

		CHECK(cudaMemcpyToSymbol(d_gauss, kernel, (HGK + 1) * sizeof(float), 0, cudaMemcpyHostToDevice));
	}


	void setAngleParam(int* param)
	{
		CHECK(cudaMemcpyToSymbol(angle_param, param, MAX_OCTAVE * 2 * sizeof(int), 0, cudaMemcpyHostToDevice));
	}


	//void setHammingTable()
	//{
	//	int htable[256];
	//	for (unsigned int i = 0; i < 256; i++)
	//	{
	//		htable[i] = __popcnt(i);
	//	}
	//	CHECK(cudaMemcpyToSymbol(hamming_table, htable, 256 * sizeof(int), 0, cudaMemcpyHostToDevice));
	//}



	__global__ void doubleImage(unsigned char* src, unsigned char* dst, int3 swhp, int3 dwhp)
	{
		int xl = blockIdx.x * X2 + threadIdx.x;
		int yu = blockIdx.y * X2 + threadIdx.y;
		if (xl < swhp.x && yu < swhp.y)
		{
			int xr = min(xl + 1, swhp.x - 1);
			int yd = min(yu + 1, swhp.y - 1);
			int ustart = yu * swhp.z;
			int dstart = yd * swhp.z;
			int vul = __uchar2int(src[ustart + xl]);
			int vur = __uchar2int(src[ustart + xr]);
			int vdl = __uchar2int(src[dstart + xl]);
			int vdr = __uchar2int(src[dstart + xr]);

			int rid = (yu + yu) * dwhp.z + (xl + xl);
			dst[rid] = vul;
			dst[rid + 1] = (vul + vur) >> 1;

			rid += dwhp.z;
			dst[rid] = (vul + vdl) >> 1;
			dst[rid + 1] = __float2int_rn((vul + vur + vdl + vdr) * 0.25f);
		}
	}


	__global__ void scaleDown(unsigned char* src, unsigned char* dst, int3 swhp, int3 dwhp, int factor)
	{
		int xi = blockIdx.x * X2 + threadIdx.x;
		int yi = blockIdx.y * X2 + threadIdx.y;
		if (xi < dwhp.x && yi < dwhp.z)
		{
			int di = yi * dwhp.z + xi;
			int si = (yi * swhp.z + xi) * factor;
			dst[di] = src[si];
		}
	}


#if (FAST_PATTERN == 16)
	
	__device__ float fastScore(const unsigned char* ptr, const int* pixel, int threshold)
	{
		int k, v = ptr[0];
		short d[N];
		for (k = 0; k < N; k++)
			d[k] = (short)(v - ptr[pixel[k]]);

		int a0 = threshold;
		for (k = 0; k < 16; k += 2)
		{
			int a = min((int)d[k + 1], (int)d[k + 2]);
			a = min(a, (int)d[k + 3]);
			if (a <= a0)
				continue;
			a = min(a, (int)d[k + 4]);
			a = min(a, (int)d[k + 5]);
			a = min(a, (int)d[k + 6]);
			a = min(a, (int)d[k + 7]);
			a = min(a, (int)d[k + 8]);
			a0 = max(a0, min(a, (int)d[k]));
			a0 = max(a0, min(a, (int)d[k + 9]));
		}

		int b0 = -a0;
		for (k = 0; k < 16; k += 2)
		{
			int b = max((int)d[k + 1], (int)d[k + 2]);
			b = max(b, (int)d[k + 3]);
			b = max(b, (int)d[k + 4]);
			b = max(b, (int)d[k + 5]);
			if (b >= b0)
				continue;
			b = max(b, (int)d[k + 6]);
			b = max(b, (int)d[k + 7]);
			b = max(b, (int)d[k + 8]);

			b0 = min(b0, max(b, (int)d[k]));
			b0 = min(b0, max(b, (int)d[k + 9]));
		}

		threshold = -b0 - 1;
		return __int2float_rn(threshold);
	}

#elif (FAST_PATTERN == 12)

	__device__ float fastScore(const unsigned char* ptr, const int* pixel, int threshold)
	{
		int k, v = ptr[0];
		short d[N + 4];
		for (k = 0; k < N; k++)
			d[k] = (short)(v - ptr[pixel[k]]);
		int a0 = threshold;
		for (k = 0; k < 12; k += 2)
		{
			int a = min((int)d[k + 1], (int)d[k + 2]);
			if (a <= a0)
				continue;
			a = min(a, (int)d[k + 3]);
			a = min(a, (int)d[k + 4]);
			a = min(a, (int)d[k + 5]);
			a = min(a, (int)d[k + 6]);
			a0 = max(a0, min(a, (int)d[k]));
			a0 = max(a0, min(a, (int)d[k + 7]));
		}

		int b0 = -a0;
		for (k = 0; k < 12; k += 2)
		{
			int b = max((int)d[k + 1], (int)d[k + 2]);
			b = max(b, (int)d[k + 3]);
			b = max(b, (int)d[k + 4]);
			if (b >= b0)
				continue;
			b = max(b, (int)d[k + 5]);
			b = max(b, (int)d[k + 6]);

			b0 = min(b0, max(b, (int)d[k]));
			b0 = min(b0, max(b, (int)d[k + 7]));
		}

		threshold = -b0 - 1;
		return __int2float_rn(threshold);
	}

#elif (FAST_PATTERN == 8)

	__device__ float fastScore(const unsigned char* ptr, const int* pixel, int threshold)
	{
		int k, v = ptr[0];
		short d[N];
		for (k = 0; k < N; k++)
			d[k] = (short)(v - ptr[pixel[k]]);
		int a0 = threshold;
		for (k = 0; k < 8; k += 2)
		{
			int a = min((int)d[k + 1], (int)d[k + 2]);
			if (a <= a0)
				continue;
			a = min(a, (int)d[k + 3]);
			a = min(a, (int)d[k + 4]);
			a0 = max(a0, min(a, (int)d[k]));
			a0 = max(a0, min(a, (int)d[k + 5]));
		}

		int b0 = -a0;
		for (k = 0; k < 8; k += 2)
		{
			int b = max((int)d[k + 1], (int)d[k + 2]);
			b = max(b, (int)d[k + 3]);
			if (b >= b0)
				continue;
			b = max(b, (int)d[k + 4]);

			b0 = min(b0, max(b, (int)d[k]));
			b0 = min(b0, max(b, (int)d[k + 5]));
		}

		threshold = -b0 - 1;
		return __int2float_rn(threshold);
	}

#endif // FAST_PATTERN


	__device__ float harrisScore(const unsigned char* ptr, const int pitch)
	{
		int dx = 0, dy = 0, dxx = 0, dyy = 0, dxy = 0;
		const unsigned char* temp_ptr = ptr + (-HARRIS_SIZE / 2) * pitch + (-HARRIS_SIZE / 2);
		for (int i = 0; i < HARRIS_SIZE * HARRIS_SIZE; i++)
		{
			const unsigned char* curr_ptr = temp_ptr + ofs[i];
			dx = (__uchar2int(curr_ptr[1]) - __uchar2int(curr_ptr[-1])) * 2 + 
				(__uchar2int(curr_ptr[-pitch + 1]) - __uchar2int(curr_ptr[-pitch - 1])) + 
				(__uchar2int(curr_ptr[pitch + 1]) - __uchar2int(curr_ptr[pitch - 1]));
			dy = (__uchar2int(curr_ptr[pitch]) - __uchar2int(curr_ptr[-pitch])) * 2 + 
				(__uchar2int(curr_ptr[pitch - 1]) - __uchar2int(curr_ptr[-pitch - 1])) + 
				(__uchar2int(curr_ptr[pitch + 1]) - __uchar2int(curr_ptr[-pitch + 1]));
			dxx += dx * dx;
			dyy += dy * dy;
			dxy += dx * dy;
		}
		float fxx = __int2float_rn(dxx);
		float fyy = __int2float_rn(dyy);
		float fxy = __int2float_rn(dxy);
		float fsxy = fxx + fyy;
		return (fxx * fyy - fxy * fxy - HARRIS_K * fsxy * fsxy) * d_scale_sq_sq;
	}


	__global__ void fastDetect(unsigned char* image, OrbPoint* points, int3 iwhp, int3 owhp, int threshold, int border, int octave)
	{
		int ix = blockIdx.x * blockDim.x + threadIdx.x + border;
		int iy = blockIdx.y * blockDim.y + threadIdx.y + border;
		if (ix >= owhp.x - border || iy >= owhp.y - border)
		{
			return;
		}
		//int x0 = scale < 0 ? (ix >> -scale) : (ix << scale);
		//int y0 = scale < 0 ? (iy >> -scale) : (iy << scale);
		//if (x0 < border || y0 < border || x0 >= iwhp.x - border || y0 >= iwhp.y - border)
		//{
		//	return;
		//}

		int idx = iy * owhp.z + ix;
		const unsigned char* ptr = image + idx;
		const unsigned char* tab = dthresh_table + 255 - ptr[0];
		const int* odpixel = dpixel + 25 * octave;
		int d = tab[ptr[odpixel[0]]] | tab[ptr[odpixel[8]]];
		if (d == 0)
			return;

		d &= tab[ptr[odpixel[2]]] | tab[ptr[odpixel[10]]];
		d &= tab[ptr[odpixel[4]]] | tab[ptr[odpixel[12]]];
		d &= tab[ptr[odpixel[6]]] | tab[ptr[odpixel[14]]];
		if (d == 0)
			return;

		d &= tab[ptr[odpixel[1]]] | tab[ptr[odpixel[9]]];
		d &= tab[ptr[odpixel[3]]] | tab[ptr[odpixel[11]]];
		d &= tab[ptr[odpixel[5]]] | tab[ptr[odpixel[13]]];
		d &= tab[ptr[odpixel[7]]] | tab[ptr[odpixel[15]]];

		bool is_corner = false;
		if (d & 1)
		{
			int vt = ptr[0] - threshold, count = 0;
			for (int k = 0; k < N; k++)
			{
				if (ptr[odpixel[k]] < vt)
				{
					if (++count > K)
					{
						is_corner = true;
						break;
					}
				}
				else
				{
					count = 0;
				}
			}
		}
		else if (d & 2)
		{
			int vt = ptr[0] + threshold, count = 0;
			for (int k = 0; k < N; k++)
			{
				if (ptr[odpixel[k]] > vt)
				{
					if (++count > K)
					{
						is_corner = true;
						break;
					}
				}
				else
				{
					count = 0;
				}
			}
		}

		if (is_corner && d_point_counter < d_max_num_points)
		{
			unsigned int pi = atomicInc(&d_point_counter, 0x7fffffff);
			if (pi < d_max_num_points)
			{
				points[pi].x = ix;
				points[pi].y = iy;
				points[pi].octave = octave;
			}
		}
	}


	__global__ void fastDetectWithNMS(unsigned char* image, float* vmap, int3 owhp, int threshold, int octave, bool harris_score)
	{
		int ix = blockIdx.x * blockDim.x + threadIdx.x + 3;
		int iy = blockIdx.y * blockDim.y + threadIdx.y + 3;
		if (ix >= owhp.x - 3 || iy >= owhp.y - 3)
		{
			return;
		}

		// Detect corner
		int idx = iy * owhp.z + ix;
		const unsigned char* ptr = image + idx;
		const unsigned char* tab = dthresh_table + 255 - ptr[0];
		const int* odpixel = dpixel + 25 * octave;
		int d = tab[ptr[odpixel[0]]] | tab[ptr[odpixel[8]]];
		if (d == 0)
			return;

		d &= tab[ptr[odpixel[2]]] | tab[ptr[odpixel[10]]];
		d &= tab[ptr[odpixel[4]]] | tab[ptr[odpixel[12]]];
		d &= tab[ptr[odpixel[6]]] | tab[ptr[odpixel[14]]];
		if (d == 0)
			return;

		d &= tab[ptr[odpixel[1]]] | tab[ptr[odpixel[9]]];
		d &= tab[ptr[odpixel[3]]] | tab[ptr[odpixel[11]]];
		d &= tab[ptr[odpixel[5]]] | tab[ptr[odpixel[13]]];
		d &= tab[ptr[odpixel[7]]] | tab[ptr[odpixel[15]]];

		bool is_corner = false;
		if (d & 1)
		{
			int vt = ptr[0] - threshold, count = 0;
			for (int k = 0; k < N; k++)
			{
				if (ptr[odpixel[k]] < vt)
				{
					if (++count > K)
					{
						is_corner = true;
						break;
					}
				}
				else
				{
					count = 0;
				}
			}
		}
		else if (d & 2)
		{
			int vt = ptr[0] + threshold, count = 0;
			for (int k = 0; k < N; k++)
			{
				if (ptr[odpixel[k]] > vt)
				{
					if (++count > K)
					{
						is_corner = true;
						break;
					}
				}
				else
				{
					count = 0;
				}
			}
		}

		// Compute score
		if (is_corner)
		{
			//int x = scale < 0 ? (ix >> -scale) : (ix << scale);
			//int y = scale < 0 ? (iy >> -scale) : (iy << scale);
			vmap[idx] = harris_score ? harrisScore(ptr, owhp.z) : fastScore(ptr, odpixel, threshold);
			//vmap[idx] = max(vmap[idx], score);
			//printf("x=%d, y=%d, score=%f, vmap[vidx]=%f\n", x, y, score, vmap[vidx]);
			//atomicMax(&vmap[vidx], score);
		}
	}


	__global__ void nms(float* vmap, int3 vwhp, OrbPoint* points, int border, int octave)
	{
		int ix = blockIdx.x * blockDim.x + threadIdx.x + border;
		int iy = blockIdx.y * blockDim.y + threadIdx.y + border;
		if (ix >= vwhp.x - border || iy >= vwhp.y - border)
		{
			return;
		}

		int vidx = iy * vwhp.z + ix;
		if (d_point_counter < d_max_num_points && 
			vmap[vidx] > 0 && vmap[vidx] > vmap[vidx - 1] && vmap[vidx] > vmap[vidx + 1] &&
			vmap[vidx] > vmap[vidx - vwhp.z] && vmap[vidx] > vmap[vidx - vwhp.z - 1] && vmap[vidx] > vmap[vidx - vwhp.z + 1] &&
			vmap[vidx] > vmap[vidx + vwhp.z] && vmap[vidx] > vmap[vidx + vwhp.z - 1] && vmap[vidx] > vmap[vidx + vwhp.z + 1])
		{
			unsigned int pi = atomicInc(&d_point_counter, 0x7fffffff);
			if (pi < d_max_num_points)
			{
				points[pi].x = ix;
				points[pi].y = iy;
				points[pi].octave = octave;
				points[pi].score = vmap[vidx];
			}
		}
	}


	__global__ void bitonicSort(OrbPoint* points, int p, int n, bool descending)
	{
		unsigned int tid = threadIdx.x;

		// Padding
		for (int i = tid; i < (1 << p); i += blockDim.x)
		{
			if (i >= n)
			{
				points[i].score = descending ? -FLT_MAX : FLT_MAX;
			}
		}
		__syncthreads();

		// Sorting
		int pstride, pstride_half, ps, ps_half, s_half, i, j, k, n_half;
		bool orange;
		n_half = 1 << (p - 1);
		pstride_half = 0;
		for (pstride = 1; pstride <= p; pstride++)
		{
			ps = pstride;
			while (ps >= 1)
			{
				ps_half = ps - 1;
				s_half = 1 << ps_half;
				for (i = tid; i < n_half; i += blockDim.x)
				{
					orange = (i >> pstride_half) % 2 == 0;
					j = ((i >> ps_half) << ps) + (i % s_half);
					k = j + s_half;
					OrbPoint* p1 = &points[j];
					OrbPoint* p2 = &points[k];
					if ((descending && ((orange && p1->score < p2->score) || (!orange && p1->score > p2->score))) ||
						(!descending && ((orange && p1->score > p2->score) || (!orange && p1->score < p2->score))))
					{
						OrbPoint t = *p1;
						*p1 = *p2;
						*p2 = t;
					}
				}
				__syncthreads();
				ps = ps_half;
			}
			pstride_half++;
		}
	}


	inline __device__ float dFastAtan2(float y, float x)
	{
		const float absx = fabs(x);
		const float absy = fabs(y);
		const float a = __fdiv_rn(min(absx, absy), max(absx, absy));
		const float s = a * a;
		float r = __fmaf_rn(__fmaf_rn(__fmaf_rn(-0.0464964749f, s, 0.15931422f), s, -0.327622764f), s * a, a);
		//float r = ((-0.0464964749f * s + 0.15931422f) * s - 0.327622764f) * s * a + a;
		r = (absy > absx ? H_PI - r : r);
		r = (x < 0 ? M_PI - r : r);
		r = (y < 0 ? -r : r);
		return r;
	}


	__global__ void angleIC(unsigned char* image, OrbPoint* points, int3 iwhp, int half_k, int npts)
	{
		unsigned int pi = blockIdx.x * blockDim.x + threadIdx.x;
		if (pi < npts)
		{
			OrbPoint* p = &points[pi];
			const unsigned char* center = image + p->y * iwhp.z + p->x;

			int m_01 = 0, m_10 = 0;

			// Treat the center line differently, v=0
			for (int u = -half_k; u <= half_k; ++u)
				m_10 += u * center[u];

			// Go line by line in the circular patch
			int v_sum = 0, pofs = 0, val_plus = 0, val_minus = 0;
			for (int v = 1; v <= half_k; ++v)
			{
				// Proceed over the two lines
				v_sum = 0;
				pofs = v * iwhp.z;
				for (int u = -d_umax[v]; u <= d_umax[v]; ++u)
				{
					val_plus = center[u + pofs];
					val_minus = center[u - pofs];
					v_sum += (val_plus - val_minus);
					m_10 += u * (val_plus + val_minus);
				}
				m_01 += v * v_sum;
			}

			p->angle = dFastAtan2(__int2float_rn(m_01), __int2float_rn(m_10));
		}
	}


	__global__ void angleIC(unsigned char* images, OrbPoint* points, int half_k, int npts)
	{
		unsigned int pi = blockIdx.x * blockDim.x + threadIdx.x;
		if (pi < npts)
		{
			OrbPoint* p = &points[pi];
			const int pitch = angle_param[p->octave];
			const int moffset = angle_param[p->octave + MAX_OCTAVE];
			const unsigned char* center = images + moffset  + p->y * pitch + p->x;

			int m_01 = 0, m_10 = 0;

			// Treat the center line differently, v=0
			for (int u = -half_k; u <= half_k; ++u)
				m_10 += u * center[u];

			// Go line by line in the circular patch
			int v_sum = 0, pofs = 0, val_plus = 0, val_minus = 0;
			for (int v = 1; v <= half_k; ++v)
			{
				// Proceed over the two lines
				v_sum = 0;
				pofs = v * pitch;
				for (int u = -d_umax[v]; u <= d_umax[v]; ++u)
				{
					val_plus = center[u + pofs];
					val_minus = center[u - pofs];
					v_sum += (val_plus - val_minus);
					m_10 += u * (val_plus + val_minus);
				}
				m_01 += v * v_sum;
			}

			p->angle = dFastAtan2(__int2float_rn(m_01), __int2float_rn(m_10));
		}
	}


	__global__ void gaussFilter(unsigned char* src, unsigned char* dst, int3 whp)
	{
#define SMZ	(X2 + HGK + HGK)
		__shared__ float smem[SMZ][SMZ];
		
		int tix = threadIdx.x;
		int tiy = threadIdx.y;
		int ix = blockIdx.x * blockDim.x + tix + HGK;
		int iy = blockIdx.y * blockDim.y + tiy + HGK;
		if (ix >= whp.x - HGK || iy >= whp.y - HGK)
		{
			return;
		}

		// Store data to shared memory
		int rstart = iy * whp.z;
		int3 sx, sy;
		sx.x = tix; sx.y = sx.x + HGK; sx.z = sx.y + HGK;
		sy.x = tiy; sy.y = sy.x + HGK; sy.z = sy.y + HGK;
		bool xlc = tix < HGK;
		bool ytc = tiy < HGK;
		bool xrc = sx.z >= X2 || ix + HGK >= whp.x;
		bool ybc = sy.z >= X2 || iy + HGK >= whp.y;
		// Center
		smem[sy.y][sx.y] = __int2float_rn(__uchar2int(src[rstart + ix]));
		int left, top, right, bottom;
		if (xlc)	// Left
		{
			//left = ix < HGK ? HGK - ix : ix - HGK;
			left = ix - HGK;
			smem[sy.y][sx.x] = __int2float_rn(__uchar2int(src[rstart + left]));
		}
		if (ytc)	// Top
		{
			//top = iy < HGK ? HGK - iy : iy - HGK;
			top = iy - HGK;
			smem[sy.x][sx.y] = __int2float_rn(__uchar2int(src[top * whp.z + ix]));
		}
		if (xrc)	// Right
		{
			right = ix + HGK;
			//if (right >= whp.x)
			//	right = whp.x + whp.x - 2 - right;
			smem[sy.y][sx.z] = __int2float_rn(__uchar2int(src[rstart + right]));
		}
		if (ybc)	// Bottom
		{
			bottom = iy + HGK;
			//if (bottom >= whp.y)
			//	bottom = whp.y + whp.y - 2 - bottom;
			smem[sy.z][sx.y] = __int2float_rn(__uchar2int(src[bottom * whp.z + ix]));
		}
		if (xlc && ytc)	// Left top
		{
			smem[sy.x][sx.x] = __int2float_rn(__uchar2int(src[top * whp.z + left]));
		}
		if (xlc && ybc)	// Left bottom
		{
			smem[sy.z][sx.x] = __int2float_rn(__uchar2int(src[bottom * whp.z + left]));
		}
		if (xrc && ytc)	// Right top
		{
			smem[sy.x][sx.z] = __int2float_rn(__uchar2int(src[top * whp.z + right]));
		}
		if (xrc && ybc)	// Right bottom
		{
			smem[sy.z][sx.z] = __int2float_rn(__uchar2int(src[bottom * whp.z + right]));
		}
		__syncthreads();
		
		// Blurring by row
		smem[sy.y][sx.y] = d_gauss[3] * smem[sy.y][sx.y] +
			d_gauss[2] * (smem[sy.y][sx.y - 1] + smem[sy.y][sx.y + 1]) +
			d_gauss[1] * (smem[sy.y][sx.y - 2] + smem[sy.y][sx.y + 2]) +
			d_gauss[0] * (smem[sy.y][sx.y - 3] + smem[sy.y][sx.y + 3]);
		if (ytc)
		{
			smem[sy.x][sx.y] = d_gauss[3] * smem[sy.x][sx.y] +
				d_gauss[2] * (smem[sy.x][sx.y - 1] + smem[sy.x][sx.y + 1]) +
				d_gauss[1] * (smem[sy.x][sx.y - 2] + smem[sy.x][sx.y + 2]) +
				d_gauss[0] * (smem[sy.x][sx.y - 3] + smem[sy.x][sx.y + 3]);
		}
		else if (ybc)
		{
			smem[sy.z][sx.y] = d_gauss[3] * smem[sy.z][sx.y] +
				d_gauss[2] * (smem[sy.z][sx.y - 1] + smem[sy.z][sx.y + 1]) +
				d_gauss[1] * (smem[sy.z][sx.y - 2] + smem[sy.z][sx.y + 2]) +
				d_gauss[0] * (smem[sy.z][sx.y - 3] + smem[sy.z][sx.y + 3]);
		}
		__syncthreads();

		// Blurring by col
		dst[rstart + ix] = (unsigned char)__float2int_rn(d_gauss[3] * smem[sy.y][sx.y] +
			d_gauss[2] * (smem[sy.y - 1][sx.y] + smem[sy.y + 1][sx.y]) +
			d_gauss[1] * (smem[sy.y - 2][sx.y] + smem[sy.y + 2][sx.y]) +
			d_gauss[0] * (smem[sy.y - 3][sx.y] + smem[sy.y + 3][sx.y]));
	}


	__inline__ __device__ unsigned char getValue(const unsigned char* center, const int2* pattern, float sine, float cose, int pitch, int idx)
	{
		const int ix = __float2int_rn(pattern[idx].x * cose - pattern[idx].y * sine);
		const int iy = __float2int_rn(pattern[idx].x * sine + pattern[idx].y * cose);
		return *(center + iy * pitch + ix);
	}


	__device__ unsigned char feature2(const unsigned char* center, const int2* pattern, float sine, float cose, int pitch)
	{
		unsigned char t0, t1, val;
		t0 = getValue(center, pattern, sine, cose, pitch, 0); 
		t1 = getValue(center, pattern, sine, cose, pitch, 1);
		val = t0 < t1;
		t0 = getValue(center, pattern, sine, cose, pitch, 2);
		t1 = getValue(center, pattern, sine, cose, pitch, 3);
		val |= (t0 < t1) << 1;
		t0 = getValue(center, pattern, sine, cose, pitch, 4);
		t1 = getValue(center, pattern, sine, cose, pitch, 5);
		val |= (t0 < t1) << 2;
		t0 = getValue(center, pattern, sine, cose, pitch, 6);
		t1 = getValue(center, pattern, sine, cose, pitch, 7);
		val |= (t0 < t1) << 3;
		t0 = getValue(center, pattern, sine, cose, pitch, 8); 
		t1 = getValue(center, pattern, sine, cose, pitch, 9);
		val |= (t0 < t1) << 4;
		t0 = getValue(center, pattern, sine, cose, pitch, 10);
		t1 = getValue(center, pattern, sine, cose, pitch, 11);
		val |= (t0 < t1) << 5;
		t0 = getValue(center, pattern, sine, cose, pitch, 12);
		t1 = getValue(center, pattern, sine, cose, pitch, 13);
		val |= (t0 < t1) << 6;
		t0 = getValue(center, pattern, sine, cose, pitch, 14);
		t1 = getValue(center, pattern, sine, cose, pitch, 15);
		val |= (t0 < t1) << 7;
		return val;
	}


	__device__ unsigned char feature3(const unsigned char* center, const int2* pattern, float sine, float cose, int pitch)
	{
		unsigned char t0, t1, t2, val;
		t0 = getValue(center, pattern, sine, cose, pitch, 0);
		t1 = getValue(center, pattern, sine, cose, pitch, 1); 
		t2 = getValue(center, pattern, sine, cose, pitch, 2);
		val = t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0);

		t0 = getValue(center, pattern, sine, cose, pitch, 3); 
		t1 = getValue(center, pattern, sine, cose, pitch, 4);
		t2 = getValue(center, pattern, sine, cose, pitch, 5);
		val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 2;

		t0 = getValue(center, pattern, sine, cose, pitch, 6);
		t1 = getValue(center, pattern, sine, cose, pitch, 7);
		t2 = getValue(center, pattern, sine, cose, pitch, 8);
		val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 4;

		t0 = getValue(center, pattern, sine, cose, pitch, 8);
		t1 = getValue(center, pattern, sine, cose, pitch, 9);
		t2 = getValue(center, pattern, sine, cose, pitch, 11);
		val |= (t2 > t1 ? (t2 > t0 ? 2 : 0) : (t1 > t0)) << 6;
		return val;
	}


	__device__ unsigned char feature4(const unsigned char* center, const int2* pattern, float sine, float cose, int pitch)
	{
		unsigned char t0, t1, t2, t3, u, v, k, val;
		t0 = getValue(center, pattern, sine, cose, pitch, 0); 
		t1 = getValue(center, pattern, sine, cose, pitch, 1);
		t2 = getValue(center, pattern, sine, cose, pitch, 2); 
		t3 = getValue(center, pattern, sine, cose, pitch, 3);
		u = 0, v = 2;
		if (t1 > t0) t0 = t1, u = 1;
		if (t3 > t2) t2 = t3, v = 3;
		k = t0 > t2 ? u : v;
		val = k;

		t0 = getValue(center, pattern, sine, cose, pitch, 4);
		t1 = getValue(center, pattern, sine, cose, pitch, 5);
		t2 = getValue(center, pattern, sine, cose, pitch, 6);
		t3 = getValue(center, pattern, sine, cose, pitch, 7);
		u = 0, v = 2;
		if (t1 > t0) t0 = t1, u = 1;
		if (t3 > t2) t2 = t3, v = 3;
		k = t0 > t2 ? u : v;
		val |= k << 2;

		t0 = getValue(center, pattern, sine, cose, pitch, 8);
		t1 = getValue(center, pattern, sine, cose, pitch, 9);
		t2 = getValue(center, pattern, sine, cose, pitch, 10);
		t3 = getValue(center, pattern, sine, cose, pitch, 11);
		u = 0, v = 2;
		if (t1 > t0) t0 = t1, u = 1;
		if (t3 > t2) t2 = t3, v = 3;
		k = t0 > t2 ? u : v;
		val |= k << 4;

		t0 = getValue(center, pattern, sine, cose, pitch, 12); 
		t1 = getValue(center, pattern, sine, cose, pitch, 13);
		t2 = getValue(center, pattern, sine, cose, pitch, 14); 
		t3 = getValue(center, pattern, sine, cose, pitch, 15);
		u = 0, v = 2;
		if (t1 > t0) t0 = t1, u = 1;
		if (t3 > t2) t2 = t3, v = 3;
		k = t0 > t2 ? u : v;
		val |= k << 6;

		return val;
	}


	__global__ void describle(unsigned char* image, OrbPoint* points, int3 iwhp, int wta_k, int npts)
	{
		unsigned int pi = blockIdx.z;
		if (pi >= npts)
		{
			return;
		}

		OrbPoint* p = &points[pi];
		const float sine = __sinf(p->angle);
		const float cose = __cosf(p->angle);
		const unsigned char* center = image + p->y * iwhp.z + p->x;
		
		unsigned int tix = threadIdx.x;
		const int pstep = (wta_k == 2 || wta_k == 4) ? 16 : 12;
		const int2* pattern = d_pattern + pstep * tix;
		switch (wta_k)
		{
		case 2:
			p->features[tix] = feature2(center, pattern, sine, cose, iwhp.z);
			break;
		case 3:
			p->features[tix] = feature3(center, pattern, sine, cose, iwhp.z);
			break;
		case 4:
			p->features[tix] = feature4(center, pattern, sine, cose, iwhp.z);
			break;
		}
	}


	__global__ void describle(unsigned char* images, OrbPoint* points, int wta_k, int npts)
	{
		unsigned int pi = blockIdx.z;
		if (pi >= npts)
		{
			return;
		}

		OrbPoint* p = &points[pi];
		const float sine = __sinf(p->angle);
		const float cose = __cosf(p->angle);
		const int pitch = angle_param[p->octave];
		const int moffset = angle_param[p->octave + MAX_OCTAVE];
		const unsigned char* center = images + moffset + p->y * pitch + p->x;

		unsigned int tix = threadIdx.x;
		const int pstep = (wta_k == 2 || wta_k == 4) ? 16 : 12;
		const int2* pattern = d_pattern + pstep * tix;
		switch (wta_k)
		{
		case 2:
			p->features[tix] = feature2(center, pattern, sine, cose, pitch);
			break;
		case 3:
			p->features[tix] = feature3(center, pattern, sine, cose, pitch);
			break;
		case 4:
			p->features[tix] = feature4(center, pattern, sine, cose, pitch);
			break;
		}
	}


	__global__ void rescale(OrbPoint* points, int npts)
	{
		unsigned int pi = blockIdx.x * blockDim.x + threadIdx.x;
		if (pi >= npts)
		{
			return;
		}

		OrbPoint* p = &points[pi];
		const int scale = angle_param[p->octave];
		p->x = scale < 0 ? (p->x >> -scale) : (p->x << scale);
		p->y = scale < 0 ? (p->y >> -scale) : (p->y << scale);
	}


	inline __device__ int hammingDistance2(unsigned char* f1, unsigned char* f2)
	{
		int dist = 0;
		unsigned int* v1 = (unsigned int*)f1;
		unsigned int* v2 = (unsigned int*)f2;
		for (int i = 0; i < 8; i ++)
		{
			//dist += __popc(f1[i] ^ f2[i]) + __popc(f1[i + 1] ^ f2[i + 1]) + __popc(f1[i + 2] ^ f2[i + 2]) + __popc(f1[i + 3] ^ f2[i + 3]);
			dist += __popc(v1[i] ^ v2[i]);
		}
				
		// Even slower
		//for (int i = 0; i < 32; i += 4)
		//{
		//	dist += hamming_table[f1[i] ^ f2[i]] + hamming_table[f1[i + 1] ^ f2[i + 1]] + 
		//		hamming_table[f1[i + 2] ^ f2[i + 2]] + hamming_table[f1[i + 3] ^ f2[i + 3]];
		//}
		return dist;
	}


	__global__ void hammingMatch(OrbPoint* points1, OrbPoint* points2, int n1, int n2)
	{
		__shared__ unsigned char ofeat[32];
		__shared__ int distance[X2];
		__shared__ int indice[X2];
		__shared__ int flags[X2];

		unsigned int tid = threadIdx.x;
		unsigned int bid = blockIdx.x;
		if (bid >= n1)
		{
			return;
		}

		OrbPoint* p1 = &points1[bid];
		OrbPoint* p2 = &points2[tid];

		// Compute in base shared memory
		if (tid == 0)
		{
			for (int i = 0; i < 32; i++)
			{
				ofeat[i] = p1->features[i];
			}
		}
		__syncthreads();

		distance[tid] = hammingDistance2(ofeat, p2->features);
		indice[tid] = tid;
		__syncthreads();

		// Compute in template shared memory
		for (int pi = tid + X2; pi < n2; pi += X2)
		{
			// Compute hamming distance
			p2 = &points2[pi];
			int dist = hammingDistance2(ofeat, p2->features);
			if (dist < distance[tid])
			{
				distance[tid] = dist;
				indice[tid] = pi;
			}
			__syncthreads();
		}

		// Find minimum 
		for (int stride = X2 / 2; stride > 0; stride >>= 1)
		{
			int ntid = tid + stride;
			if (tid < stride && distance[ntid] < distance[tid])
			{
				int temp = distance[tid];
				distance[tid] = distance[ntid];
				distance[ntid] = temp;

				temp = indice[tid];
				indice[tid] = indice[ntid];
				indice[ntid] = temp;
			}
			__syncthreads();
		}

		// Flags for matching condition
		flags[tid] = distance[0] < distance[tid] ? 1 : 0;
		__syncthreads();

		// Sum over
		if (tid < 16)
		{
			volatile int* vsmem = flags;
			vsmem[tid] += vsmem[tid + 16];
			vsmem[tid] += vsmem[tid + 8];
			vsmem[tid] += vsmem[tid + 4];
			vsmem[tid] += vsmem[tid + 2];
			vsmem[tid] += vsmem[tid + 1];
		}

		// Get result
		if (tid == 0)
		{
			if (flags[0] == 31 && distance[0] < MAX_DIST)
			{
				p2 = &points2[indice[0]];
				p1->match = indice[0];	// indice[0];
				p1->distance = distance[0];
				p1->match_x = p2->x;
				p1->match_y = p2->y;
			}
			else
			{
				p1->match = -1;
				p1->distance = -1;
				p1->match_x = -1;
				p1->match_y = -1;
			}
		}
		__syncthreads();
	}




	void cuCreatePyramid(unsigned char* src, unsigned char* dst, int3 iwhp, int3* owhps, int* osizes, int* moffsets, int noctaves, bool doubled)
	{
		// Create pyramid
		dim3 block(X2, X2);
		dim3 grid;
		if (doubled)
		{
			// 0
			grid.x = (iwhp.x + X2 - 1) / X2;
			grid.y = (iwhp.y + X2 - 1) / X2;
			doubleImage << <grid, block >> > (src, dst, iwhp, owhps[0]);	// , 0, streams[0]

			// 1
			CHECK(cudaMemcpy(dst + moffsets[1], src, osizes[1] * sizeof(unsigned char), cudaMemcpyDeviceToDevice));

			// 2 -> 
			int factor = 2;
			for (int i = 2; i < noctaves; i++)
			{
				grid.x = (owhps[i].x + X2 - 1) / X2;
				grid.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid, block >> > (src, dst + moffsets[i], iwhp, owhps[i], factor);
				factor += factor;
			}
		}
		else
		{
			// 0
			CHECK(cudaMemcpy(dst, src, osizes[0] * sizeof(unsigned char), cudaMemcpyDeviceToDevice));

			// 1 ->
			int factor = 2;
			for (int i = 1; i < noctaves; i++)
			{
				grid.x = (owhps[i].x + X2 - 1) / X2;
				grid.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid, block >> > (src, dst + moffsets[i], iwhp, owhps[i], factor);
				factor += factor;
			}
		}

		CHECK(cudaDeviceSynchronize());

#if 0
		cv::Mat show;
		size_t spitch = 0, dpitch = 0;
		for (int i = 0; i < noctaves; i++)
		{
			show = cv::Mat(owhps[i].y, owhps[i].x, CV_8UC1);
			spitch = sizeof(unsigned char) * owhps[i].z;
			dpitch = sizeof(unsigned char) * owhps[i].x;
			CHECK(cudaMemcpy2D(show.data, dpitch, dst + moffsets[i], spitch, dpitch, owhps[i].y, cudaMemcpyDeviceToHost));
		}
#endif // 1


		CheckMsg("cuCreatePyramid() execution failed!\n");
	}


	void cuCreatePyramidAsync(unsigned char* src, unsigned char* dst, int3 iwhp, int3* owhps, int* osizes, int* moffsets, int noctaves, bool doubled)
	{
		// Create streams
		cudaStream_t* streams = (cudaStream_t*)malloc(noctaves * sizeof(cudaStream_t));
		for (int i = 0; i < noctaves; i++)
		{
			CHECK(cudaStreamCreate(&streams[i]));
		}

		// Create pyramid
		dim3 block(X2, X2);
		dim3 grid;
		if (doubled)
		{
			// 0
			grid.x = (iwhp.x + X2 - 1) / X2;
			grid.y = (iwhp.y + X2 - 1) / X2;
			doubleImage << <grid, block, 0, streams[0] >> > (src, dst, iwhp, owhps[0]);	// , 0, streams[0]

			// 1
			CHECK(cudaMemcpyAsync(dst + moffsets[1], src, osizes[1] * sizeof(unsigned char), cudaMemcpyDeviceToDevice, streams[1]));	// , streams[1]

			// 2 -> 
			int factor = 2;
			for (int i = 2; i < noctaves; i++)
			{
				grid.x = (owhps[i].x + X2 - 1) / X2;
				grid.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid, block, 0, streams[i] >> > (src, dst + moffsets[i], iwhp, owhps[i], factor);	// , 0, streams[i]
				factor += factor;
			}
		}
		else
		{
			// 0
			CHECK(cudaMemcpyAsync(dst, src, osizes[0] * sizeof(unsigned char), cudaMemcpyDeviceToDevice, streams[0]));	// , streams[0]
			
			// 1 ->
			int factor = 2;
			for (int i = 1; i < noctaves; i++)
			{
				grid.x = (owhps[i].x + X2 - 1) / X2;
				grid.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid, block, 0, streams[i] >> > (src, dst + moffsets[i], iwhp, owhps[i], factor);	// , 0, streams[i]
				factor += factor;
			}
		}		

		// release all stream
		CHECK(cudaDeviceSynchronize());
		for (int i = 0; i < noctaves; i++)
		{			
			CHECK(cudaStreamDestroy(streams[i]));
		}
		free(streams);

#if 0
		cv::Mat show;
		size_t spitch = 0, dpitch = 0;
		for (int i = 0; i < noctaves; i++)
		{
			show = cv::Mat(owhps[i].y, owhps[i].x, CV_8UC1);
			spitch = sizeof(unsigned char) * owhps[i].z;
			dpitch = sizeof(unsigned char) * owhps[i].x;
			CHECK(cudaMemcpy2D(show.data, dpitch, dst + moffsets[i], spitch, dpitch, owhps[i].y, cudaMemcpyDeviceToHost));
		}
#endif // 1


		CheckMsg("cuCreatePyramid() execution failed!\n");
	}


	void cuFastDectect(unsigned char* image, unsigned char* octave_images, OrbData& result, int3 iwhp, int3* owhps,
		int* osizes, int* moffsets, int noctaves, bool doubled, int threshold, int border)
	{
		if (border < 3)
			border = 3;

		// Create pyramid
		dim3 block(X2, X2);
		dim3 grid1;			// Grid for pyramid scale
		if (doubled)
		{
			// 0
			grid1.x = (iwhp.x + X2 - 1) / X2; 
			grid1.y = (iwhp.y + X2 - 1) / X2;
			doubleImage << <grid1, block >> > (image, octave_images, iwhp, owhps[0]);

			// 1
			CHECK(cudaMemcpy(octave_images + moffsets[1], image, osizes[1] * sizeof(unsigned char), cudaMemcpyDeviceToDevice));

			// 2 -> 
			int factor = 2;
			for (int i = 2; i < noctaves; i++)
			{
				grid1.x = (owhps[i].x + X2 - 1) / X2; 
				grid1.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid1, block >> > (image, octave_images + moffsets[i], iwhp, owhps[i], factor);
				factor += factor;
			}
		}
		else
		{
			// 0
			CHECK(cudaMemcpy(octave_images, image, osizes[0] * sizeof(unsigned char), cudaMemcpyDeviceToDevice));

			// 1 ->
			int factor = 2;
			for (int i = 1; i < noctaves; i++)
			{
				grid1.x = (owhps[i].x + X2 - 1) / X2; 
				grid1.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid1, block >> > (image, octave_images + moffsets[i], iwhp, owhps[i], factor);
				factor += factor;
			}
		}

#if 0
		cv::Mat show;
		size_t spitch = 0, dpitch = 0;
		for (int i = 0; i < noctaves; i++)
		{
			show = cv::Mat(owhps[i].y, owhps[i].x, CV_8UC1);
			spitch = sizeof(unsigned char) * owhps[i].z;
			dpitch = sizeof(unsigned char) * owhps[i].x;
			CHECK(cudaMemcpy2D(show.data, dpitch, octave_images + moffsets[i], spitch, dpitch, owhps[i].y, cudaMemcpyDeviceToHost));
		}
#endif // 1

		// Detect keypoints
		dim3 grid2;	// Grid for keypoints detection
		for (int i = 0; i < noctaves; i++)
		{
			grid2.x = (owhps[i].x - 6 + X2 - 1) / X2;
			grid2.y = (owhps[i].y - 6 + X2 - 1) / X2;
			fastDetect << <grid2, block >> > (octave_images + moffsets[i], result.d_data, iwhp, owhps[i], threshold, border, i);
		}

		CHECK(cudaDeviceSynchronize());
		CheckMsg("cuFastDectect() execution failed!\n");
	}


	void cuFastDectectAsync(unsigned char* image, unsigned char* octave_images, OrbData& result, int3 iwhp, int3* owhps,
		int* osizes, int* moffsets, int noctaves, bool doubled, int threshold, int border)
	{
		if (border < 3)
			border = 3;
		int total_border = border + border;

		// Create streams
		cudaStream_t* streams = (cudaStream_t*)malloc(noctaves * sizeof(cudaStream_t));
		for (int i = 0; i < noctaves; i++)
		{
			CHECK(cudaStreamCreate(&streams[i]));
		}

		// Create pyramid
		dim3 block(X2, X2);
		dim3 grid1;			// Grid for pyramid scale
		if (doubled)
		{
			// 0
			grid1.x = (iwhp.x + X2 - 1) / X2;
			grid1.y = (iwhp.y + X2 - 1) / X2;
			doubleImage << <grid1, block, 0, streams[0] >> > (image, octave_images, iwhp, owhps[0]);

			// 1
			CHECK(cudaMemcpyAsync(octave_images + moffsets[1], image, osizes[1] * sizeof(unsigned char), cudaMemcpyDeviceToDevice, streams[1]));

			// 2 -> 
			int factor = 2;
			for (int i = 2; i < noctaves; i++)
			{
				grid1.x = (owhps[i].x + X2 - 1) / X2;
				grid1.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid1, block, 0, streams[i] >> > (image, octave_images + moffsets[i], iwhp, owhps[i], factor);
				factor += factor;
			}
		}
		else
		{
			// 0
			CHECK(cudaMemcpyAsync(octave_images, image, osizes[0] * sizeof(unsigned char), cudaMemcpyDeviceToDevice, streams[0]));

			// 1 ->
			int factor = 2;
			for (int i = 1; i < noctaves; i++)
			{
				grid1.x = (owhps[i].x + X2 - 1) / X2;
				grid1.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid1, block, 0, streams[i] >> > (image, octave_images + moffsets[i], iwhp, owhps[i], factor);
				factor += factor;
			}
		}

#if 0
		cv::Mat show;
		size_t spitch = 0, dpitch = 0;
		for (int i = 0; i < noctaves; i++)
		{
			show = cv::Mat(owhps[i].y, owhps[i].x, CV_8UC1);
			spitch = sizeof(unsigned char) * owhps[i].z;
			dpitch = sizeof(unsigned char) * owhps[i].x;
			CHECK(cudaMemcpy2D(show.data, dpitch, octave_images + moffsets[i], spitch, dpitch, owhps[i].y, cudaMemcpyDeviceToHost));
		}
#endif // 1

		// Detect keypoints
		dim3 grid2;	// Grid for keypoints detection
		for (int i = 0; i < noctaves; i++)
		{
			grid2.x = (owhps[i].x - total_border + X2 - 1) / X2;
			grid2.y = (owhps[i].y - total_border + X2 - 1) / X2;
			fastDetect << <grid2, block, 0, streams[i] >> > (octave_images + moffsets[i], result.d_data, iwhp, owhps[i], threshold, border, i);
		}

		CHECK(cudaDeviceSynchronize());

		// release all stream
		CHECK(cudaDeviceSynchronize());
		for (int i = 0; i < noctaves; i++)
		{
			CHECK(cudaStreamDestroy(streams[i]));
		}
		free(streams);

		CheckMsg("cuFastDectectAsync() execution failed!\n");
	}


	void cuFastDectectWithNMS(unsigned char* image, unsigned char* octave_images, float* octave_vmaps, OrbData& result, int3 iwhp, int3* owhps, 
		int* osizes, int* moffsets, int noctaves, bool doubled, int threshold, int border, bool harris_score)
	{
		if (border < 3)
			border = 3;

		// Create pyramid
		dim3 block(X2, X2);
		dim3 grid1;			// Grid for pyramid scale
		if (doubled)
		{
			// 0
			grid1.x = (iwhp.x + X2 - 1) / X2;
			grid1.y = (iwhp.y + X2 - 1) / X2;
			doubleImage << <grid1, block >> > (image, octave_images, iwhp, owhps[0]);

			// 1
			CHECK(cudaMemcpy(octave_images + moffsets[1], image, osizes[1] * sizeof(unsigned char), cudaMemcpyDeviceToDevice));

			// 2 -> 
			int factor = 2;
			for (int i = 2; i < noctaves; i++)
			{
				grid1.x = (owhps[i].x + X2 - 1) / X2;
				grid1.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid1, block >> > (image, octave_images + moffsets[i], iwhp, owhps[i], factor);
				factor += factor;
			}
		}
		else
		{
			// 0
			CHECK(cudaMemcpy(octave_images, image, osizes[0] * sizeof(unsigned char), cudaMemcpyDeviceToDevice));

			// 1 ->
			int factor = 2;
			for (int i = 1; i < noctaves; i++)
			{
				grid1.x = (owhps[i].x + X2 - 1) / X2;
				grid1.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid1, block >> > (image, octave_images + moffsets[i], iwhp, owhps[i], factor);
				factor += factor;
			}
		}

#if 0
		cv::Mat show;
		size_t spitch = 0, dpitch = 0;
		for (int i = 0; i < noctaves; i++)
		{
			show = cv::Mat(owhps[i].y, owhps[i].x, CV_8UC1);
			spitch = sizeof(unsigned char) * owhps[i].z;
			dpitch = sizeof(unsigned char) * owhps[i].x;
			CHECK(cudaMemcpy2D(show.data, dpitch, octave_images + moffsets[i], spitch, dpitch, owhps[i].y, cudaMemcpyDeviceToHost));
		}
#endif // 1

		// Detect keypoints and get score map
		int total_border = border + border;
		dim3 grid2, grid3;	// Grid for keypoints detection
		for (int i = 0; i < noctaves; i++)
		{
			// Compute score map
			if (harris_score)
			{
				setHarrisOffsets(owhps[i].z);
			}
			grid2.x = (owhps[i].x - 6 + X2 - 1) / X2;
			grid2.y = (owhps[i].y - 6 + X2 - 1) / X2;
			fastDetectWithNMS << <grid2, block >> > (octave_images + moffsets[i], octave_vmaps + moffsets[i], owhps[i], threshold, i, harris_score);

			// NMS
			grid3.x = (owhps[i].x - total_border + X2 - 1) / X2;
			grid3.y = (owhps[i].y - total_border + X2 - 1) / X2;
			nms << <grid3, block >> > (octave_vmaps + moffsets[i], owhps[i], result.d_data, border, i);
		}

#if 0
		cv::Mat show(iwhp.y, iwhp.x, CV_32FC1);
		size_t spitch = sizeof(float) * iwhp.z;
		size_t dpitch = sizeof(float) * iwhp.x;
		CHECK(cudaMemcpy2D(show.data, dpitch, vmap, spitch, dpitch, iwhp.y, cudaMemcpyDeviceToHost));
#endif // 1

		//// NMS
		//dim3 grid3((iwhp.x - total_border + X2 - 1) / X2, (iwhp.y - total_border + X2 - 1) / X2);
		//nms << <grid3, block >> > (vmap, iwhp, result.d_data, border);

		CHECK(cudaDeviceSynchronize());
		CheckMsg("cuFastDectectWithNMS() execution failed!\n");
	}


	void cuFastDectectWithNMSAsync(unsigned char* image, unsigned char* octave_images, float* octave_vmaps, OrbData& result, int3 iwhp, int3* owhps,
		int* osizes, int* moffsets, int noctaves, bool doubled, int threshold, int border, bool harris_score)
	{
		if (border < 3)
			border = 3;

		// Create streams
		cudaStream_t* streams = (cudaStream_t*)malloc(noctaves * sizeof(cudaStream_t));
		for (int i = 0; i < noctaves; i++)
		{
			CHECK(cudaStreamCreate(&streams[i]));
		}

		// Create pyramid
		dim3 block(X2, X2);
		dim3 grid1;			// Grid for pyramid scale
		if (doubled)
		{
			// 0
			grid1.x = (iwhp.x + X2 - 1) / X2;
			grid1.y = (iwhp.y + X2 - 1) / X2;
			doubleImage << <grid1, block, 0, streams[0] >> > (image, octave_images, iwhp, owhps[0]);

			// 1
			CHECK(cudaMemcpyAsync(octave_images + moffsets[1], image, osizes[1] * sizeof(unsigned char), cudaMemcpyDeviceToDevice, streams[1]));

			// 2 -> 
			int factor = 2;
			for (int i = 2; i < noctaves; i++)
			{
				grid1.x = (owhps[i].x + X2 - 1) / X2;
				grid1.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid1, block, 0, streams[i] >> > (image, octave_images + moffsets[i], iwhp, owhps[i], factor);
				factor += factor;
			}
		}
		else
		{
			// 0
			CHECK(cudaMemcpyAsync(octave_images, image, osizes[0] * sizeof(unsigned char), cudaMemcpyDeviceToDevice, streams[0]));

			// 1 ->
			int factor = 2;
			for (int i = 1; i < noctaves; i++)
			{
				grid1.x = (owhps[i].x + X2 - 1) / X2;
				grid1.y = (owhps[i].y + X2 - 1) / X2;
				scaleDown << <grid1, block, 0, streams[i] >> > (image, octave_images + moffsets[i], iwhp, owhps[i], factor);
				factor += factor;
			}
		}

		// Detect keypoints and get score map
		int total_border = border + border;
		dim3 grid2, grid3;	// Grid for keypoints detection
		for (int i = 0; i < noctaves; i++)
		{
			if (harris_score)
			{
				setHarrisOffsets(owhps[i].z);
			}
			grid2.x = (owhps[i].x - 6 + X2 - 1) / X2;
			grid2.y = (owhps[i].y - 6 + X2 - 1) / X2;
			fastDetectWithNMS << <grid2, block, 0, streams[i] >> > (octave_images + moffsets[i], octave_vmaps + moffsets[i], owhps[i], threshold, i, harris_score);

			// NMS
			grid3.x = (owhps[i].x - total_border + X2 - 1) / X2;
			grid3.y = (owhps[i].y - total_border + X2 - 1) / X2;
			nms << <grid3, block, 0, streams[i] >> > (octave_vmaps + moffsets[i], owhps[i], result.d_data, border, i);

			CHECK(cudaDeviceSynchronize());
		}

		// release all stream
		CHECK(cudaDeviceSynchronize());
		for (int i = 0; i < noctaves; i++)
		{
			CHECK(cudaStreamDestroy(streams[i]));
		}
		free(streams);

		//// NMS
		//dim3 grid3((iwhp.x - total_border + X2 - 1) / X2, (iwhp.y - total_border + X2 - 1) / X2);
		//nms << <grid3, block >> > (vmap, iwhp, result.d_data, border);

		CHECK(cudaDeviceSynchronize());
		CheckMsg("cuFastDectectWithNMSAsync() execution failed!\n");
	}


	void cuRetainTopN(OrbData& result, const int n)
	{
		// Sort
		const int p = iExp2UpP(result.num_pts);
		bitonicSort << <1, X2 >> > (result.d_data, p, result.num_pts, true);
		result.num_pts = n;

		CHECK(cudaDeviceSynchronize());
		CheckMsg("cuRetainTopN() execution failed!\n");
	}


	void cuComputeAngle(unsigned char* image, OrbData& result, int3 iwhp, int patch_size)
	{
		dim3 block(X1);
		dim3 grid((result.num_pts + X1 - 1) / X1);
		angleIC << <grid, block >> > (image, result.d_data, iwhp, patch_size / 2, result.num_pts);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("cuComputeAngle() execution failed!\n");
	}


	void cuComputeAngle(unsigned char* octave_images, OrbData& result, int3* owhps, int* moffsets, int noctaves, int patch_size)
	{
		int aparams[MAX_OCTAVE * 2];
		for (int i = 0; i < noctaves; i++)
		{
			aparams[i] = owhps[i].z;
			aparams[i + MAX_OCTAVE] = moffsets[i];
		}
		setAngleParam(aparams);

		dim3 block(X1);
		dim3 grid((result.num_pts + X1 - 1) / X1);
		angleIC << <grid, block >> > (octave_images, result.d_data, patch_size / 2, result.num_pts);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("cuComputeAngle() execution failed!\n");
	}


	void cuGassianBlur(unsigned char* src, unsigned char* dst, int3 whp)
	{
		dim3 block(X2, X2);
		dim3 grid((whp.x - HGK - HGK + X2 - 1) / X2, (whp.y - HGK - HGK + X2 - 1) / X2);
		gaussFilter<<<grid, block>>>(src, dst, whp);
		CHECK(cudaDeviceSynchronize());

#if 0
		cv::Mat show0(whp.y, whp.x, CV_8UC1);
		cv::Mat show1(whp.y, whp.x, CV_8UC1);
		const size_t spitch = sizeof(unsigned char) * whp.z;
		const size_t dpitch = sizeof(unsigned char) * whp.x;
		CHECK(cudaMemcpy2D(show0.data, dpitch, src, spitch, dpitch, whp.y, cudaMemcpyDeviceToHost));
		CHECK(cudaMemcpy2D(show1.data, dpitch, dst, spitch, dpitch, whp.y, cudaMemcpyDeviceToHost));
#endif // SHOW

		CheckMsg("cuGassianBlur() execution failed!\n");
	}


	void cuGassianBlur(unsigned char* octave_images, int3* owhps, int* moffsets, int noctaves)
	{
		dim3 block(X2, X2), grid;
		for (int i = 0; i < noctaves; i++)
		{
			grid.x = (owhps[i].x - HGK - HGK + X2 - 1) / X2;
			grid.y = (owhps[i].y - HGK - HGK + X2 - 1) / X2;
			unsigned char* mem = octave_images + moffsets[i];
			gaussFilter << <grid, block >> > (mem, mem, owhps[i]);
		}
		CHECK(cudaDeviceSynchronize());

#if 0
		cv::Mat show;
		for (int i = 0; i < noctaves; i++)
		{
			show = cv::Mat(owhps[i].y, owhps[i].x, CV_8UC1);
			const size_t spitch = sizeof(unsigned char) * owhps[i].z;
			const size_t dpitch = sizeof(unsigned char) * owhps[i].x;
			unsigned char* mem = octave_images + moffsets[i];
			CHECK(cudaMemcpy2D(show.data, dpitch, mem, spitch, dpitch, owhps[i].y, cudaMemcpyDeviceToHost));
		}
#endif // SHOW

		CheckMsg("cuGassianBlur() execution failed!\n");
	}


	void cuDescribe(unsigned char* image, OrbData& result, int3 iwhp, int wta_k)
	{
		dim3 block(X2);
		dim3 grid(1, 1, result.num_pts);
		describle << <grid, block >> > (image, result.d_data, iwhp, wta_k, result.num_pts);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("cuDescribe() execution failed!\n");
	}


	void cuDescribe(unsigned char* octave_images, OrbData& result, int wta_k)
	{
		dim3 block(X2);
		dim3 grid(1, 1, result.num_pts);
		describle << <grid, block >> > (octave_images, result.d_data, wta_k, result.num_pts);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("cuDescribe() execution failed!\n");
	}


	void cuRescale(OrbData& result, int* scales, int noctaves)
	{
		CHECK(cudaMemcpyToSymbol(angle_param, scales, noctaves * sizeof(int), 0, cudaMemcpyHostToDevice));

		dim3 block(X1);
		dim3 grid((result.num_pts + X1 - 1) / X1);
		rescale << <grid, block >> > (result.d_data, result.num_pts);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("cuRescale() execution failed!\n");
	}


	void cuHammingMatch(OrbData& result1, OrbData& result2)
	{
		dim3 block(X2);
		dim3 grid(result1.num_pts);
		hammingMatch << <grid, block >> > (result1.d_data, result2.d_data, result1.num_pts, result2.num_pts);
		CHECK(cudaDeviceSynchronize());
		CheckMsg("cuHammingMatch() execution falied!\n");
	}


}