#include "orb.h"
#include "warmup.h"
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>


void cudaOrbDemo2(int argc, char** argv);
void cvOrbDemo2(int argc, char** argv);
void drawKeypoints(orb::OrbData& a, cv::Mat& img, cv::Mat& dst);
void drawMatches(orb::OrbData& a, orb::OrbData& b, cv::Mat& img1, cv::Mat& img2, cv::Mat& dst, const bool horizontal = true);
void drawMatches(std::vector<cv::KeyPoint>& a, std::vector<cv::KeyPoint>& b, std::vector<cv::DMatch>& matches, 
	cv::Mat& img1, cv::Mat& img2, cv::Mat& dst, const bool horizontal = true);


int main(int argc, char** argv)
{
	warmup();
    cudaOrbDemo2(argc, argv);
	cvOrbDemo2(argc, argv);

    CHECK(cudaDeviceReset());
    return 0;
}


void drawKeypoints(orb::OrbData& a, cv::Mat& img, cv::Mat& dst)
{
	orb::OrbPoint* data = a.h_data;
	cv::merge(std::vector<cv::Mat>{ img, img, img }, dst);
	for (int i = 0; i < a.num_pts; i++)
	{
		orb::OrbPoint& p = data[i];
		cv::Point center(cvRound(p.x), cvRound(p.y));
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
		cv::circle(dst, center, MAX(1, MIN(5, log10(p.score))), color);
	}
}


void drawMatches(orb::OrbData& a, orb::OrbData& b, cv::Mat& img1, cv::Mat& img2, cv::Mat& dst, const bool horizontal)
{
	int num_pts = a.num_pts;
	orb::OrbPoint* orb1 = a.h_data;
	orb::OrbPoint* orb2 = b.h_data;
	const int h1 = img1.rows;
	const int h2 = img2.rows;
	const int w1 = img1.cols;
	const int w2 = img2.cols;
	if (horizontal)
	{
		cv::Mat cat_img = cv::Mat::zeros(std::max<int>(h1, h2), w1 + w2, CV_32FC1);
		img1.copyTo(cat_img(cv::Rect(0, 0, w1, h1)));
		img2.copyTo(cat_img(cv::Rect(w1, 0, w2, h2)));
		cv::merge(std::vector<cv::Mat>{cat_img, cat_img, cat_img}, dst);
	}
	else
	{
		cv::Mat cat_img = cv::Mat::zeros(h1 + h2, std::max<int>(w1, w2), CV_32FC1);
		img1.copyTo(cat_img(cv::Rect(0, 0, w1, h1)));
		img2.copyTo(cat_img(cv::Rect(0, h1, w2, h2)));
		cv::merge(std::vector<cv::Mat>{cat_img, cat_img, cat_img}, dst);
	}
	dst.convertTo(dst, CV_8UC3);

	//// Compute min distance
	//int min_dist = (1 << 16), max_dist = 0;
	//for (int i = 0; i < num_pts; i++)
	//{
	//	if (orb1[i].distance > -1 && orb1[i].distance < min_dist)
	//	{
	//		min_dist = orb1[i].distance;
	//	}
	//	else if (orb1[i].distance > max_dist)
	//	{
	//		max_dist = orb1[i].distance;
	//	}
	//}
	//int threshold = (max_dist - min_dist) / 3;

	// Filter by distance
	for (int i = 0; i < num_pts; i++)
	{
		int k = orb1[i].match;
		//int d = orb1[i].distance;
		if (k != -1)	//  && d - min_dist < threshold
		{
			cv::Point p1(cvRound(orb1[i].x), cvRound(orb1[i].y));
			cv::Point p2(cvRound(orb2[k].x), cvRound(orb2[k].y));
			if (horizontal)
				p2.x += w1;
			else
				p2.y += h1;
			cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
			cv::line(dst, p1, p2, color);
		}
	}
}


void drawMatches(std::vector<cv::KeyPoint>& a, std::vector<cv::KeyPoint>& b, std::vector<cv::DMatch>& matches,
	cv::Mat& img1, cv::Mat& img2, cv::Mat& dst, const bool horizontal)
{
	const int h1 = img1.rows;
	const int h2 = img2.rows;
	const int w1 = img1.cols;
	const int w2 = img2.cols;
	if (horizontal)
	{
		cv::Mat cat_img = cv::Mat::zeros(std::max<int>(h1, h2), w1 + w2, CV_32FC1);
		img1.copyTo(cat_img(cv::Rect(0, 0, w1, h1)));
		img2.copyTo(cat_img(cv::Rect(w1, 0, w2, h2)));
		cv::merge(std::vector<cv::Mat>{cat_img, cat_img, cat_img}, dst);
	}
	else
	{
		cv::Mat cat_img = cv::Mat::zeros(h1 + h2, std::max<int>(w1, w2), CV_32FC1);
		img1.copyTo(cat_img(cv::Rect(0, 0, w1, h1)));
		img2.copyTo(cat_img(cv::Rect(0, h1, w2, h2)));
		cv::merge(std::vector<cv::Mat>{cat_img, cat_img, cat_img}, dst);
	}
	dst.convertTo(dst, CV_8UC3);

	for (size_t i = 0; i < matches.size(); i++)
	{
		cv::Point p1, p2;
		p1.x = cvRound(a[matches[i].queryIdx].pt.x);
		p1.y = cvRound(a[matches[i].queryIdx].pt.y);
		p2.x = cvRound(b[matches[i].trainIdx].pt.x);
		p2.y = cvRound(b[matches[i].trainIdx].pt.y);
		if (horizontal)
			p2.x += w1;
		else
			p2.y += h1;
		cv::Scalar color(rand() % 255, rand() % 255, rand() % 255);
		cv::line(dst, p1, p2, color);
	}
}


void cudaOrbDemo2(int argc, char** argv)
{
	std::cout << "===== Registration by CUDA-ORB =====" << std::endl;

	int devNum = 0;
	if (argc > 1)
		devNum = std::atoi(argv[1]);
	std::string save_dir = "";

	// Read images using OpenCV
	cv::Mat limg, rimg;
	if (argc > 2)
		limg = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
	if (argc > 3)
		rimg = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
	if (argc > 4)
		save_dir = argv[4];
	if (limg.empty() || rimg.empty())
	{
		std::cout << "Please input as follow: " << std::endl
			<< "\t1. DeviceNumber" << std::endl
			<< "\t2. Path to template image" << std::endl
			<< "\t3. Path to target image" << std::endl
			<< "\t4. Directory to save results" << std::endl;
		return;
	}		

	// Initial Cuda images and download images to device
	std::cout << "Initializing data..." << std::endl;
	initDevice(devNum);

	unsigned int w = limg.cols;
	unsigned int h = limg.rows;
	std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

	// The number of pyramid levels.
	int noctaves = 5;
	// This is size of the border where the features are not detected.
	int edge_threshold = 31;
	// Zoom in image to double size
	//bool doubled = false;
	// The number of points that produce each element of the oriented BRIEF descriptor.
	int wta_k = 4;
	// The default HARRIS_SCORE means that Harris algorithm is used to rank features
	// (the score is written to KeyPoint::scoreand is used to retain best nfeatures features);
	// FAST_SCORE is alternative value of the parameter that produces slightly less stable keypoints,
	// but it is a little faster to compute.
	orb::ScoreType score_type = orb::ScoreType::HARRIS_SCORE;
	// The size of the patch used by the oriented BRIEF descriptor.
	int patch_size = 31;
	// The fast threshold
	int fast_threshold = 20;
	// The max number of keypoints allowed
	int max_pts = 10000;
	// The top-n best keypoints retained for matching. If equal to -1, retain all
	int retain_topn = 0;
	//// NMS ? 
	//bool nonmax_suppression = true;

	GpuTimer timer(0);
	int3 whp1, whp2;
	whp1.x = limg.cols; whp1.y = limg.rows; whp1.z = iAlignUp(whp1.x, 128);
	whp2.x = rimg.cols; whp2.y = rimg.rows; whp2.z = iAlignUp(whp2.x, 128);
	size_t size1 = whp1.y * whp1.z * sizeof(uchar);
	size_t size2 = whp2.y * whp2.z * sizeof(uchar);
	uchar* img1 = NULL;
	uchar* img2 = NULL;
	size_t tmp_pitch = 0;
	CHECK(cudaMallocPitch((void**)&img1, &tmp_pitch, sizeof(uchar) * whp1.x, whp1.y));
	CHECK(cudaMallocPitch((void**)&img2, &tmp_pitch, sizeof(uchar) * whp2.x, whp2.y));
	const size_t dpitch1 = sizeof(uchar) * whp1.z;
	const size_t spitch1 = sizeof(uchar) * whp1.x;
	const size_t dpitch2 = sizeof(uchar) * whp2.z;
	const size_t spitch2 = sizeof(uchar) * whp2.x;
	CHECK(cudaMemcpy2D(img1, dpitch1, limg.data, spitch1, spitch1, whp1.y, cudaMemcpyHostToDevice));
	CHECK(cudaMemcpy2D(img2, dpitch2, rimg.data, spitch2, spitch2, whp2.y, cudaMemcpyHostToDevice));
	float t0 = timer.read();

	std::unique_ptr<orb::Orbor> detector(new orb::Orbor);
	detector->init(noctaves, edge_threshold, wta_k, score_type, patch_size, fast_threshold, retain_topn, max_pts);

	// Initialize ORB keypoints
	orb::OrbData orb_data1, orb_data2;
	detector->initOrbData(orb_data1, max_pts, true, true);
	detector->initOrbData(orb_data2, max_pts, true, true);

	// Initialize ORB descriptors
	unsigned char* orb_desc1 = NULL;
	unsigned char* orb_desc2 = NULL;

	int nrepeats = 100;
	float t1 = timer.read();
	for (int i = 0; i < nrepeats; i++)
	{
		detector->detectAndCompute(img1, orb_data1, whp1, (void**)&orb_desc1, true);
		detector->detectAndCompute(img2, orb_data2, whp2, (void**)&orb_desc2, true);
	}

	float t2 = timer.read();
	for (int i = 0; i < nrepeats; i++)
	{
		detector->match(orb_data1, orb_data2, orb_desc1, orb_desc2);
	}

	float t3 = timer.read();
	std::cout << "Number of features1: " << orb_data1.num_pts << std::endl
		<< "Number of features2: " << orb_data2.num_pts << std::endl;
	std::cout << "Time for allocating image memory:  " << t0 << std::endl
		<< "Time for allocating point memory:  " << t1 - t0 << std::endl
		<< "Time of detection and computation: " << (t2 - t1) / nrepeats << std::endl
		<< "Time of matching ORB keypoints:   " << (t3 - t2) / nrepeats << std::endl;

	// Show
	cv::Mat show1, show2, show_matched;
	drawKeypoints(orb_data1, limg, show1);
	drawKeypoints(orb_data2, rimg, show2);
	drawMatches(orb_data1, orb_data2, limg, rimg, show_matched, false);
	cv::imwrite(save_dir + "/orb_show1.jpg", show1);
	cv::imwrite(save_dir + "/orb_show2.jpg", show2);
	cv::imwrite(save_dir + "/orb_show_matched.jpg", show_matched);

	// Free ORB data from device
	detector->freeOrbData(orb_data1);
	detector->freeOrbData(orb_data2);
	if (orb_desc1)
		CHECK(cudaFree(orb_desc1));
	if (orb_desc2)
		CHECK(cudaFree(orb_desc2));
	CHECK(cudaFree(img1));
	CHECK(cudaFree(img2));
}


void cvOrbDemo2(int argc, char** argv)
{
	std::cout << "===== Registration by OpenCV =====" << std::endl;
	int devNum = 0;
	if (argc > 1)
		devNum = std::atoi(argv[1]);
	std::string save_dir = "";

	// Read images using OpenCV
	cv::Mat limg, rimg;
	if (argc > 2)
		limg = cv::imread(argv[2], cv::IMREAD_GRAYSCALE);
	if (argc > 3)
		rimg = cv::imread(argv[3], cv::IMREAD_GRAYSCALE);
	if (argc > 4)
		save_dir = argv[4];
	if (limg.empty() || rimg.empty())
	{
		std::cout << "Please input as follow: " << std::endl
			<< "\t1. DeviceNumber" << std::endl
			<< "\t2. Path to template image" << std::endl
			<< "\t3. Path to target image" << std::endl
			<< "\t4. Directory to save results" << std::endl;
		return;
	}		

	// Initial Cuda images and download images to device
	std::cout << "Initializing data..." << std::endl;
	initDevice(devNum);
	unsigned int w = limg.cols;
	unsigned int h = limg.rows;
	std::cout << "Image size = (" << w << "," << h << ")" << std::endl;

	int nrepeats = 100;
	cv::Mat desc_r, desc_t;
	std::vector<cv::KeyPoint> kpt_r, kpt_t;
	cv::Ptr<cv::ORB> detector = cv::ORB::create(3072);
	std::vector<cv::DMatch> imatches;
	cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

	GpuTimer timer(0);
	for (int i = 0; i < nrepeats; i++)
	{
		detector->detectAndCompute(limg, cv::Mat(), kpt_r, desc_r);
		detector->detectAndCompute(rimg, cv::Mat(), kpt_t, desc_t);
	}

	float t0 = timer.read();
	for (int i = 0; i < nrepeats; i++)
	{
		matcher->match(desc_r, desc_t, imatches);
	}

	float t1 = timer.read();
	cv::Mat show_matched;
	drawMatches(kpt_r, kpt_t, imatches, limg, rimg, show_matched, false);
	cv::imwrite(save_dir + "/cvshow_matched.jpg", show_matched);

	std::cout << "Number of original features: " << kpt_t.size() << " " << kpt_r.size() << std::endl;
	std::cout << "Number of matching features: " << imatches.size() << std::endl;
	std::cout << "Time of detection and computation: " << t0 / nrepeats << std::endl
		<< "Time of matching orb keypoints:   " << (t1 - t0) / nrepeats << std::endl;
}
