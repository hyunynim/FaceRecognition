#define DLIB_JPEG_SUPPORT
#define DLIB_USE_CUDA
#define DEBUG
#include <dlib/dnn.h>
#include <dlib/gui_widgets.h>
#include <dlib/clustering.h>
#include <dlib/string.h>
#include <dlib/image_io.h>
#include <dlib/image_processing/frontal_face_detector.h>
#include<bits/stdc++.h>
#include <dlib/svm.h>
#include <dlib/svm_threaded.h>

using namespace std;
using namespace dlib;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual = add_prev1<block<N, BN, 1, tag1<SUBNET>>>;

template <template <int, template<typename>class, int, typename> class block, int N, template<typename>class BN, typename SUBNET>
using residual_down = add_prev2<avg_pool<2, 2, 2, 2, skip1<tag2<block<N, BN, 2, tag1<SUBNET>>>>>>;

template <int N, template <typename> class BN, int stride, typename SUBNET>
using block  = BN<con<N, 3, 3, 1, 1, relu<BN<con<N, 3, 3, stride, stride, SUBNET>>>>>;

template <int N, typename SUBNET> using ares      = relu<residual<block, N, affine, SUBNET>>;
template <int N, typename SUBNET> using ares_down = relu<residual_down<block, N, affine, SUBNET>>;

template <typename SUBNET> using alevel0 = ares_down<256, SUBNET>;
template <typename SUBNET> using alevel1 = ares<256, ares<256, ares_down<256, SUBNET>>>;
template <typename SUBNET> using alevel2 = ares<128, ares<128, ares_down<128, SUBNET>>>;
template <typename SUBNET> using alevel3 = ares<64, ares<64, ares<64, ares_down<64, SUBNET>>>>;
template <typename SUBNET> using alevel4 = ares<32, ares<32, ares<32, SUBNET>>>;

using anet_type = loss_metric<fc_no_bias<128, avg_pool_everything<
	alevel0<
	alevel1<
	alevel2<
	alevel3<
	alevel4<
	max_pool<3, 3, 2, 2, relu<affine<con<32, 7, 7, 2, 2,
	input_rgb_image_sized<150>
	>>>>>>>>>>>>;

char msg[1010];
string rootDir = "..\\";
typedef long long ll;

typedef matrix<float, 128, 1> sampleType;

typedef radial_basis_kernel<sampleType> kernelType;
typedef linear_kernel<sampleType> lineearKernel;

frontal_face_detector detector = get_frontal_face_detector();

shape_predictor sp;
anet_type net;
multiclass_linear_decision_function<lineearKernel, string> df;

std::vector<matrix<rgb_pixel>> JitterImage(const matrix<rgb_pixel>& img) {
	// All this function does is make 100 copies of img, all slightly jittered by being
	// zoomed, rotated, and translated a little bit differently. They are also randomly
	// mirrored left to right.
	thread_local dlib::rand rnd;
	const int jiiterNum = 100;

	std::vector<matrix<rgb_pixel>> crops;
	for (int i = 0; i < jiiterNum; ++i)
		crops.push_back(::jitter_image(img, rnd));
	return crops;
}

void PrintLog(char * msg) {
#ifdef DEBUG
	puts(msg);
#endif
}
void PrintLog(const char* msg) {
#ifdef DEBUG
	puts(msg);
#endif
}
void MakeFileList(string dir) {
	FILE* fp = fopen((rootDir + "list.txt").c_str(), "w");
	fclose(fp);
	sprintf(msg, "dir %s /b >> %s\\list.txt", dir.c_str(), rootDir.c_str());
	system(msg);
}
std::vector<matrix<rgb_pixel>> LoadImages(std::vector<std::string> & labels, std::string imgDir) {
	MakeFileList(rootDir + imgDir);

	FILE* fp = fopen((rootDir + "list.txt").c_str(), "r");
	std::vector<matrix<rgb_pixel>> imgs;
	matrix<rgb_pixel> img;
	std::vector<std::string> exp = { ".jpg", ".jpeg", ".png" };
	char cTmp;
	while (~fscanf(fp, "%[^\n]", msg)) {
		fscanf(fp, "%c", &cTmp);
		string strTmp = msg;
		for (int i = 0; i < strTmp.size(); ++i)
			if ('A' <= strTmp[i] && strTmp[i] <= 'Z')
				strTmp[i] += 'a' - 'A';
		ll pos = std::string::npos;
		for (int i = 0; i < exp.size(); ++i) {
			pos = strTmp.find(exp[i]);
			if (pos != std::string::npos)
				break;
		}
		if (pos == std::string::npos) continue;
		pos = strTmp.find('\n');
		if (pos != std::string::npos)
			strTmp = strTmp.substr(0, pos);
		PrintLog(strTmp.c_str());
		load_image(img, (rootDir + imgDir + strTmp).c_str());
		imgs.push_back(img);
		
		auto crop = strTmp.find('.');
		if (crop != std::string::npos)
			strTmp = strTmp.substr(0, crop);
		labels.push_back(strTmp);
	}
	fclose(fp);
	return imgs;
}

void TestModel() {
	std::vector<sampleType> samples;
	std::vector<string> labels;;

	PrintLog("Test start...");

	PrintLog("Load test images");
	std::vector<matrix<rgb_pixel>> anne = LoadImages(labels, "test\\anne_hathaway\\");
	std::vector<matrix<rgb_pixel>> hong = LoadImages(labels, "test\\honghyun\\");
	std::vector<matrix<rgb_pixel>> jung = LoadImages(labels, "test\\jungjae_lee\\");
	std::vector<matrix<rgb_pixel>> margot = LoadImages(labels, "test\\margot_robbie\\");
	std::vector<matrix<rgb_pixel>> sheldon = LoadImages(labels, "test\\sheldon_cooper\\");
	std::vector<matrix<rgb_pixel>> unknown = LoadImages(labels, "test\\unknown\\");
	std::vector<matrix<rgb_pixel>> imgs;
	PrintLog("Done...\n");

	labels.clear();

	PrintLog("Make labels");
	for (int i = 0; i < anne.size(); ++i) {
		imgs.push_back(anne[i]);
		labels.push_back("anne hathaway");
	}
	for (int i = 0; i < hong.size(); ++i) {
		imgs.push_back(hong[i]);
		labels.push_back("honghyun ahn");
	}
	for (int i = 0; i < jung.size(); ++i) {
		imgs.push_back(jung[i]);
		labels.push_back("jungjae lee");
	}
	for (int i = 0; i < margot.size(); ++i) {
		imgs.push_back(margot[i]);
		labels.push_back("margot robbie");
	}
	for (int i = 0; i < sheldon.size(); ++i) {
		imgs.push_back(sheldon[i]);
		labels.push_back("sheldon cooper");
	}
	for (int i = 0; i < unknown.size(); ++i) {
		imgs.push_back(unknown[i]);
		labels.push_back("unknown");
	}
	PrintLog("Done...\n");

	PrintLog("Crop faces");
	std::vector<matrix<rgb_pixel>> faces;
	for (auto img : imgs) {
		for (auto face : detector(img))
		{
			auto shape = sp(img, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
		}
	}
	PrintLog("Done...\n");

	PrintLog("Make faces to feature");
	std::vector<matrix<float, 0, 1>> fd = net(faces);
	PrintLog("Done...");

	PrintLog("Start prediction");
	int ans = 0;
	for (int i = 0; i < fd.size(); ++i) {
		sampleType sample;
		int j = 0;
		for (auto it = fd[i].begin(); it != fd[i].end(); ++it) {
			sample(j++) = *it;
		}
		auto res = df.predict(sample);
		printf("Pred(%d): %s(%f)\n", i + 1, res.first, res.second);
		if (res.first == labels[i]) ++ans;
	}
	printf("pass: %d, fail: %d\n", ans, fd.size() - ans);
}
int main() {
	string imgDir = "img\\";
	std::vector<sampleType> samples;
	std::vector<string> labels;

	clock_t est = clock(), eed;
	clock_t st, ed;

	PrintLog("Model Loading...");

	st = clock();
	deserialize(rootDir + "model\\shape_predictor_5_face_landmarks.dat") >> sp;

	deserialize(rootDir + "model\\dlib_face_recognition_resnet_model_v1.dat") >> net;
	ed = clock();
	sprintf(msg, "Done(%dms)\n", ed - st);
	PrintLog(msg);

	sprintf(msg, "Image loading...");
	PrintLog(msg);

	st = clock();
	std::vector<matrix<rgb_pixel>> imgs = LoadImages(labels, imgDir);

	ed = clock();
	sprintf(msg, "Done(%dms)\n", ed - st);
	PrintLog(msg);


	sprintf(msg, "Extract faces in image...");
	PrintLog(msg);

	st = clock();
	std::vector<matrix<rgb_pixel>> faces;
	for (auto img : imgs) {
		for (auto face : detector(img))
		{
			auto shape = sp(img, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(img, get_face_chip_details(shape, 150, 0.25), face_chip);
			faces.push_back(move(face_chip));
		}
	}
	ed = clock();
	sprintf(msg, "Done(%dms)\n", ed - st);
	PrintLog(msg);

	if (faces.size() == 0)
	{
		cout << "No faces found in image!" << endl;
		return 1;
	}
	sprintf(msg, "\n[Result]: %d faces found in image", faces.size());
	PrintLog(msg);

	sprintf(msg, "Make face to vector...");
	PrintLog(msg);
	st = clock();
	std::vector<matrix<float, 0, 1>> faceDescriptors = net(faces);
	ed = clock();

	sprintf(msg, "Done(%dms)\n", ed - st);
	PrintLog(msg);

	sprintf(msg, "Make trainer");
	PrintLog(msg);
	svm_multiclass_linear_trainer<lineearKernel, string> trainer;

	sprintf(msg, "[DEBUG]: Set vector");
	PrintLog(msg);

	sampleType sample;
	sprintf(msg, "[DEBUG]: Make samples");
	PrintLog(msg);
	for (int i = 0; i < faceDescriptors.size(); ++i) {
		int j = 0;

		sprintf(msg, "[DEBUG]: Allocating");
		PrintLog(msg);

		for (auto it = faceDescriptors[i].begin(); it != faceDescriptors[i].end(); ++it) 
			sample(j++) = *it;
		
		sprintf(msg, "[DEBUG]: Done");
		PrintLog(msg);

		samples.push_back(sample);
	}
	trainer.set_max_iterations(1000000);
	trainer.set_num_threads(10);
	trainer.set_epsilon(1e-4);
	trainer.set_c(1);
	trainer.set_learns_nonnegative_weights(true);

	trainer.be_verbose();

	df = trainer.train(samples, labels);

	sprintf(msg, "%d class in data", df.number_of_classes());
	PrintLog(msg);

	TestModel();
	serialize(rootDir + "model\\pretrained.dat") << df;
}