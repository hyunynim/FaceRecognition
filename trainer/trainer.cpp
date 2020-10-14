#define DLIB_JPEG_SUPPORT
#define DLIB_USE_CUDA
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

int main() {

	clock_t est = clock(), eed;
	clock_t st, ed;
	printf("Model Loading...\n");
	st = clock();
	frontal_face_detector detector = get_frontal_face_detector();

	shape_predictor sp;
	deserialize("../shape_predictor_5_face_landmarks.dat") >> sp;

	anet_type net;
	deserialize("../dlib_face_recognition_resnet_model_v1.dat") >> net;
	ed = clock();
	printf("Done(%dms)\n\n", ed - st);

	printf("Image loading...\n");
	st = clock();

	std::vector<matrix<rgb_pixel>> imgs;

	matrix<rgb_pixel> tmp1, tmp2, tmp3, tmp4;
	load_image(tmp1, "../1.jpg");
	imgs.push_back(tmp1);

	load_image(tmp2, "../2.jpg");
	imgs.push_back(tmp2);

	load_image(tmp3, "../3.jpg");
	imgs.push_back(tmp3);

	load_image(tmp4, "../test.jpg");
	imgs.push_back(tmp4);

	ed = clock();
	printf("Done(%dms)\n\n", ed - st);


	printf("Extract faces in image...\n");
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
	printf("Done(%dms)\n\n", ed - st);

	if (faces.size() == 0)
	{
		cout << "No faces found in image!" << endl;
		return 1;
	}
	cout << "\n[Result]: " << faces.size() << "faces found in image" << "\n\n";
	std::vector<matrix<rgb_pixel>> tmp = { faces[0], faces[1], faces[2] };
	auto cTmp = tmp;
	tmp.clear();
	for (int i = 0; i < 3; ++i) {
		auto res = JitterImage(cTmp[i]);
		for (auto img : res)
			tmp.push_back(img);
	}
	printf("Make face to vector...\n");
	st = clock();
	std::vector<matrix<float, 0, 1>> face_descriptors = net(faces);
	std::vector<matrix<float, 0, 1>> fd2 = net(tmp);
	ed = clock();
	printf("Done(%dms)\n\n", ed - st);
	printf("%d %d\n", face_descriptors[0].nc(), face_descriptors[0].nr());
	typedef matrix<float, 128, 1> sample_type;


	typedef radial_basis_kernel<sample_type> kernel_type;
	typedef linear_kernel<sample_type> lineearKernel;


	puts("Make trainer");
	svm_pegasos<kernel_type> trainer;

	svm_multiclass_linear_trainer<lineearKernel, string> trainer2;

	puts("[DEBUG]: Set vector");
	std::vector<sample_type> samples;
	std::vector<string> labels;

	sample_type sample;
	puts("[DEBUG]: Make samples");
	printf("\t %d \t\n", fd2.size());
	for (int i = 0; i < fd2.size(); ++i) {
		int j = 0;
		puts("[DEBUG]: Allocating");
		for (auto it = fd2[i].begin(); it != fd2[i].end(); ++it) {
			sample(j++) = *it;
		}
		puts("[DEBUG]: Done");
		samples.push_back(sample);
		if (i / 100 == 0)
			labels.push_back("honghyun");
		else if (i / 100 == 1)
			labels.push_back("cooper");
		else if (i / 100 == 2)
			labels.push_back("¸ô¶ó ´©±ºÁö");
	}
	trainer2.set_max_iterations(1000000);
	trainer2.set_num_threads(10);
	trainer2.set_epsilon(0.00001);
	trainer2.set_c(1);
	trainer2.set_learns_nonnegative_weights(false);

	trainer2.be_verbose();

	sample_type s2;
	int j = 0;
	for (auto it = face_descriptors[3].begin(); it != face_descriptors[3].end(); ++it) {
		s2(j++) = *it;
	}
	multiclass_linear_decision_function<lineearKernel, string> df = trainer2.train(samples, labels);

	auto res = df.predict(s2);
	serialize("pretrained.dat") << df;
	cout << res.first << ' ' << res.second;	//threshold is about 0.1

}