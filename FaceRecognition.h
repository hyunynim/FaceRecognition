#pragma once
typedef matrix<float, 0, 1> faceFeature;
typedef matrix<float, 128, 1> sampleType;
typedef linear_kernel<sampleType> lineearKernel;

class FaceRecognition{
public:
	FaceRecognition();

private:
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
protected:
	frontal_face_detector detector;
	shape_predictor sp;
	anet_type net;
	Mat defaultImage = imread("Lenna.png");
	Mat frame;
	//image_window win;
	std::vector<matrix<rgb_pixel>> faces;

	

	std::vector<matrix<float, 0, 1>> faceDescriptors;	
	
	std::vector<sampleType> samples;	//Samples for training
	std::vector<string> labels;			//Label for training (temporary: need to change type to string(name))

	svm_multiclass_linear_trainer<lineearKernel, string> trainer;
	multiclass_linear_decision_function<lineearKernel, string> decisionFunction;
	
	/*************************
	Training parameter
	*************************/
	const float trainC = 1.0;
	const float trainEpsilon = 1e-3;
	const bool trainNonnegativeWeight = true;
	const int trainMaxIteration = 1e8;
	const int trainNumThread = 10;
public:
	std::vector<matrix<rgb_pixel>> JitterImage(const matrix<rgb_pixel>& img);
	std::vector<matrix<rgb_pixel>> GetFaceFromImage(std::vector<Mat> & imgs);
	std::vector<faceFeature> FacesToVector(std::vector<matrix<rgb_pixel>> & imgs);

	sampleType FaceFeatureToSample(faceFeature& ft);
	sampleType MatToSample(Mat& img);

	void SetLabels(std::vector<std::string>& labels);
	void SamplingForTraining();
	void MakeEdges();
	void GetClusterNum();
	void ShowClusteringResult();
	matrix<rgb_pixel> MatToRGB(Mat& src);
	void AddToSVM(std::vector<Mat> & imgs);
	void Train();
	std::vector<faceFeature> MatToFaceFeture(Mat& img);
	std::vector<std::string> Prediction(Mat& img);
};

