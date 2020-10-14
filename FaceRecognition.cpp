#include "pch.h"
#include "FaceRecognition.h"
const int DEBUGING = 0;
string rootDir = "";
FaceRecognition::FaceRecognition() {
	detector = get_frontal_face_detector();

	deserialize(rootDir + "model\\shape_predictor_5_face_landmarks.dat") >> sp;

	deserialize(rootDir + "model\\dlib_face_recognition_resnet_model_v1.dat") >> net;

	if (DEBUGING) {
		matrix<rgb_pixel> img;

		cv_image<bgr_pixel> tmpImage(defaultImage);
		assign_image(img, tmpImage);

		//win.set_image(img);
	}
}

matrix<rgb_pixel> FaceRecognition::MatToRGB(Mat& src) {
	matrix<rgb_pixel> img;

	cv_image<bgr_pixel> tmpImage(src);
	assign_image(img, tmpImage);

	return img;
}
//For SVM training
std::vector<matrix<rgb_pixel>> FaceRecognition::JitterImage(const matrix<rgb_pixel>& img) {
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

std::vector<matrix<rgb_pixel>> FaceRecognition::GetFaceFromImage(std::vector<Mat> & imgs) {
	std::vector<matrix<rgb_pixel>> res;
	for (auto img : imgs) {
		auto frame = MatToRGB(img);

		for (auto face : detector(frame)) {
			auto shape = sp(frame, face);
			matrix<rgb_pixel> face_chip;
			extract_image_chip(frame, get_face_chip_details(shape, 150, 0.25), face_chip);
			res.push_back(move(face_chip));

			//win.add_overlay(face);
		}
	}
	return res;
}

std::vector<faceFeature> FaceRecognition::FacesToVector(std::vector<matrix<rgb_pixel>> & imgs) {
	//Make face to 128D vector
	auto res = net(imgs);
	return res;
}
void FaceRecognition::LoadModel() {
	deserialize("model\\pretrained.dat") >> decisionFunction;
}
void FaceRecognition::MakeEdges() {
}

void FaceRecognition::GetClusterNum() {	
}

void FaceRecognition::ShowClusteringResult() {
}

void FaceRecognition::AddToSVM(std::vector<Mat> & imgs) {
	faces = GetFaceFromImage(imgs);
	faceDescriptors = FacesToVector(faces);
}

void FaceRecognition::Train() {

	trainer.set_c(trainC);
	trainer.set_epsilon(trainEpsilon);
	trainer.set_learns_nonnegative_weights(trainNonnegativeWeight);
	trainer.set_max_iterations(trainMaxIteration);
	trainer.set_num_threads(trainNumThread);
	trainer.be_verbose();
	FILE* fp = freopen("er.txt", "w", stdout);
	printf("%d %d\n", samples.size(), labels.size());
	for (int i = 0; i < samples.size(); ++i) {
		for (int j = 0; j < samples[i].nr(); ++j)
			printf("%f ", samples[i](j));
		printf("\t%s", labels[i].c_str());
		puts("");
	}

	decisionFunction = trainer.train(samples, labels);

	deserialize(rootDir + "model\\pretrained.dat") >> decisionFunction;
	char msg[12];
	sprintf(msg, "%d", decisionFunction.number_of_classes());
	MessageBox(0, msg, 0, 0);
	puts("\n\n");
	printf("%d \n", decisionFunction.number_of_classes());
	for (int i = 0; i < decisionFunction.weights.nc(); ++i) {
		printf("%f ", decisionFunction.weights(i));
	}
}
std::vector<faceFeature> FaceRecognition::MatToFaceFeture(Mat& img) {
	auto unknownFaces = GetFaceFromImage(std::vector<Mat>{img});
	auto feature = FacesToVector(unknownFaces);
	return feature;
}

std::vector<std::string> FaceRecognition::Prediction(Mat& img) {
	auto features = MatToFaceFeture(img);
	std::vector<std::string> res;
	freopen("er.txt", "w", stdout);
	for (int i = 0; i < features.size(); ++i) {
		auto sample = FaceFeatureToSample(features[i]);
		for (int j = 0; j < sample.nr(); ++j)
			printf("%f ", sample(j));
		auto pred = decisionFunction.predict(sample);
		res.push_back(pred.first);

		char msg[1234];
		sprintf(msg, "%s(%f)", pred.first, pred.second);
		MessageBox(0, msg, 0, 0);
	}
	
	return res;
}
sampleType FaceRecognition::MatToSample(Mat& img) {
	sampleType res;
}
sampleType FaceRecognition::FaceFeatureToSample(faceFeature& ft) {
	sampleType res;
	int i = 0;
	for (auto it = ft.begin(); it != ft.end(); ++it)
		res(i++) = *it;
	return res;
}

void FaceRecognition::SamplingForTraining() {
	samples.clear();
	for (int i = 0; i < faceDescriptors.size(); ++i) {
		samples.push_back(FaceFeatureToSample(faceDescriptors[i]));
		//Need to add labels processing
	}
}

void FaceRecognition::SetLabels(std::vector<std::string>& labels) {
	this->labels = labels;
}