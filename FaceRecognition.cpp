#include "pch.h"
#include "FaceRecognition.h"
const int DEBUGING = 0;

FaceRecognition::FaceRecognition() {
	detector = get_frontal_face_detector();

	deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

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

void FaceRecognition::MakeEdges() {
	for (size_t i = 0; i < faceDescriptors.size(); ++i)
		for (size_t j = i; j < faceDescriptors.size(); ++j)
			if (length(faceDescriptors[i] - faceDescriptors[j]) < 0.6)
				edges.push_back(sample_pair(i, j));
}

void FaceRecognition::GetClusterNum() {
	clusterNum = chinese_whispers(edges, labels);
	
}

void FaceRecognition::ShowClusteringResult() {
	std::vector<image_window> win_clusters(clusterNum);

	for (size_t cluster_id = 0; cluster_id < clusterNum; ++cluster_id) {
		std::vector<matrix<rgb_pixel>> temp;
		for (size_t j = 0; j < labels.size(); ++j) {
			if (cluster_id == labels[j])
				temp.push_back(faces[j]);
		}
		win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
		win_clusters[cluster_id].set_image(tile_images(temp));
	}
}

void FaceRecognition::AddToSVM(std::vector<Mat> & imgs) {
	faces = GetFaceFromImage(imgs);
	faceDescriptors = FacesToVector(faces);
}

void FaceRecognition::Train() {
	trainer.set_lambda(lambda);
	trainer.set_kernel(kernelType(kernelT));
	trainer.set_max_num_sv(iterCount);


	for (int i = 0; i < faceDescriptors.size(); ++i) 
		labels.push_back(i + 1);
	MessageBox(NULL, "Parameters setting done", 0, 0);
	//cross_validate_trainer(batch_cached(trainer, 0.1), faceDescriptors, labels, 4);
	MessageBox(NULL, "cross_validate_trainer setting done", 0, 0);
	df = verbose_batch_cached(trainer, 0.1).train(faceDescriptors, labels);
}
std::vector<faceFeature> FaceRecognition::MatToFaceFeture(Mat& img) {
	auto unknownFaces = GetFaceFromImage(std::vector<Mat>{img});
	auto feature = FacesToVector(unknownFaces);
	return feature;
}

float FaceRecognition::Prediction(Mat& img) {
	auto feature = MatToFaceFeture(img);
	FILE * fp = freopen("result.txt", "w", stdout);
	for (int i = 0; i < feature.size(); ++i) {
		printf("%f\n", df(feature[i]));
	}
	fclose(fp);
	return 0.0;
}