#include "pch.h"
#include "FaceRecognition.h"
const int DEBUGING = 1;

FaceRecognition::FaceRecognition() {
	detector = get_frontal_face_detector();

	deserialize("shape_predictor_5_face_landmarks.dat") >> sp;

	deserialize("dlib_face_recognition_resnet_model_v1.dat") >> net;

	if (DEBUGING) {
		matrix<rgb_pixel> img;

		cv_image<bgr_pixel> tmpImage(defaultImage);
		assign_image(img, tmpImage);

		win.set_image(img);
	}
}

std::vector<matrix<rgb_pixel>> FaceRecognition::JitterImage(const matrix<rgb_pixel>& img) {
	// All this function does is make 100 copies of img, all slightly jittered by being
	// zoomed, rotated, and translated a little bit differently. They are also randomly
	// mirrored left to right.
	thread_local dlib::rand rnd;

	std::vector<matrix<rgb_pixel>> crops;
	for (int i = 0; i < 100; ++i)
		crops.push_back(::jitter_image(img, rnd));
	return crops;
}

int FaceRecognition::GetFaceFromImage() {
	faces.clear();

	for (auto face : detector(frame)) {
		auto shape = sp(frame, face);
		matrix<rgb_pixel> face_chip;
		extract_image_chip(frame, get_face_chip_details(shape, 150, 0.25), face_chip);
		faces.push_back(move(face_chip));

		win.add_overlay(face);
	}

	return faces.size();
}

void FaceRecognition::Face2Vec128D() {
	face_descriptors = net(faces);
}

void FaceRecognition::MakeEdges() {
	for (size_t i = 0; i < face_descriptors.size(); ++i)
		for (size_t j = i; j < face_descriptors.size(); ++j)
			if (length(face_descriptors[i] - face_descriptors[j]) < 0.6)
				edges.push_back(sample_pair(i, j));
}

void FaceRecognition::GetClusterNum() {
	clusterNum = chinese_whispers(edges, labels);
}

void FaceRecognition::ShowClusteringResult() {
	std::vector<image_window> win_clusters(clusterNum);
	for (size_t cluster_id = 0; cluster_id < clusterNum; ++cluster_id)
	{
		std::vector<matrix<rgb_pixel>> temp;
		for (size_t j = 0; j < labels.size(); ++j)
		{
			if (cluster_id == labels[j])
				temp.push_back(faces[j]);
		}
		win_clusters[cluster_id].set_title("face cluster " + cast_to_string(cluster_id));
		win_clusters[cluster_id].set_image(tile_images(temp));
	}
}