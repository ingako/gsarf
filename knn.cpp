#include "opencv2/highgui.hpp"
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include <iostream>

using namespace cv;
using namespace cv::ml;
using namespace std;

int read_data_from_csv(FILE *f,
		Mat &data, Mat &responses,
		int n_samples, int attributes_per_sample) {

	data = Mat(n_samples, attributes_per_sample, CV_32FC1);
	responses = Mat(n_samples, 1, CV_32FC1);

	float tmp;

	// for each sample in the file
	for (int line = 0; line < n_samples; line++) {

		// for each attribute on the line in the file
		for (int attribute = 0; attribute < (attributes_per_sample + 1);
				attribute++) {
			if (attribute < attributes_per_sample) {
				// attributes
				fscanf(f, "%f,", &tmp);
				data.at<float>(line, attribute) = tmp;
				if (n_samples == 1) {
					printf("%f,", data.at<float>(line, attribute));
				}

			} else if (attribute == attributes_per_sample) {
				// last attribute is the class label
				fscanf(f, "%f,", &tmp);
				responses.at<float>(line, 0) = tmp;
				// printf("%f\n", classes.at<float>(line, 0));
			}
		}
		// printf("\n");
	}

	return 0;
}

int main() {

	int number_of_train_elements = 200000;
	int number_of_sample_elements = 1;
	int attributes_per_sample = 24;
	const char* filename = "data/led_knn/0.csv";

	Mat matTrainFeatures;
	Mat matTrainLabels;

	Mat matSample;
	Mat matSampleLabels;

	Mat matResults(0, 0, CV_32F);
	Mat matResponses(0, 0, CV_32F);
	Mat matDistances(0, 0, CV_32F);

	FILE *f = fopen(filename, "r");
	if (!f) {
		printf("ERROR: cannot read file %s\n", filename);
		return 1;
	} else {
		printf("Reading file %s...\n", filename);
	}

	// loading data into Mat variables
	read_data_from_csv(f,
			matTrainFeatures, matTrainLabels,
			number_of_train_elements, attributes_per_sample);

	read_data_from_csv(f,
			matSample, matSampleLabels,
			number_of_sample_elements, attributes_per_sample);

	cout << "matSample: " << endl << matSample << endl << endl;
	fclose(f);

	Ptr<TrainData> trainingData;
	Ptr<KNearest> kclassifier = KNearest::create();

	trainingData = TrainData::create(matTrainFeatures, SampleTypes::ROW_SAMPLE,
			matTrainLabels);

	kclassifier->setIsClassifier(true);
	kclassifier->setAlgorithmType(KNearest::Types::BRUTE_FORCE);
	kclassifier->setDefaultK(1000);

	kclassifier->train(trainingData);
	kclassifier->findNearest(matSample, kclassifier->getDefaultK(), matResults, matResponses, matDistances);

	// log settings
	cout << "Training data: " << endl
		<< "getSamplesSize\t" << trainingData->getNSamples() << endl
		// << "getSamples\n"
		// << trainingData->getSamples() << endl
		<< endl;

	cout << "Classifier: " << endl
		<< "kclassifier->getDefaultK(): " << kclassifier->getDefaultK() << endl
		<< "kclassifier->getIsClassifier(): "
		<< kclassifier->getIsClassifier() << endl
		<< "kclassifier->getAlgorithmType(): " << kclassifier->getAlgorithmType()
		<< endl
		<< endl;

	// confirming sample order
	cout << "matSample: " << endl << matSample << endl << endl;

	// displaying the results
	cout << "matResults: " << endl << matResults << endl << endl;
	cout << "matResponses: " << endl << matResponses << endl << endl;
	cout << "matDistances: " << endl << matDistances << endl << endl;

	return 0;
}
