#pragma once
#include "LearnIndexAggregate.h"

#include <string>
#include <windows.h>
//using namespace std;

void GenerateRandomQuerySet(string filename_query) {
	arma::mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	dataset.shed_rows(1, 2);

	//double max = dataset.max(); // 84.9807
	//double min = dataset.min(); // -85.0511
	//cout << max << " " << min << endl;
	double max = 84.9807;
	double min = -85.0511;
	double range = max - min;

	ofstream outfile;
	outfile.open(filename_query);

	double upper, lower;
	srand((unsigned)time(NULL));

	vector<double> uppers;
	vector<double> lowers;
	for (int i = 0; i < 1000; i++) {
		upper = rand() / double(RAND_MAX) * range + min;
		lower = rand() / double(RAND_MAX) * range + min;
		uppers.push_back(upper);
	}
	Sleep(1000);
	for (int i = 0; i < 1000; i++) {
		upper = rand() / double(RAND_MAX) * range + min;
		lower = rand() / double(RAND_MAX) * range + min;
		lowers.push_back(lower);
	}

	for (int i = 0; i < 1000; i++) {
		if (uppers[i] >= lowers[i]) {
			outfile << lowers[i] << "," << uppers[i] << endl;
		}
		else {
			outfile << uppers[i] << "," << lowers[i] << endl;
		}
	}
	outfile.close();
}

void Statistic(string filename_query) {

}

void CalculateRealSum(vector<double> &results) {

}