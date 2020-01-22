#pragma once
#include <vector>
#include <math.h>
#include <algorithm>

using namespace std;

double inline CalculateEntropy(double val1, double val2) {
	//double temp1 = -val1 * log2(val1);
	//double temp2 = -val2 * log2(val2);
	//return temp1 + temp2;
	return -val1 * log2(val1) - val2 * log2(val2);
}

double inline CalculateEntropy(vector<double> &values, double &total) {
	double entropy = 0;
	for (int i = 0; i < values.size(); i++) {
		entropy += (values[i] / total)*log2(values[i] / total);
	}
	return -entropy;
}

vector<int> EntropyHist(int buckets, vector<double> &keys) {
	
	// order and get statistic about key
	vector<pair<double, int>> key_frequency;
	vector<pair<double, double>> key_spread; // s0 = v1, sD = 1
	vector<pair<double, double>> key_area; 

	double current_key = keys[0];
	int frequency = 0;
	double spread = 0;
	for (int i = 0; i < keys.size(); i++) {
		if (current_key == keys[i]) {
			frequency++;
		}
		else {
			key_frequency.push_back(pair<double, int>(current_key, frequency));
			spread = keys[i] - current_key;
			key_spread.push_back(pair<double, double>(current_key, spread));
			key_area.push_back(pair<double, double>(current_key, frequency*spread));
			
			frequency = 1;
			current_key = keys[i];
		}
	}
	key_frequency.push_back(pair<double, int>(current_key, frequency));
	spread = 1;
	key_spread.push_back(pair<double, double>(current_key, spread));
	key_area.push_back(pair<double, double>(current_key, frequency*spread));

	// generate boundary position
	vector<int> previous_buckets;
	vector<int> boundary;
	
	vector<double> area_list;
	double bucket_weight; // sum of area
	
	int local_cut_position;
	double local_min_entropy = std::numeric_limits<double>::infinity(); // to be MAX
	double left_total_frequency = 0;
	double right_totl_frequency = 0;

	double left_entropy = 0;
	double right_entropy = 0;
	double original_entropy = 0;

	double x; // jth_frequency

	double entropy_deduction;
	double weighted_entropy_deduction;

	double min_weighted_entropy_deduction = std::numeric_limits<double>::infinity();
	int min_weight_entropy_bucket;
	int min_local_cut_pos = 0;

	//previous_buckets.push_back(0);
	boundary.push_back(0);

	while (boundary.size() < buckets) {
		cout << boundary.size() << endl;
		min_weighted_entropy_deduction = std::numeric_limits<double>::infinity();

		for (int i = 0; i < boundary.size(); i++) {

			// get area list of this bucket, Ab
			area_list.clear();
			double bucket_boundary_key1 = keys[boundary[i]];
			double bucket_boundary_key2 = 0;
			if (i + 1 < boundary.size()) {
				bucket_boundary_key2 = keys[boundary[i + 1]];
			}
			else {
				bucket_boundary_key2 = keys[keys.size() - 1] + 1;
			}
			for (int j = 0; j < key_area.size(); j++) {
				if (key_area[j].first >= bucket_boundary_key1 && key_area[j].first < bucket_boundary_key2) {
					area_list.push_back(key_area[j].second);
				}
			}

			// if more than 1 value
			if (area_list.size() > 1) {
				bucket_weight = 0;
				for (int j = 0; j < area_list.size(); j++) {
					bucket_weight += area_list[j];
				}

				// init local cut pos to be 0
				local_cut_position = 0;
				local_min_entropy = std::numeric_limits<double>::infinity(); // Max
				left_total_frequency = 0;
				right_totl_frequency = bucket_weight;
				left_entropy = 0;

				// calculate original entropy, H(Ab)
				original_entropy = CalculateEntropy(area_list, bucket_weight);
				right_entropy = original_entropy;

				// cout << right_entropy << " " << original_entropy << endl;

				// find local best cut
				for (int j = 0; j < area_list.size() - 1; j++) {

					/*if (j == 485347) {
						cout << "debug here " << area_list[j] << endl;
					}*/

					x = area_list[j]; //jth_frequency

					if (j == 0) {
						// notice HL(1) is also 0
						left_entropy = 0;
					}
					else {
						// for HL(1), it cannot be updated by this
						left_entropy = CalculateEntropy(x / (x + left_total_frequency), left_total_frequency / (x + left_total_frequency)) + left_total_frequency / (x + left_total_frequency)*left_entropy;
					}

					//cout << left_entropy << endl;

					right_entropy = (right_entropy - CalculateEntropy(x/right_totl_frequency,(right_totl_frequency-x)/right_totl_frequency)) * right_totl_frequency / (right_totl_frequency - x);

					//cout << right_entropy << endl;

					left_total_frequency += x;
					right_totl_frequency -= x;

					entropy_deduction = original_entropy - left_entropy - right_entropy;

					//cout << entropy_deduction << " " << original_entropy << endl;

					if (isnan(entropy_deduction)) {
						cout << "debug here " << j << " " << i << " " << boundary.size() << left_entropy << " " << right_entropy  << " " << right_totl_frequency - x << " " << x << " " << right_totl_frequency << endl;
					}

					if (entropy_deduction < local_min_entropy) {
						local_cut_position = j+1;
						local_min_entropy = entropy_deduction;
					}
				} // end for

				// compare for global best cut
				weighted_entropy_deduction = bucket_weight * local_min_entropy;
				if (weighted_entropy_deduction < min_weighted_entropy_deduction) {
					min_weight_entropy_bucket = i;
					min_local_cut_pos = local_cut_position;
					min_weighted_entropy_deduction = weighted_entropy_deduction;
				}

			} // end if
		} // end for

		// set the global cut pos
		boundary.push_back(boundary[min_weight_entropy_bucket]+ min_local_cut_pos);
		// sort the boundary
		std::sort(boundary.begin(), boundary.end());
	}

	for (int i = 0; i < boundary.size(); i++) {
		cout << boundary[i] << endl;
	}

	mat dataset;
	VectorToRowvec(dataset, boundary);
	dataset.save("C:/Users/Cloud/Desktop/LearnIndex/data/EntropyHistSplits10000.bin");

	return boundary;
}