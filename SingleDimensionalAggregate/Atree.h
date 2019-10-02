#pragma once

#include "STXBtreeAggregate.h"
#include "mlpack/core.hpp"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include "Utils.h"

using namespace std;

class ATree {
public:
	ATree(int error = 100) {
		this->error_threshold = error;
	}

	void TrainAtree(const arma::rowvec& dataset, const arma::rowvec& labels) {
		
		vector<double> key_v, position_v;
		RowvecToVector(dataset, key_v);
		RowvecToVector(labels, position_v);
		this->keys = key_v;
		
		int TOTAL_SIZE = key_v.size();
		double slope_high, slope_low;

		int current_index = 0;
		int cone_origin_index = 0;
		int cone_shift_index = 0;

		double upper_pos, lower_pos;

		double current_slope;
		dataset_range.clear();
		bool exit_tag = false;

		// while the dataset is not empty, build segements
		while (cone_origin_index < TOTAL_SIZE) {

			/*if (dataset_range.size() == 121) {
				cout << "debug here" << endl;
			}*/

			// build the cone with first and second point
			cone_shift_index = cone_origin_index + 1;

			if (cone_shift_index >= TOTAL_SIZE) {
				dataset_range.push_back(pair<double, double>(key_v[cone_origin_index], key_v[cone_shift_index]));
				break;
			}

			//cout << position_v[cone_shift_index] << " " << position_v[cone_origin_index] << " " << key_v[cone_shift_index] << " " << key_v[cone_origin_index] << endl;

			while (key_v[cone_shift_index] == key_v[cone_origin_index]) {
				cone_shift_index++;
			}

			//cout << position_v[cone_shift_index] << " " << position_v[cone_origin_index] << " " << key_v[cone_shift_index] << " " << key_v[cone_origin_index] << endl;

			slope_high = (position_v[cone_shift_index] + error_threshold - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
			slope_low = (position_v[cone_shift_index] - error_threshold - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);

			//cout << "slope: " << slope_high << " " << slope_low << endl;

			exit_tag = false;

			// test if the following points are in the cone
			while (cone_shift_index < TOTAL_SIZE) {
				cone_shift_index++;

				// test if exceed  the border, if the first time, if the last segement

				upper_pos = slope_high * (key_v[cone_shift_index] - key_v[cone_origin_index]) + position_v[cone_origin_index];
				lower_pos = slope_low * (key_v[cone_shift_index] - key_v[cone_origin_index]) + position_v[cone_origin_index];

				if (position_v[cone_shift_index] < upper_pos && position_v[cone_shift_index] > lower_pos) {
					// inside the conde, update the slope
					if (position_v[cone_shift_index] + error_threshold < upper_pos) {
						// update slope_high
						slope_high = (position_v[cone_shift_index] + error_threshold - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
					}
					if (position_v[cone_shift_index] - error_threshold > lower_pos) {
						// update slope_low
						slope_low = (position_v[cone_shift_index] - error_threshold - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
					}
				}
				else {
					// outside the cone, start a new segement.
					exit_tag = true;
					break;
				}
			}
			//  save the current segement
			if (exit_tag) {
				//cout << cone_origin_index << " " << cone_shift_index << " " << key_v[cone_shift_index] << " " << key_v[cone_shift_index+1] << endl;

				dataset_range.push_back(pair<double, double>(key_v[cone_origin_index], key_v[cone_shift_index-1]));
				current_slope = (position_v[cone_shift_index-1] - position_v[cone_origin_index]) / (key_v[cone_shift_index-1] - key_v[cone_origin_index]); // monotonous
				//atree_parameters.push_back(pair<double, double>(current_slope, position_v[cone_origin_index]));
				vector<double> paras;
				paras.push_back(current_slope);
				paras.push_back(position_v[cone_origin_index]);
				atree_parameters.push_back(paras);
				//cout << cone_origin_index << " " << cone_shift_index - 1 << " " << current_slope << " " << position_v[cone_origin_index] << endl;
				cone_origin_index = cone_shift_index;
			}
			else {
				dataset_range.push_back(pair<double, double>(key_v[cone_origin_index], key_v[cone_shift_index-1])); // exit loop as the border is met
				current_slope = (position_v[cone_shift_index-1] - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]); // monotonous
				//atree_parameters.push_back(pair<double, double>(current_slope, position_v[cone_origin_index]));
				vector<double> paras;
				paras.push_back(current_slope);
				paras.push_back(position_v[cone_origin_index]);
				atree_parameters.push_back(paras);
				break; // all the data set has been scanned through
			}
		}

		// build a btree on top of the models
		bottom_layer_index.clear();
		for (int i = 0; i < dataset_range.size(); i++) {
			bottom_layer_index.insert(pair<double, int>(dataset_range[i].second, i));
		}

		/*stx::btree<double, int>::iterator iter;
		iter = this->bottom_layer_index.lower_bound(-0.047);
		cout << "model index: " << iter->second << endl;*/
	}

	void Predict(vector<double> &queryset, vector<int> &results) {
		stx::btree<double, int>::iterator iter;
		results.clear();
		int model_index = 0;
		double a, b;
		double result;
		for (int i = 0; i < queryset.size(); i++) {
			/*if (i == 717) {
				cout << "debug here!" << endl;
			}*/
			iter = this->bottom_layer_index.lower_bound(queryset[i]); // bug for STXBtree when search -0.041 between -0.048 and 0.00055, results in the last element.
			/*if (iter == this->bottom_layer_index.end()) {
				cout << "end!" << endl;
			}*/
			model_index = iter->second;
			//cout << "queryset[i]: " << queryset[i] << "  model index:" << model_index << endl;
			//a = atree_parameters[model_index].first;
			//b = atree_parameters[model_index].second;
			a = atree_parameters[model_index][0];
			b = atree_parameters[model_index][1];
			//cout << a << " " << b << " " << model_index << endl;
			result = a * (queryset[i] - dataset_range[model_index].first) + b;
			results.push_back(result);
		}
	}

	void PredictExact(vector<double> &queryset, vector<int> &results) {
		stx::btree<double, int>::iterator iter;
		results.clear();
		int model_index = 0;
		double a, b;
		double result;
		for (int i = 0; i < queryset.size(); i++) {
			iter = this->bottom_layer_index.lower_bound(queryset[i]); // bug for STXBtree when search -0.041 between -0.048 and 0.00055, results in the last element.
			model_index = iter->second;
			//cout << "queryset[i]: " << queryset[i] << "  model index:" << model_index << endl;
			//a = atree_parameters[model_index].first;
			//b = atree_parameters[model_index].second;
			a = atree_parameters[model_index][0];
			b = atree_parameters[model_index][0];
			//cout << a << " " << b << " " << model_index << endl;
			result = a * (queryset[i] - dataset_range[model_index].first) + b;

			// do binary search
			//int loop = 0;
			int temp_threshold = error_threshold;
			int lower_pos = result - temp_threshold, upper_pos = result + temp_threshold;
			while (keys[result] != queryset[i] && temp_threshold >= 1) {
				if (keys[result] > queryset[i]) {
					upper_pos = result;
					result = int(lower_pos + upper_pos) / 2;
				}
				else if (keys[result] < queryset[i]) {
					lower_pos = result;
					result = int(lower_pos + upper_pos) / 2;
				}
				temp_threshold /= 2;
				//loop++;
			}
			//cout << loop << endl;
			results.push_back(result);
		}
	}

	vector<double> keys;
	vector<pair<double, double>> dataset_range;
	//vector<pair<double, double>> atree_parameters; // first: slope, second:cone origin position, I think this maybe a bit faster, but as ROL use vector as inner, then its unfair
	vector<vector<double>> atree_parameters;

	stx::btree<double, int> bottom_layer_index; // for btree index

	int error_threshold; // denote the maximum prediction error
};