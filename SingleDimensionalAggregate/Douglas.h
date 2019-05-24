#pragma once

#include "STXBtreeAggregate.h"
#include "mlpack/core.hpp"
#include "Utils.h"

using namespace std;

class Douglas {
public:
	Douglas(int error = 100) {
		this->error_threshold = error;
	}

	void inline DogulasPeucker(int start_index, int end_index) {
		double dmax = 0;
		int index = start_index + 1;
		int dmax_index = index;

		double slope = (position_v[end_index] - position_v[start_index]) / (key_v[end_index] - key_v[start_index]);
		double distance = 0;

		for (; index < end_index; index++) {
			distance = abs(slope * (key_v[index] - key_v[start_index]) + position_v[start_index] - position_v[index]);
			if (distance > dmax) {
				dmax = distance;
				dmax_index = index;
			}
		}

		if (dmax > error_threshold) {
			//cout << start_index << " " << dmax_index << " " << end_index << " " << dmax << " " << key_v[dmax_index] << " " << key_v[end_index] << endl;
			DogulasPeucker(start_index, dmax_index);
			DogulasPeucker(dmax_index, end_index);
		}
		else {
			dataset_range.push_back(pair<double, double>(key_v[start_index], key_v[end_index])); // exit loop as the border is met
			parameters.push_back(pair<double, double>(slope, position_v[start_index]));
		}
	}

	void BuildDouglas(const arma::rowvec& dataset, const arma::rowvec& labels) {

		//vector<double> key_v, position_v;
		RowvecToVector(dataset, key_v);
		RowvecToVector(labels, position_v);

		DogulasPeucker(0, key_v.size() - 1);

		// build a btree on top of the models
		bottom_layer_index.clear();
		for (int i = 0; i < dataset_range.size(); i++) {
			bottom_layer_index.insert(pair<double, int>(dataset_range[i].second, i));
		}
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
			a = parameters[model_index].first;
			b = parameters[model_index].second;
			//cout << a << " " << b << " " << model_index << endl;
			result = a * (queryset[i] - dataset_range[model_index].first) + b;
			results.push_back(result);
		}
	}

	vector<double> key_v, position_v;
	vector<pair<double, double>> dataset_range;
	vector<pair<double, double>> parameters; // first: slope, second:cone origin position
	stx::btree<double, int> bottom_layer_index; // for btree index

	int error_threshold; // denote the maximum prediction error
};