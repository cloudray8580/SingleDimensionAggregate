#pragma once

#include "STXBtreeAggregate.h"
#include "mlpack/core.hpp"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include "Utils.h"

using namespace std;

class ATree {
public:
	ATree(double error = 100, double relative_error_threshold = 0.01) {
		this->error_threshold = error;
		this->relative_error_threshold = relative_error_threshold;
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
			bottom_layer_index.insert(pair<double, int>(dataset_range[i].first, i));
		}

		/*stx::btree<double, int>::iterator iter;
		iter = this->bottom_layer_index.lower_bound(-0.047);
		cout << "model index: " << iter->second << endl;*/
	}

	void TrainAtree(vector<double> &key_v, vector<double> &position_v) {

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

				dataset_range.push_back(pair<double, double>(key_v[cone_origin_index], key_v[cone_shift_index - 1]));
				current_slope = (position_v[cone_shift_index - 1] - position_v[cone_origin_index]) / (key_v[cone_shift_index - 1] - key_v[cone_origin_index]); // monotonous
				//atree_parameters.push_back(pair<double, double>(current_slope, position_v[cone_origin_index]));
				vector<double> paras;
				paras.push_back(current_slope);
				paras.push_back(position_v[cone_origin_index]);
				atree_parameters.push_back(paras);
				//cout << cone_origin_index << " " << cone_shift_index - 1 << " " << current_slope << " " << position_v[cone_origin_index] << endl;
				cone_origin_index = cone_shift_index;
			}
			else {
				dataset_range.push_back(pair<double, double>(key_v[cone_origin_index], key_v[cone_shift_index - 1])); // exit loop as the border is met
				current_slope = (position_v[cone_shift_index - 1] - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]); // monotonous
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
			bottom_layer_index.insert(pair<double, int>(dataset_range[i].first, i));
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
			iter = this->bottom_layer_index.upper_bound(queryset[i]); // bug for STXBtree when search -0.041 between -0.048 and 0.00055, results in the last element.
			iter--;
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
			iter = this->bottom_layer_index.upper_bound(queryset[i]); // bug for STXBtree when search -0.041 between -0.048 and 0.00055, results in the last element.
			iter--;
			model_index = iter->second;
			//cout << "queryset[i]: " << queryset[i] << "  model index:" << model_index << endl;
			//a = atree_parameters[model_index].first;
			//b = atree_parameters[model_index].second;
			a = atree_parameters[model_index][0];
			b = atree_parameters[model_index][1];
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

	// with refinement 
	void CountPrediction(vector<double> &queryset_low, vector<double> &queryset_up, vector<int> &results, vector<double> &key_v, string recordfilepath = "C:/Users/Cloud/Desktop/LearnedAggregateData/experiment_result.csv") {

		// build the full key index
		stx::btree<double, int> full_key_index;
		for (int i = 0; i < key_v.size(); i++) {
			full_key_index.insert(pair<double, int>(key_v[i], i));
		}

		results.clear();
		stx::btree<double, int>::iterator iter;
		int model_index = 0;
		double result, result_low, result_up, key_low, key_up;

		int count_refinement = 0;
		double max_err_rel = 0; // the estimated maximum possible relative error

		double a, b;

		auto t0 = chrono::steady_clock::now();
		// for each range query pair
		for (int i = 0; i < queryset_low.size(); i++) {

			// calculate the lower key position
			iter = this->bottom_layer_index.upper_bound(queryset_low[i]);
			iter--;
			model_index = iter->second;
			a = atree_parameters[model_index][0];
			b = atree_parameters[model_index][1];
			//cout << a << " " << b << " " << model_index << endl;
			result_low = a * (queryset_low[i] - dataset_range[model_index].first) + b;

			// calculate the upper key position
			iter = this->bottom_layer_index.upper_bound(queryset_up[i]);
			iter--;
			model_index = iter->second;
			a = atree_parameters[model_index][0];
			b = atree_parameters[model_index][1];
			//cout << a << " " << b << " " << model_index << endl;
			result_up = a * (queryset_up[i] - dataset_range[model_index].first) + b;
			
			// calculate the COUNT
			result = result_up - result_low;

			// analysis estimated maximum relative error:
			max_err_rel = (2 * error_threshold) / (result - 2 * error_threshold);
			//cout << result_low << "  " << result_up << "  " << result << "  " << max_err_rel << endl;
			if (max_err_rel > relative_error_threshold || max_err_rel < 0) {
				count_refinement++;
				// do refinement
				iter = full_key_index.find(queryset_low[i]);
				result_low = iter->second;
				iter = full_key_index.find(queryset_up[i]);
				result_up = iter->second;
				result = result_up - result_low;
			}

			results.push_back(result);
		}
		auto t1 = chrono::steady_clock::now();
		cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
		cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset_low.size()) << " ns" << endl;
		cout << "refinement count: " << count_refinement << endl;
		cout << "hit probability: " << 1000 - count_refinement << " / 1000" << endl;

		// record experiment result;
		/*ofstream outfile_exp;
		outfile_exp.open(recordfilepath, std::ios_base::app);
		outfile_exp << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset_low.size()) << "," << 1000-count_refinement << endl;
		outfile_exp.close();*/
	}

	QueryResult CountPrediction2(vector<double> &queryset_low, vector<double> &queryset_up, vector<int> &results, vector<double> &key_v) {

		// build the full key index
		stx::btree<double, int> full_key_index;
		for (int i = 0; i < key_v.size(); i++) {
			full_key_index.insert(pair<double, int>(key_v[i], i));
		}

		results.clear();
		stx::btree<double, int>::iterator iter;
		int model_index = 0;
		double result, result_low, result_up, key_low, key_up;

		int count_refinement = 0;
		double max_err_rel = 0; // the estimated maximum possible relative error

		double a, b;

		auto t0 = chrono::steady_clock::now();
		// for each range query pair
		for (int i = 0; i < queryset_low.size(); i++) {

			// calculate the lower key position
			iter = this->bottom_layer_index.upper_bound(queryset_low[i]);
			iter--;
			model_index = iter->second;
			a = atree_parameters[model_index][0];
			b = atree_parameters[model_index][1];
			//cout << a << " " << b << " " << model_index << endl;
			result_low = a * (queryset_low[i] - dataset_range[model_index].first) + b;

			// calculate the upper key position
			iter = this->bottom_layer_index.upper_bound(queryset_up[i]);
			iter--;
			model_index = iter->second;
			a = atree_parameters[model_index][0];
			b = atree_parameters[model_index][1];
			//cout << a << " " << b << " " << model_index << endl;
			result_up = a * (queryset_up[i] - dataset_range[model_index].first) + b;

			// calculate the COUNT
			result = result_up - result_low;

			// analysis estimated maximum relative error:
			max_err_rel = (2 * error_threshold) / (result - 2 * error_threshold);
			//cout << result_low << "  " << result_up << "  " << result << "  " << max_err_rel << endl;
			if (max_err_rel > relative_error_threshold || max_err_rel < 0) {
				count_refinement++;
				// do refinement
				iter = full_key_index.find(queryset_low[i]);
				result_low = iter->second;
				iter = full_key_index.find(queryset_up[i]);
				result_up = iter->second;
				result = result_up - result_low;
			}

			results.push_back(result);
		}
		auto t1 = chrono::steady_clock::now();

		/*cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
		cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset_low.size()) << " ns" << endl;
		cout << "refinement count: " << count_refinement << endl;
		cout << "hit probability: " << 1000 - count_refinement << " / 1000" << endl*/;

		auto average_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / queryset_low.size();
		auto total_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();

		double MEabs, MErel;
		MeasureAccuracy(results, "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/TWEET_1D.csv", MEabs, MErel);

		QueryResult query_result;
		query_result.average_query_time = average_time;
		query_result.total_query_time = total_time;
		query_result.measured_absolute_error = MEabs;
		query_result.measured_relative_error = MErel;
		query_result.model_amount = dataset_range.size();
		query_result.tree_paras = this->bottom_layer_index.CountParametersPrimary();
		query_result.total_paras = this->dataset_range.size() * 4 + query_result.tree_paras;

		return query_result;
	}


	vector<double> keys;
	vector<pair<double, double>> dataset_range;
	//vector<pair<double, double>> atree_parameters; // first: slope, second:cone origin position, I think this maybe a bit faster, but as ROL use vector as inner, then its unfair
	vector<vector<double>> atree_parameters;

	stx::btree<double, int> bottom_layer_index; // for btree index

	double error_threshold; // denote the maximum prediction error
	double relative_error_threshold;
};