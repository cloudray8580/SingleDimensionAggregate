#pragma once

#include "STXBtreeAggregate.h"
#include "mlpack/core.hpp"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <vector>

using namespace std;
using namespace mlpack::regression;

class StageModelBottomUp {
public:
	StageModelBottomUp(int level = 4, int blocks = 1000, double threshold=200) {
		this->level = level;
		this->initial_blocks = blocks;
		this->threshold = threshold;

		vector<LinearRegression> stage_layer;
		for (int i = 0; i < level; i++) {
			stage_model_BU.push_back(stage_layer);
		}

		vector<pair<double, double>> dataset_range_layer;
		for (int i = 0; i < level; i++) {
			dataset_range.push_back(dataset_range_layer);
		}

		vector<vector<double>> stage_model_parameter_layer;
		for (int i = 0; i < level; i++) {
			stage_model_parameter.push_back(stage_model_parameter_layer);
		}
	}

	// the step is adaptive to the distribution
	// when increase, += step
	// when decrease, /= 2
	void TrainAdaptiveBottomLayer(const arma::mat& dataset, const arma::rowvec& labels) {
		double current_error = 0;
		int continue_block = 1;
		int TOTAL_SIZE = dataset.n_cols;
		int initial_step = TOTAL_SIZE / initial_blocks;
		int step = initial_step; // the initial block
		int dataset_begin = 0, dataset_end = 0 + step;

		int model_index = 0;
		int layer = this->level - 1;

		// for each continous block
		arma::mat current_trainingset = dataset.cols(dataset_begin, dataset_end);
		arma::rowvec current_response = labels.cols(dataset_begin, dataset_end);

		double current_accuracy, stable_accuracy;
		double current_absolute_accuracy, stable_absolute_accuracy;
		bool first_flag = true;
		int block_count = 0;
		LinearRegression lr_stable;

		// for the entire dataset, train the bottom layer!
		while (true) {

			first_flag = true;
			block_count = 1;

			if (model_index == 14) {
				cout << "debug here!" << endl;
			}

			// for an continous_block
			do {
				current_trainingset = dataset.cols(dataset_begin, dataset_end);
				current_response = labels.cols(dataset_begin, dataset_end);

				//cout << dataset_begin << " " << dataset_end << endl;

				LinearRegression lr_poineer(current_trainingset, current_response);

				// check finished?
				if (dataset_end == TOTAL_SIZE - 1) {
					stage_model_BU[layer].push_back(lr_stable);
					dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1)));
					break;
				}

				// Test the lr's accuracy
				current_accuracy = MeasureAccuracySingleLR(current_trainingset, current_response, lr_poineer);
				current_absolute_accuracy = MeasureAbsoluteAccuracySingleLR(current_trainingset, current_response, lr_poineer);

				// if the accuracy is below threshold, continous merge with the next block
				if (current_absolute_accuracy < threshold) {
					//step += step;
					step *= 2;
					dataset_end += step; // judge border!!!
					if (dataset_end >= TOTAL_SIZE) {
						dataset_end = TOTAL_SIZE - 1;
					}
					first_flag = false;
					block_count++;
					lr_stable = lr_poineer;
					stable_accuracy = current_accuracy;
					stable_absolute_accuracy = current_absolute_accuracy;
					continue;
				}
				else if (first_flag == true) {// if the the first block accuracy is tooo large, divide the region into half and start again

					step /= 2;
					dataset_end -= step;
					continue;

					//// save the model
					//lr_stable = lr_poineer;
					//stage_model_BU[layer].push_back(lr_stable);
					////dataset_range[layer].push_back(pair<int, int>(dataset_begin, dataset_end));
					//dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1)));
					//stable_accuracy = current_accuracy;
					//stable_absolute_accuracy = current_absolute_accuracy;
					//break; // currently, just break;
				}
				else { // else if the second time accuracy is too large
					dataset_end -= step; // backward
					block_count--;
					// save the best range and model
					stage_model_BU[layer].push_back(lr_stable);
					//dataset_range[layer].push_back(pair<int, int>(dataset_begin, dataset_end));
					current_trainingset = dataset.cols(dataset_begin, dataset_end); // as the dataset_end here has been adjusted!
					dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1)));
					break;
				}

			} while (true);

			// reset the start and end for the next continous block(s)

			cout << model_index << " " << dataset_begin << " " << dataset_end << " " << stable_accuracy << " " << stable_absolute_accuracy << " " << block_count << endl;
			dataset_begin = dataset_end + 1;
			if (dataset_begin >= TOTAL_SIZE) {
				break; // meet the end
			}
			else {
				step = initial_step; // reset step!
				dataset_end = dataset_begin + step;
				if (dataset_end >= TOTAL_SIZE) {
					dataset_end = TOTAL_SIZE - 1;
				}
				model_index++;
			}
		}
	}

	void TrainBottomLayer(const arma::mat& dataset, const arma::rowvec& labels) {
		double current_error = 0;
		int continue_block = 1;
		int TOTAL_SIZE = dataset.n_cols;
		int step = TOTAL_SIZE / initial_blocks;
		int dataset_begin = 0, dataset_end = 0 + step;

		int model_index = 0;
		int layer = this->level - 1;

		// for each continous block
		arma::mat current_trainingset = dataset.cols(dataset_begin, dataset_end);
		arma::rowvec current_response = labels.cols(dataset_begin, dataset_end);

		double current_accuracy, stable_accuracy;
		double current_absolute_accuracy, stable_absolute_accuracy;
		bool first_flag = true;
		int block_count = 0;
		LinearRegression lr_stable;

		// for the entire dataset, train the bottom layer!
		while (true) {

			first_flag = true;
			block_count = 1;

			// for an continous_block
			do {
				current_trainingset = dataset.cols(dataset_begin, dataset_end);
				current_response = labels.cols(dataset_begin, dataset_end);

				//cout << dataset_begin << " " << dataset_end << endl;

				LinearRegression lr_poineer(current_trainingset, current_response);
				
				// check finished?
				if (dataset_end == TOTAL_SIZE - 1) {
					stage_model_BU[layer].push_back(lr_stable);
					dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1)));
					break;
				}

				// Test the lr's accuracy
				current_accuracy = MeasureAccuracySingleLR(current_trainingset, current_response, lr_poineer);
				current_absolute_accuracy = MeasureAbsoluteAccuracySingleLR(current_trainingset, current_response, lr_poineer);

				// if the accuracy is below threshold, continous merge with the next block
				if (current_absolute_accuracy < threshold) {
					dataset_end += step; // judge border!!!
					if (dataset_end >= TOTAL_SIZE) {
						dataset_end = TOTAL_SIZE - 1;
					}
					first_flag = false;
					block_count++;
					lr_stable = lr_poineer;
					stable_accuracy = current_accuracy;
					stable_absolute_accuracy = current_absolute_accuracy;
					continue;
				}
				else if(first_flag == true) {// if the first_accuracy is already too large, start a new block, except it is the first block, if it's the first block, divide the region into half?
					// save the model
					lr_stable = lr_poineer;
					stage_model_BU[layer].push_back(lr_stable);
					//dataset_range[layer].push_back(pair<int, int>(dataset_begin, dataset_end));
					dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1)));
					stable_accuracy = current_accuracy;
					stable_absolute_accuracy = current_absolute_accuracy;
					break; // currently, just break;
				}
				else { // else if the second time accuracy is too large
					dataset_end -= step; // backward
					block_count--;
					// save the best range and model
					stage_model_BU[layer].push_back(lr_stable);
					//dataset_range[layer].push_back(pair<int, int>(dataset_begin, dataset_end));
					dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0,0), current_trainingset.at(0, current_trainingset.n_cols-1)));
					break;
				}

			} while (true);

			// reset the start and end for the next continous block(s)
			cout << model_index << " " << dataset_begin << " " << dataset_end << " " << stable_accuracy << " " << stable_absolute_accuracy << " " << block_count << endl;
			dataset_begin = dataset_end + 1;
			if (dataset_begin >= TOTAL_SIZE) {
				break; // meet the end
			}
			else {
				dataset_end = dataset_begin + step;
				if (dataset_end >= TOTAL_SIZE) {
					dataset_end = TOTAL_SIZE - 1;
				}
				model_index++;
			}
		}
	}

	// take input as key, model_id as labels
	void TrainNonLeafLayer() {

	}

	void BuildNonLeafLayerWithBtree() {
		bottom_layer_index.clear();
		int layer = this->level - 1; // bottom layer
		for (int i = 0; i < dataset_range[layer].size(); i++) {
			bottom_layer_index.insert(pair<double, int>(dataset_range[layer][i].second, i));
		}
	}

	void PredictWithLR() {
		// should we do sampling first?
	}

	void PredictWithStxBtree(vector<double> &queryset, vector<int> &results) {
		stx::btree<double, int>::iterator iter;
		results.clear();
		int model_index = 0;
		int layer = this->level - 1;
		double a, b;
		double result;
		for (int i = 0; i < queryset.size(); i++) {
			/*if (i == 19) {
				cout << "debug here!" << endl;
			}*/
			iter = this->bottom_layer_index.lower_bound(queryset[i]);
			model_index = iter->second;
			a = stage_model_parameter[layer][model_index][0];
			b = stage_model_parameter[layer][model_index][1];
			//cout << a << " " << b << " " << model_index << endl;
			result = a * queryset[i] + b;
			results.push_back(result);
		}
	}

	double inline MeasureAccuracySingleLR(arma::mat &testset, arma::rowvec &response, LinearRegression &lr) {
		 // do the predicition
		arma::rowvec prediction;
		lr.Predict(testset, prediction);

		arma::rowvec relative_error = abs(prediction - response);
		relative_error /= response;
		relative_error.elem(find_nonfinite(relative_error)).zeros(); // 'remove' ¡ÀInf or NaN
		return arma::mean(relative_error);
	}

	double inline MeasureAbsoluteAccuracySingleLR(arma::mat &testset, arma::rowvec &response, LinearRegression &lr) {
		// do the predicition
		arma::rowvec prediction;
		lr.Predict(testset, prediction);

		arma::rowvec absolute_error = abs(prediction - response);
		absolute_error.elem(find_nonfinite(absolute_error)).zeros(); // 'remove' ¡ÀInf or NaN
		return arma::mean(absolute_error);
	}

	void DumpParameter() {
		stage_model_parameter.clear();
		vector<vector<double>> stage_model_parameter_layer;
		vector<double> model;
		for (int k = 0; k < stage_model_BU.size(); k++) {
			stage_model_parameter_layer.clear();
			for (int i = 0; i < stage_model_BU[k].size(); i++) {
				model.clear();
				arma::vec paras = stage_model_BU[k][i].Parameters();
				//cout << i << " " << j << ": " << endl;
				//paras.print();
				model.push_back(paras[1]); // a
				model.push_back(paras[0]); // b
				stage_model_parameter_layer.push_back(model);
			}
			stage_model_parameter.push_back(stage_model_parameter_layer);
		}
	}

	int level;
	int initial_blocks;
	double threshold;

	vector<vector<LinearRegression>> stage_model_BU;
	vector<vector<pair<double, double>>> dataset_range;
	stx::btree<double, int> bottom_layer_index;

	vector<vector<vector<double>>> stage_model_parameter;
};