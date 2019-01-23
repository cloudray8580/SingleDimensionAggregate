#pragma once

#pragma once

#include "mlpack/core.hpp"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <vector>

using namespace std;
using namespace mlpack::regression;

class StageModel2D_2 {
public:

	StageModel2D_2(vector<pair<int, int>> &architecture) {
		this->architecture = architecture;
	}

	// for count
	// dataset are the sampled dataset!
	void Train(const arma::mat& dataset, const arma::rowvec& labels) {
		TOTAL_SIZE = dataset.size();

		// initialize stage_dataset and stage_label
		vector<arma::mat> stage_dataset_layer_y;
		vector<arma::rowvec> stage_label_layer_y;
		vector<vector<arma::mat>> stage_dataset_layer_xy;
		vector<vector<arma::rowvec>> stage_label_layer_xy;

		for (int k = 0; k < architecture.size(); k++) {
			stage_dataset_layer_xy.clear();
			stage_label_layer_xy.clear();
			for (int i = 0; i < architecture[k].first; i++) {
				stage_dataset_layer_y.clear();
				stage_label_layer_y.clear();
				for (int j = 0; j < architecture[k].second; j++) {
					arma::mat data;
					arma::rowvec label;
					stage_dataset_layer_y.push_back(data);
					stage_label_layer_y.push_back(label);
				}
				stage_dataset_layer_xy.push_back(stage_dataset_layer_y);
				stage_label_layer_xy.push_back(stage_label_layer_y);
			}
			stage_dataset.push_back(stage_dataset_layer_xy);
			stage_label.push_back(stage_label_layer_xy);
		}
		stage_dataset[0][0][0] = dataset;
		stage_label[0][0][0] = labels;

		vector<LinearRegression> stage_layer_y;
		vector<vector<LinearRegression>> stage_layer_xy;
		for (int k = 0; k < architecture.size(); k++) {
			stage_layer_xy.clear();
			for (int i = 0; i < architecture[k].first; i++) {
				stage_layer_y.clear();
				for (int j = 0; j < architecture[k].second; j++) {
					LinearRegression lr(stage_dataset[k][i][j], stage_label[k][i][j]);
					stage_layer_y.push_back(lr);
					arma::rowvec predictions;
					lr.Predict(stage_dataset[k][i][j], predictions);
					if (k != architecture.size() - 1) {
						// distribute training set
						DistributeTrainingSetOptimized(predictions, stage_dataset[k][i][j], stage_label[k][i][j], k + 1, architecture[k + 1].first, architecture[k + 1].second, TOTAL_SIZE);
					}
				}
				stage_layer_xy.push_back(stage_layer_y);
			}
			stage_model.push_back(stage_layer_xy);
		}

		this->DumpParameters();
	}

	void DistributeTrainingSetOptimized(arma::rowvec &predictions, arma::mat& dataset, arma::rowvec& labels, int target_layer, int target_layer_model_count_x, int target_layer_model_count_y, int TOTAL_SIZE) {

		arma::rowvec predictions_y = predictions;

		predictions.for_each([=](arma::mat::elem_type &val) {
			val /= TOTAL_SIZE;
			val *= target_layer_model_count_x;
			val = int(val);
			if (val < 0) {
				val = 0;
			}
			else if (val >= target_layer_model_count_x) {
				val = target_layer_model_count_x - 1;
			}
		});

		predictions_y.for_each([=](arma::mat::elem_type &val) {
			val /= TOTAL_SIZE;
			val *= target_layer_model_count_y;
			val = int(val);
			if (val < 0) {
				val = 0;
			}
			else if (val >= target_layer_model_count_y) {
				val = target_layer_model_count_y - 1;
			}
		});


		for (int i = 0; i < stage_dataset[target_layer].size(); i++) {
			//cout << stage_dataset[target_layer].size() << endl;
			for (int j = 0; j < stage_dataset[target_layer][0].size(); j++) {
				//cout << stage_dataset[target_layer][0].size() << endl;
				arma::uvec indices = find(predictions == i && predictions_y == j);
				//cout << "distributing...target_layer=" << target_layer << " model=" << j << endl;
				stage_dataset[target_layer][i][j].insert_cols(stage_dataset[target_layer][i][j].n_cols, dataset.cols(indices)); // using index for dataset may be a bug
				stage_label[target_layer][i][j].insert_cols(stage_label[target_layer][i][j].n_cols, labels.cols(indices));
			}
		}
	}

	void DumpParameters() {
		stage_model_parameters.clear();

		vector<vector<vector<double>>> layer_xy;
		vector<vector<double>> layer_y;
		vector<double> model;

		for (int k = 0; k < stage_model.size(); k++) {
			layer_xy.clear();
			for (int i = 0; i < stage_model[k].size(); i++) {
				layer_y.clear();
				for (int j = 0; j < stage_model[k][i].size(); j++) {
					model.clear();
					arma::vec paras = stage_model[k][i][j].Parameters();
					//cout << i << " " << j << ": " << endl;
					//paras.print();
					model.push_back(paras[1]); // a
					model.push_back(paras[2]); // b
					model.push_back(paras[0]); // c
					layer_y.push_back(model);
				}
				layer_xy.push_back(layer_y);
			}
			stage_model_parameters.push_back(layer_xy);
		}
	}

	void PredictVector(vector<double> &queryset_x, vector<double> &queryset_y, vector<double> &results) {
		double result;
		results.clear();
		double a, b, c;
		int index_x = 0, index_y = 0;
		for (int n = 0; n < queryset_x.size(); n++) {
			index_x = 0;
			index_y = 0;
			for (int k = 0; k < stage_model_parameters.size(); k++) {
				a = stage_model_parameters[k][index_x][index_y][0];
				b = stage_model_parameters[k][index_x][index_y][1];
				c = stage_model_parameters[k][index_x][index_y][2];
				result = a * queryset_x[n] + b * queryset_y[n] + c;
				if (k < stage_model_parameters.size() - 1) {
					index_x = (result / TOTAL_SIZE * stage_model_parameters[k + 1].size());
					index_y = (result / TOTAL_SIZE * stage_model_parameters[k + 1][0].size());
					if (index_x < 0) {
						index_x = 0;
					}
					else if (index_x > stage_model_parameters[k + 1].size() - 1) {
						index_x = stage_model_parameters[k + 1].size() - 1;
					}
					if (index_y < 0) {
						index_y = 0;
					}
					else if (index_y > stage_model_parameters[k + 1][0].size() - 1) {
						index_y = stage_model_parameters[k + 1][0].size() - 1;
					}
				}
			}
			results.push_back(result);
		}
	}

	int TOTAL_SIZE;
	vector<pair<int, int>> architecture; // level; x_models, y_models

	vector<vector<vector<LinearRegression>>> stage_model; // level, x_pos, y_pos
	vector<vector<vector<arma::mat>>> stage_dataset;  // level, x_pos, y_pos
	vector<vector<vector<arma::rowvec>>> stage_label; // level, x_pos, y_pos
	vector<vector<vector<vector<double>>>> stage_model_parameters; // level; x_pos; y_pos; Parameter: a,b,c or a,b
};