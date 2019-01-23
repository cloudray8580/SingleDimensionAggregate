#pragma once

#include "mlpack/core.hpp"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <vector>

using namespace std;
using namespace mlpack::regression;

class StageModel2D {
public:

	StageModel2D(vector<int> &architecture) {
		this->architecture = architecture;
	}

	// for count
	// dataset are the sampled dataset!
	void Train(const arma::mat& dataset, const arma::rowvec& labels) {
		TOTAL_SIZE = dataset.size();

		// initialize stage_dataset and stage_label
		vector<arma::mat> stage_dataset_layer;
		vector<arma::rowvec> stage_label_layer;
		for (int i = 0; i < architecture.size(); i++) {
			stage_dataset_layer.clear();
			stage_label_layer.clear();
			for (int j = 0; j < architecture[i]; j++) {
				arma::mat data;
				arma::rowvec label;
				stage_dataset_layer.push_back(data);
				stage_label_layer.push_back(label);
			}
			stage_dataset.push_back(stage_dataset_layer);
			stage_label.push_back(stage_label_layer);
		}
		stage_dataset[0][0] = dataset;
		stage_label[0][0] = labels;

		vector<LinearRegression> stage_layer;
		for (int i = 0; i < architecture.size(); i++) {
			stage_layer.clear();
			for (int j = 0; j < architecture[i]; j++) {
				LinearRegression lr(stage_dataset[i][j], stage_label[i][j]);
				stage_layer.push_back(lr);
				arma::rowvec predictions;
				lr.Predict(stage_dataset[i][j], predictions);
				if (i != architecture.size() - 1) {
					// distribute training set
					DistributeTrainingSetOptimized(predictions, stage_dataset[i][j], stage_label[i][j], i + 1, architecture[i + 1], TOTAL_SIZE);
				}
			}
			stage_model.push_back(stage_layer);
		}

		this->DumpParameters();
	}

	void DistributeTrainingSetOptimized(arma::rowvec &predictions, arma::mat& dataset, arma::rowvec& labels, int target_layer, int target_layer_model_count, int TOTAL_SIZE) {

		predictions.for_each([=](arma::mat::elem_type &val) {
			val /= TOTAL_SIZE;
			val *= target_layer_model_count;
			val = int(val);
			if (val < 0) {
				val = 0;
			}
			else if (val >= target_layer_model_count) {
				val = target_layer_model_count - 1;
			}
		});

		for (int j = 0; j < stage_dataset[target_layer].size(); j++) {
			arma::uvec indices = find(predictions == j);
			//cout << "distributing...target_layer=" << target_layer << " model=" << j << endl;
			stage_dataset[target_layer][j].insert_cols(stage_dataset[target_layer][j].n_cols, dataset.cols(indices)); // using index for dataset may be a bug
			stage_label[target_layer][j].insert_cols(stage_label[target_layer][j].n_cols, labels.cols(indices));
		}
	}

	void DumpParameters() {
		stage_model_parameters.clear();
		vector<vector<double>> layer;
		vector<double> model;
		for (int i = 0; i < stage_model.size(); i++) {
			layer.clear();
			for (int j = 0; j < stage_model[i].size(); j++) {
				model.clear();
				arma::vec paras = stage_model[i][j].Parameters();
				//cout << i << " " << j << ": " << endl;
				//paras.print();
				model.push_back(paras[1]); // a
				model.push_back(paras[2]); // b
				model.push_back(paras[0]); // c

				layer.push_back(model);
			}
			stage_model_parameters.push_back(layer);
		}
	}

	void PredictVector(vector<double> &queryset_x, vector<double> &queryset_y, vector<double> &results) {
		double result;
		results.clear();
		double a, b, c;
		int index = 0;
		for (int k = 0; k < queryset_x.size(); k++) {
			index = 0;
			for (int i = 0; i < stage_model_parameters.size(); i++) {
				a = stage_model_parameters[i][index][0];
				b = stage_model_parameters[i][index][1];
				c = stage_model_parameters[i][index][2];
				result = a * queryset_x[k] + b * queryset_y[k] + c;
				if (i < stage_model_parameters.size() - 1) {
					index = (result / TOTAL_SIZE * stage_model_parameters[i + 1].size());
					if (index < 0) {
						index = 0;
					}
					else if (index > stage_model_parameters[i + 1].size() - 1) {
						index = stage_model_parameters[i + 1].size() - 1;
					}
				}
			}
			results.push_back(result);
		}
	}

	int TOTAL_SIZE;
	vector<int> architecture;
	vector<vector<LinearRegression>> stage_model;
	vector<vector<arma::mat>> stage_dataset;
	vector<vector<arma::rowvec>> stage_label;
	vector<vector<vector<double>>> stage_model_parameters; // level; model; Parameter: a,b,c
};