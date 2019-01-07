#pragma once
#include "mlpack/core.hpp"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <vector>

using namespace mlpack::regression;
using namespace std;

class StageModel {
 public:
	StageModel(const arma::mat& dataset, const arma::rowvec& labels, vector<int> &architecture) {
		int TOTAL_SIZE = dataset.size();
		this->TOTAL_SIZE = TOTAL_SIZE;

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

		// training and distribute training set
		vector<LinearRegression> stage_layer;
		for(int i = 0; i < architecture.size(); i++) {
			stage_layer.clear();
			if (i == 0) {
				LinearRegression lr(dataset, labels);
				stage_layer.push_back(lr);
				arma::rowvec predictions;
				lr.Predict(dataset, predictions);
				// distribute training set
				DistributeTrainingSetOptimized(predictions, stage_dataset[i][0], stage_label[i][0], i + 1, architecture[i + 1], TOTAL_SIZE);
			}
			else {
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
			}
			stage_model.push_back(stage_layer);
		 }
	}

	void InitQuerySet(const arma::mat& queryset) {

		stage_queryset.clear();
		vector<arma::mat> stage_queryset_layer;
		for (int i = 0; i < stage_model.size(); i++) {
			stage_queryset_layer.clear();
			for (int j = 0; j < stage_model[i].size(); j++) {
				arma::mat data;
				stage_queryset_layer.push_back(data);
			}
			stage_queryset.push_back(stage_queryset_layer);
		}

		int count = 0;
		rowvec order(queryset.n_cols);
		mat query;
		query = join_cols(queryset, order.imbue([&]() {return count++; })); // first row on top, second row on the bottom

		stage_queryset[0][0] = query;
	}

	// wait for test
	void Predict(const arma::mat& queryset, arma::rowvec& predictions) {

		predictions.clear();
		mat predictions_with_order;

		for (int i = 0; i < stage_model.size(); i++) {
			for (int j = 0; j < stage_model[i].size(); j++) {
				arma::rowvec predictions_temp;
				if (stage_queryset[i][j].n_cols != 0) {
					stage_model[i][j].Predict(stage_queryset[i][j].row(0), predictions_temp);
				}
				else {
					continue;
				}
				if (i < stage_model.size() - 1) {
					int target_layer_model_count = stage_model[i + 1].size();
					predictions_temp.for_each([=](mat::elem_type& val) {
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
					for (int k = 0; k < stage_model[i + 1].size(); k++) {
						uvec indices = find(predictions_temp == k);
						stage_queryset[i+1][k].insert_cols(stage_queryset[i+1][k].n_cols, stage_queryset[i][j].cols(indices)); // using index for dataset may be a bug
					}
				}
				else {
					// save and concat result set.
					mat temp = join_cols(predictions_temp, stage_queryset[i][j].row(1)); // first row on top, second row on the bottom
					predictions_with_order = join_rows(predictions_with_order, temp);
				}
			}
		}

		uvec index = sort_index(predictions_with_order.row(1), "ascend");
		//predictions = predictions_with_order.cols(index);
		predictions = predictions_with_order.row(0);
		predictions = predictions.cols(index);
	}

	// this method is too slow!!!
	void PredictNaive(arma::mat& queryset, arma::rowvec& predictions) {

		arma::rowvec temp;
		for (int item = 0; item < queryset.size(); item++) {
			int index = 0;
			for (int i = 0; i < stage_model.size(); i++) {
				stage_model[i][index].Predict(queryset.col(i), temp);
				if (i < stage_model.size() - 1) {
					index = temp[0] / TOTAL_SIZE * stage_model[i + 1].size();
					index < 0 ? 0 : index;
					index >= stage_model[i + 1].size() - 1 ? stage_model[i + 1].size() - 1 : index;
				}
				predictions.insert_cols(i, temp[0]);
			}
		}
	}

	void DistributeTrainingSet(arma::rowvec &predictions, arma::mat& dataset, arma::rowvec& labels, int target_layer, int target_layer_model_count, int TOTAL_SIZE) {
		
		predictions /= TOTAL_SIZE;
		predictions *= target_layer_model_count;
		int model_id = 0;
		for (int i = 0; i < predictions.size(); i++) {
			model_id = int(predictions[i]);
			if (model_id < 0) {
				model_id = 0;
			}
			else if (model_id >= target_layer_model_count) {
				model_id = target_layer_model_count - 1;
			}
			stage_dataset[target_layer][model_id].insert_cols(stage_dataset[target_layer][model_id].n_cols, dataset.col(i));
			stage_label[target_layer][model_id].insert_cols(stage_label[target_layer][model_id].n_cols, labels.col(i));
			if (i % 10000 == 0) {
				cout << "distributing..." << i << endl;
			}
		}
	}

	void DistributeTrainingSetOptimized(arma::rowvec &predictions, arma::mat& dataset, arma::rowvec& labels, int target_layer, int target_layer_model_count, int TOTAL_SIZE) {

		predictions.for_each([=](mat::elem_type& val) {
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
			uvec indices = find(predictions == j);
			cout << "distributing...target_layer=" << target_layer << " model=" << j << endl;
			stage_dataset[target_layer][j].insert_cols(stage_dataset[target_layer][j].n_cols, dataset.cols(indices)); // using index for dataset may be a bug
			stage_label[target_layer][j].insert_cols(stage_label[target_layer][j].n_cols, labels.cols(indices));
		}
	}

	double MeasureAccuracy(arma::rowvec predicted_range, string filename_result="C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResults.csv") {
		mat real_result;
		bool loaded3 = mlpack::data::Load(filename_result, real_result);
		arma::rowvec real_range = real_result.row(0);

		arma::rowvec relative_error = abs(predicted_range - real_range);
		relative_error /= real_range;
		double total_error = arma::accu(relative_error);
		double average_relative_error = total_error / relative_error.size();
		cout << "average error: " << average_relative_error << endl;
		return average_relative_error;
	}

	int TOTAL_SIZE = 1;
	vector<vector<LinearRegression>> stage_model;
	vector<vector<arma::mat>> stage_dataset;
	vector<vector<arma::rowvec>> stage_label;

	vector<vector<arma::mat>> stage_queryset;
};