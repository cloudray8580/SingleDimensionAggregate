#pragma once

//#define __ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES 0
#include "STXBtreeAggregate.h"
#include "mlpack/core.hpp"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <vector>

using namespace std;
using namespace mlpack::regression;

// descending
bool cmp_magnitude(pair<int, double> &p1, pair<int, double> &p2) {
	return p1.second > p2.second;
}

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

	// do not need to dump parameter for the bottom layer anymore
	void LoadBottomModelFromFile(string dataset_file) {
		
		mat dataset;
		bool loaded = mlpack::data::Load(dataset_file, dataset);

		arma::rowvec start_keys = dataset.row(2); // the third column, start key
		arma::rowvec end_keys = dataset.row(3); // the fourth column, end key
		arma::rowvec weight = dataset.row(4); // the fifth column, weight
		arma::rowvec bias = dataset.row(5); // the sixth column, bias
		
		int layer = this->level - 1;
		vector<double> key1_v, key2_v, weight_v, bias_v;

		RowvecToVector(start_keys, key1_v);
		RowvecToVector(end_keys, key2_v);
		RowvecToVector(weight, weight_v);
		RowvecToVector(bias, bias_v);

		vector<vector<double>> stage_model_parameter_layer;
		vector<double> model;

		for (int i = 0; i < key1_v.size(); i++) {
			dataset_range[layer].push_back(pair<double, double>(key1_v[i],key2_v[i]));
			model.clear();
			model.push_back(weight[i]); // a
			model.push_back(bias[i]); // b
			stage_model_parameter_layer.push_back(model);
		}
		stage_model_parameter[level-1] = stage_model_parameter_layer;
	}

	void BuildWithSegmentation() {

	}

	void ApproximateMaxLossLinearRegression(int origin_index, int shift_index, vector<double> &key_v, vector<double> &position_v, const arma::mat& current_trainingset, const arma::rowvec&current_response, double sampling_control, double error_threshold, double sampling_percentage, double &slope, double &intercept, double &maxloss) {

		auto t0 = chrono::steady_clock::now();

		// 1. first train an linear regression as the axes
		LinearRegression lr(current_trainingset, current_response);

		// 2. calculate the candidate max error points
		arma::vec paras = lr.Parameters();
		double a, b;
		a = paras[1]; // a
		b = paras[0]; // b

		// lower the LR to make the dataset above the line to ease the following calculation
		double signed_error = 0, max_negative_error = 0;
		double predicted_position;
		for (int i = origin_index; i <= shift_index; i++) {
			predicted_position = a * key_v[i] + b;
			signed_error = position_v[i] - predicted_position;
			if (signed_error < max_negative_error) {
				max_negative_error = signed_error;
				//cout << i << " " << max_negative_error << " " << signed_error << endl;
			}
		}
		b += max_negative_error;

		double local_maximum = 0, local_minimum = shift_index - origin_index;
		double pre_local_maximum = 0, pre_local_minimum = 0;
		int local_maximum_pos = 0, local_minimum_pos = 0;
		bool max_min_ready_flag = false, min_max_ready_flag = false;

		// scan the dataset to find those significant turning points
		int segment_size = shift_index - origin_index + 1;
		double error = 0;
		vector<pair<int, double>> segmented_pos; // record the segmentation points
		for (int i = 0; i < segment_size; i++) {
			predicted_position = a * key_v[i + origin_index] + b;
			error = position_v[origin_index + i] - predicted_position;
			if (i == 0) {
				pre_local_maximum = error;
				pre_local_minimum = error;
			}
			if (error > local_maximum) {
				local_maximum = error;
				local_maximum_pos = i;
				if (!min_max_ready_flag && local_maximum - local_minimum > sampling_control * error_threshold) {
					min_max_ready_flag = true;
				}
			}
			// when a draw make the turn very sharp, record the local minimum point
			if (max_min_ready_flag && error - local_minimum > sampling_control * error_threshold) {
				segmented_pos.push_back(pair<int, double>(origin_index + local_minimum_pos, pre_local_maximum - local_minimum));
				max_min_ready_flag = false;
				local_maximum = error;
				local_maximum_pos = i;
				pre_local_minimum = local_minimum;
			}
			if (error < local_minimum) {
				local_minimum = error;
				local_minimum_pos = i;
				if (!max_min_ready_flag && local_maximum - local_minimum > sampling_control * error_threshold) {
					max_min_ready_flag = true;
				}
			}
			// when a drop make the turn very sharp, record the local maximum point
			if (min_max_ready_flag && local_maximum - error > sampling_control * error_threshold) {
				segmented_pos.push_back(pair<int, double>(origin_index + local_maximum_pos, local_maximum - pre_local_minimum));
				min_max_ready_flag = false;
				local_minimum = error;
				local_minimum_pos = i;
				pre_local_maximum = local_maximum;
			}
		}

		// sort these segmented position according to its scope		
		std::sort(segmented_pos.begin(), segmented_pos.end(), cmp_magnitude);

		// 3. extract a percent of them
		int extracted_amount = segmented_pos.size() * sampling_percentage;
		vector<arma::u64> selected_index;
		for (int i = 0; i < extracted_amount; i++) {
			selected_index.push_back(segmented_pos[i].first);
		}
		selected_index.push_back(0);
		selected_index.push_back(segment_size - 1);
		arma::uvec selected_index_arma(selected_index);

		// 4. train an linear regression based on these points
		arma::mat selected_points_key = current_trainingset.cols(selected_index_arma);
		arma::rowvec selected_points_pos = current_response.cols(selected_index_arma);

		// 5. adjust the bias term to balance the error
		LinearRegression lr_maxloss(selected_points_key, selected_points_pos);
		paras = lr_maxloss.Parameters();
		a = paras[1]; // a
		b = paras[0]; // b

		// find the max positive error and the max negative error
		double max_pos_err = 0, max_neg_err = 0;
		for (int i = 0; i < segment_size; i++) {
			predicted_position = a * key_v[i + origin_index] + b;
			error = position_v[origin_index + i] - predicted_position;
			if (error > max_pos_err) {
				max_pos_err = error;
			}
			else if (error < max_neg_err) {
				max_neg_err = error;
			}	
		}
		double adjusted_b = b + (max_pos_err - abs(max_neg_err)) / 2;
		double maxerr = (max_pos_err - max_neg_err) / 2;
		//cout << "max error for approx max loss:" << maxerr << endl;

		slope = a;
		intercept = adjusted_b;
		maxloss = maxerr;

		//auto t1 = chrono::steady_clock::now();
		//cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	}

	// do not need to dump parameter for the bottom layer anymore
	void TrainBottomLayerWithFastDetectForMaxLoss(const arma::mat& dataset, const arma::rowvec& labels, int step = 1000, double sampling_control=2, double sampling_percentage=0.1) {
		// Todo
		int layer = this->level - 1;

		vector<double> key_v, position_v;
		RowvecToVector(dataset, key_v);
		RowvecToVector(labels, position_v);

		int TOTAL_SIZE = dataset.n_cols; // 1157570

		int origin_index = 0;
		int shift_index = origin_index + step;
		int pre_origin_index = 0, pre_shift_index = shift_index;

		bool lr_correction_tag = false;
		double current_absolute_accuracy;
		int detected_points = 0;

		pair<double, double> temp_dataset_range;
		vector<double> model, temp_model;
		vector<vector<double>> stage_model_parameter_layer;
		double a, b; // slope and bias

		while (origin_index < TOTAL_SIZE) {

			//cout << stage_model_BU[layer].size() << " " << origin_index << " " << shift_index << " " << step << endl;

			if (shift_index >= TOTAL_SIZE) {
				shift_index = TOTAL_SIZE - 1;
			}

			arma::mat current_trainingset = dataset.cols(origin_index, shift_index); // included [origin_index, shift_index]
			arma::rowvec current_response = labels.cols(origin_index, shift_index);

			//LinearRegression lr_poineer(current_trainingset, current_response);

			ApproximateMaxLossLinearRegression(origin_index, shift_index, key_v, position_v, current_trainingset, current_response, sampling_control, threshold, sampling_percentage, a, b, current_absolute_accuracy);
			model.clear();
			model.push_back(a);
			model.push_back(b);

			if (current_absolute_accuracy > threshold) {
				if (lr_correction_tag) {
					// save the previous model
					stage_model_parameter_layer.push_back(temp_model);
					dataset_range[layer].push_back(temp_dataset_range);

					// roll back index
					origin_index = pre_origin_index;
					shift_index = pre_shift_index;

					// update step
					step = shift_index - origin_index;
					step *= 2;

					// prepare for the next segment
					origin_index = shift_index + 1; // verify whether + 1 is necessary here !
					shift_index = origin_index + step;
					lr_correction_tag = false;
				}
				else {
					// shink step
					step /= 2;
					shift_index = origin_index + step;
					if (shift_index >= TOTAL_SIZE) {
						shift_index = TOTAL_SIZE - 1;
					}
				}
			}
			else {
				detected_points = 0;
				// start the fast detect algorithm
				double predicted_pos;
				while (shift_index < TOTAL_SIZE - 1) {
					predicted_pos = a * key_v[shift_index] + b;
					if (position_v[shift_index] <= predicted_pos + threshold && position_v[shift_index] >= predicted_pos - threshold) {
						shift_index += 1;
						detected_points += 1;
					}
					else {
						break;
					}
				}
				if (detected_points > 0) {
					// do regression correction
					lr_correction_tag = true;
					temp_dataset_range = pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1));
					temp_model.clear();
					temp_model = model;
					pre_origin_index = origin_index;
					pre_shift_index = shift_index;
					continue;
				}
				else {
					// no more points, save the LR
					model.clear();
					model.push_back(a);
					model.push_back(b);
					stage_model_parameter_layer.push_back(model);
					dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1)));
					// update the step
					step = shift_index - origin_index;
					step *= 2;
					// prepare for the next segment
					origin_index = shift_index + 1; // verify whether + 1 is necessary here !
					shift_index = origin_index + step;
					lr_correction_tag = false;
				}
			}
		}
		stage_model_parameter[level - 1] = stage_model_parameter_layer;
	}

	// need to dump parameter for the bottom layer
	void TrainBottomLayerWithFastDetectOptimized(const arma::mat& dataset, const arma::rowvec& labels, int step = 1000) {
		// Todo
		int layer = this->level - 1;

		vector<double> key_v, position_v;
		RowvecToVector(dataset, key_v);
		RowvecToVector(labels, position_v);

		int TOTAL_SIZE = dataset.n_cols; // 1157570

		int origin_index = 0;
		int shift_index = origin_index + step;
		int pre_origin_index = 0, pre_shift_index = shift_index;

		bool lr_correction_tag = false;
		double current_absolute_accuracy;
		int detected_points = 0;

		pair<double, double> temp_dataset_range;
		LinearRegression temp_lr;

		while (origin_index < TOTAL_SIZE) {

			//cout << stage_model_BU[layer].size() << " " << origin_index << " " << shift_index << " " << step << endl;

			if (shift_index >= TOTAL_SIZE) {
				shift_index = TOTAL_SIZE - 1;
			}

			arma::mat current_trainingset = dataset.cols(origin_index, shift_index); // included [origin_index, shift_index]
			arma::rowvec current_response = labels.cols(origin_index, shift_index);

			LinearRegression lr_poineer(current_trainingset, current_response);
			current_absolute_accuracy = MeasureMaxAbsoluteError(current_trainingset, current_response, lr_poineer);

			if (current_absolute_accuracy > threshold) {
				if (lr_correction_tag) {
					// save the previous model
					stage_model_BU[layer].push_back(temp_lr);
					dataset_range[layer].push_back(temp_dataset_range);

					// roll back index
					origin_index = pre_origin_index;
					shift_index = pre_shift_index;

					// update step
					step = shift_index - origin_index;
					step *= 2;

					// prepare for the next segment
					origin_index = shift_index + 1; // verify whether + 1 is necessary here !
					shift_index = origin_index + step;
					lr_correction_tag = false;
				}
				else {
					// shink step
					step /= 2;
					shift_index = origin_index + step;
					if (shift_index >= TOTAL_SIZE) {
						shift_index = TOTAL_SIZE - 1;
					}
				}
			}
			else {
				detected_points = 0;
				// start the fast detect algorithm
				arma::vec paras = lr_poineer.Parameters();
				double a, b;
				a = paras[1]; // a
				b = paras[0]; // b
				double predicted_pos;
				while (shift_index < TOTAL_SIZE-1) {
					predicted_pos = a * key_v[shift_index] + b;
					if (position_v[shift_index] <= predicted_pos + threshold && position_v[shift_index] >= predicted_pos - threshold) {
						shift_index += 1;
						detected_points += 1;
					}
					else {
						break;
					}
				}
				if (detected_points > 0) {
					// do regression correction
					lr_correction_tag = true;
					temp_dataset_range = pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1));
					temp_lr = lr_poineer;
					pre_origin_index = origin_index;
					pre_shift_index = shift_index;
					continue;
				}
				else {
					// no more points, save the LR
					stage_model_BU[layer].push_back(lr_poineer);
					dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1)));
					// update the step
					step = shift_index - origin_index;
					step *= 2;
					// prepare for the next segment
					origin_index = shift_index + 1; // verify whether + 1 is necessary here !
					shift_index = origin_index + step;
					lr_correction_tag = false;
				}
			}
		}
	}

	void TrainBottomLayerWithFastDetect(const arma::mat& dataset, const arma::rowvec& labels) {
		// train an linear regression with step length

		// if error less than error_threshold, start fast detect with double error bound
		//     if point outside error bound, retrain the linear regression keep fast detect
		// else, half the step and try again

		int layer = this->level - 1;

		vector<double> key_v, position_v;
		RowvecToVector(dataset, key_v);
		RowvecToVector(labels, position_v);

		int TOTAL_SIZE = dataset.n_cols;
		int initial_step = TOTAL_SIZE / initial_blocks;
		int step = initial_step; // the initial block
		int last_stop_shift_index;

		int origin_index = 0, shift_index = 0 + step;

		double current_accuracy, stable_accuracy;
		double current_absolute_accuracy, stable_absolute_accuracy;

		bool firsttag = true;

		while (origin_index < TOTAL_SIZE) {

			cout << stage_model_BU[layer].size() << " " << origin_index << " " << shift_index << " " << step << endl;

			if (shift_index == 439474) {
				cout << "debug here" << endl;
			}

			arma::mat current_trainingset = dataset.cols(origin_index, shift_index);
			arma::rowvec current_response = labels.cols(origin_index, shift_index);

  			LinearRegression lr_poineer(current_trainingset, current_response);
			 
			//current_absolute_accuracy = MeasureAbsoluteAccuracySingleLR(current_trainingset, current_response, lr_poineer);
			current_absolute_accuracy = MeasureMaxAbsoluteError(current_trainingset, current_response, lr_poineer);
			/*if (current_absolute_accuracy > 2 * threshold) {
				cout << "2 error warning !" << endl;
			}*/

			//test if the new generated lr is within the original error band region

			last_stop_shift_index = shift_index;

			if (current_absolute_accuracy < threshold) {
				if (firsttag && step < INT_MAX/2) {
					//step *= 2; // modification !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
				}
				firsttag = false;
				arma::vec paras = lr_poineer.Parameters();
				double a, b;
				a = paras[1]; // a
				b = paras[0]; // b

				double predicted_pos;
				while (shift_index < TOTAL_SIZE) {
					shift_index++;
					predicted_pos = a * key_v[shift_index] + b;
					if (position_v[shift_index] < predicted_pos + threshold && position_v[shift_index] > predicted_pos - threshold) {
						continue;
					}
					else {
						// 1. save the current model and start the next segement
						// 2. retrain the model using the origin and current dataset, then start the next segement
						// 3. retrain the model using the origin and current dataset and start fast detect again, until the training accuracy is less than threshold.

						// if the shift_index == the following point of the last shift_index, start a new segement
						if (shift_index == last_stop_shift_index + 1){
							// start a new segement
							stage_model_BU[layer].push_back(lr_poineer);
							dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1)));
							// reset control parameters

							step = (shift_index - origin_index) * 2; // modification !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

							firsttag = true;
							origin_index = shift_index + 1;
							//cout << "origin_index: " << origin_index << " shift_index:" << shift_index << " step:" << step << endl;
							shift_index = origin_index + step;
							if (shift_index >= TOTAL_SIZE - 1) {
								shift_index = TOTAL_SIZE - 1;
							}
							break;
						}
						else {
							// I choose 3
							shift_index -= 1;
							break;  
						}
					}
				}
			}
			else if (firsttag) {
				step /= 2;
				shift_index = origin_index + step;
				if (shift_index >= TOTAL_SIZE - 1) {
					shift_index = TOTAL_SIZE - 1;
				}
				continue;
			} else {
				// there is a bug in this version, use the optimized version. (the final corrected version may exceeds the error threshold without changing the data to its previous)

				// no first time, save the current model and start the next segement, but this branch may not happen // no, it will happen
				//cout << "current_absolute_accuracy: " << current_absolute_accuracy << endl;
				stage_model_BU[layer].push_back(lr_poineer);
				dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1)));
				// reset control parameters
				firsttag = true;

				step = (shift_index - origin_index) * 2; // modification !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

				origin_index = shift_index + 1;
				shift_index = origin_index + step;
				if (shift_index >= TOTAL_SIZE - 1) {
					shift_index = TOTAL_SIZE - 1;
				}
			}
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
				//current_accuracy = MeasureAccuracySingleLR(current_trainingset, current_response, lr_poineer);
				//current_absolute_accuracy = MeasureAbsoluteAccuracySingleLR(current_trainingset, current_response, lr_poineer);
				current_absolute_accuracy = MeasureMaxAbsoluteError(current_trainingset, current_response, lr_poineer);

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

	//==========================================================================================================

	// in such method, dump parameter should only works in the bottom layer.
	void BuildAtreeForNonLeafIndex(int error_threshold = 1) {

		double slope_high, slope_low;

		int current_index = 0;
		int cone_origin_index = 0;
		int cone_shift_index = 0;

		double upper_pos, lower_pos;

		double current_slope;
		bool exit_tag = false;

		int layer = this->level - 2;

		// build each layer from bottom to top
		while (true) {

			vector<double> key_v, model_v;
			for (int i = 0; i < dataset_range[layer+1].size(); i++) {
				key_v.push_back(dataset_range[layer+1][i].second);
				model_v.push_back(i);
			}
			int TOTAL_SIZE_MODEL = dataset_range[layer+1].size();

			cout << "layer: " << layer << endl;
			cout << "TOTAL_SIZE_MODEL: " << TOTAL_SIZE_MODEL << endl;

			vector<double> parms;
			cone_origin_index = 0;
			cone_shift_index = 0;

			// while the dataset is not empty, build segements
			while (cone_origin_index < TOTAL_SIZE_MODEL) {

				// build the cone with first and second point
				cone_shift_index = cone_origin_index + 1;

				if (cone_shift_index >= TOTAL_SIZE_MODEL) {
					dataset_range[layer].push_back(pair<double, double>(key_v[cone_origin_index], key_v[cone_shift_index]));
					break;
				}

				//cout << position_v[cone_shift_index] << " " << position_v[cone_origin_index] << " " << key_v[cone_shift_index] << " " << key_v[cone_origin_index] << endl;

				while (key_v[cone_shift_index] == key_v[cone_origin_index]) {
					cone_shift_index++;
				}

				//cout << position_v[cone_shift_index] << " " << position_v[cone_origin_index] << " " << key_v[cone_shift_index] << " " << key_v[cone_origin_index] << endl;

				slope_high = (model_v[cone_shift_index] + error_threshold - model_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
				slope_low = (model_v[cone_shift_index] - error_threshold - model_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);

				//cout << "slope: " << slope_high << " " << slope_low << endl;

				exit_tag = false;

				// test if the following points are in the cone
				while (++cone_shift_index < TOTAL_SIZE_MODEL) {
					//cone_shift_index++;

					// test if exceed the border, if the first time, if the last segement

					upper_pos = slope_high * (key_v[cone_shift_index] - key_v[cone_origin_index]) + model_v[cone_origin_index];
					lower_pos = slope_low * (key_v[cone_shift_index] - key_v[cone_origin_index]) + model_v[cone_origin_index];

					if (model_v[cone_shift_index] < upper_pos && model_v[cone_shift_index] > lower_pos) {
						// inside the conde, update the slope
						if (model_v[cone_shift_index] + error_threshold < upper_pos) {
							// update slope_high
							slope_high = (model_v[cone_shift_index] + error_threshold - model_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
						}
						if (model_v[cone_shift_index] - error_threshold > lower_pos) {
							// update slope_low
							slope_low = (model_v[cone_shift_index] - error_threshold - model_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
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
					dataset_range[layer].push_back(pair<double, double>(key_v[cone_origin_index], key_v[cone_shift_index-1]));
					current_slope = (model_v[cone_shift_index - 1] - model_v[cone_origin_index]) / (key_v[cone_shift_index - 1] - key_v[cone_origin_index]); // monotonous

					parms.clear();
					parms.push_back(current_slope);
					parms.push_back(model_v[cone_origin_index]);
					stage_model_parameter[layer].push_back(parms);

					cout << cone_origin_index << " " << cone_shift_index - 1 << " " << current_slope << " " << model_v[cone_origin_index] << endl;
					cone_origin_index = cone_shift_index;
				}
				else {
					dataset_range[layer].push_back(pair<double, double>(key_v[cone_origin_index], key_v[cone_shift_index-1])); // exit loop as the border is met
					current_slope = (model_v[cone_shift_index-1] - model_v[cone_origin_index]) / (key_v[cone_shift_index-1] - key_v[cone_origin_index]); // monotonous

					parms.clear();
					parms.push_back(current_slope);
					parms.push_back(model_v[cone_origin_index]);
					stage_model_parameter[layer].push_back(parms);

					cout << cone_origin_index << " " << cone_shift_index-1 << " " << current_slope << " " << model_v[cone_origin_index] << endl;

					break; // all the data set has been scanned through
				}
			}

			if (dataset_range[layer].size() == 1) {
				this->start_layer = layer;
				break;
			}
			else {
				layer--;
			}
		}	
	}

	// build top-down model (stage model) for non-leaf layer
	void BuildTopDownModelForNonLeafLayer() {
		vector<int> architecture;
		architecture.push_back(1);
		architecture.push_back(10);
		architecture.push_back(100);
		architecture.push_back(1000);

		vector<double> key_v, model_v;
		int layer = this->level - 1; // bottom layer
		for (int i = 0; i < dataset_range[layer].size(); i++) {
			key_v.push_back(dataset_range[layer][i].second);
			model_v.push_back(i);
		}
		int TOTAL_SIZE_MODEL = dataset_range[layer].size();
		arma::rowvec key_row, model_row;
		VectorToRowvec(key_row, key_v);
		VectorToRowvec(model_row, model_v);


		layer -= architecture.size(); // now it represent the begining layer of the learned index
		start_layer = layer;
 
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
		stage_dataset[0][0] = key_row;
		stage_label[0][0] = model_row;

		// training and distribute training set
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
					DistributeTrainingSetOptimized(predictions, stage_dataset[i][j], stage_label[i][j], i + 1, architecture[i + 1], TOTAL_SIZE_MODEL);
				}
			}
			stage_model_BU[layer] = stage_layer;
			layer++;
			//stage_model.push_back(stage_layer);
		}
	}

	// build top-down model (stage model) for non-leaf layer
	// using the origin dataset, using its key but replace its position by the model number.
	void BuildTopDownModelForNonLeafLayer2(const arma::mat& dataset) {
		vector<int> architecture;
		architecture.push_back(1);
		//architecture.push_back(10);
		//architecture.push_back(100);
		architecture.push_back(500);
		//architecture.push_back(1000);

		vector<double> key_dataset_v, model_dataset_v, key_v, model_v;

		int layer = this->level - 1; // bottom layer
		for (int i = 0; i < dataset_range[layer].size(); i++) {
			key_v.push_back(dataset_range[layer][i].second);
			model_v.push_back(i);
		}

		RowvecToVector(dataset, key_dataset_v);
		int model_index = 0;
		for (int i = 0; i < key_dataset_v.size(); i++) {
			if (key_dataset_v[i] <= key_v[model_index]) {
				model_dataset_v.push_back(model_index);
			}
			else {
				model_index++;
				model_dataset_v.push_back(model_index);
			}
		}

		int TOTAL_SIZE_MODEL = dataset_range[layer].size();
		arma::rowvec key_row, model_row;
		VectorToRowvec(key_row, key_dataset_v);
		VectorToRowvec(model_row, model_dataset_v);


		layer -= architecture.size(); // now it represent the begining layer of the learned index
		start_layer = layer;

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
		stage_dataset[0][0] = key_row;
		stage_label[0][0] = model_row;

		// training and distribute training set
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
					DistributeTrainingSetOptimized(predictions, stage_dataset[i][j], stage_label[i][j], i + 1, architecture[i + 1], TOTAL_SIZE_MODEL);
				}
			}
			stage_model_BU[layer] = stage_layer;
			layer++;
			//stage_model.push_back(stage_layer);
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
			//cout << "distributing...target_layer=" << target_layer << " model=" << j << endl;
			stage_dataset[target_layer][j].insert_cols(stage_dataset[target_layer][j].n_cols, dataset.cols(indices)); // using index for dataset may be a bug
			stage_label[target_layer][j].insert_cols(stage_label[target_layer][j].n_cols, labels.cols(indices));
		}
	}

	// if the number of upper layer models is less than threshold, using btree
	void BuildNonLeafLayerWithHybrid(int model_threshold) {
		int layer = this->level - 1;
		while (layer > 0) {
			TrainNonLeafLayer(layer, 1, 1); // configuration!!!!  layer, step, error threshold
			layer--;
			if (stage_model_BU[layer].size() <= model_threshold) {
				start_layer = layer; // record the start layer with 1 model
				// build btree index on it

				btree_model_index.clear();
				for (int i = 0; i < dataset_range[layer].size(); i++) {
					btree_model_index.insert(pair<double, int>(dataset_range[layer][i].second, i));
				}

				break; // alreay 1 model
			}
			cout << "========================================================" << endl;
		}
	}

	// bottom up
	void BuildNonLeafLayerWithLR(int max_non_leaf_layer_threshold = 6) { // should be level - 1 (bottom layer)
		if (max_non_leaf_layer_threshold != -1) {
			int layer = max_non_leaf_layer_threshold;
			while (layer > 0) {
				TrainNonLeafLayer(layer, 1, 20); // configuration!!!!
				layer--;
				if (stage_model_BU[layer].size()==1) {
					start_layer = layer; // record the start layer with 1 model
					break; // alreay 1 model
				}
				cout << "========================================================" << endl;
			}
		}
		else { // train until there is only 1 LR model
			// Todo
		}
	}

	// take input as key, model_id as labels
	void TrainNonLeafLayer(int layer, int step = 10, double threshold = 1) {
		vector<double> key_v, model_v;

		//int layer = this->level - 1; // bottom layer

		for (int i = 0; i < dataset_range[layer].size(); i++) {
			key_v.push_back(dataset_range[layer][i].second);
			model_v.push_back(i);
		}
		int TOTAL_SIZE_MODEL = dataset_range[layer].size();

		layer -= 1; // now it represent the layber above 'bottom' layer

		arma::rowvec key_row, model_row;
		VectorToRowvec(key_row, key_v);
		VectorToRowvec(model_row, model_v);

		arma::rowvec current_trainingset, current_response;
		int dataset_begin = 0, dataset_end = 0 + step <= TOTAL_SIZE_MODEL - 1 ? 0 + step : TOTAL_SIZE_MODEL - 1;
		bool first_flag = true;
		int block_count = 0;
		int model_index = 0;
		LinearRegression lr_stable;

		double current_accuracy, stable_accuracy;
		double current_absolute_accuracy, stable_absolute_accuracy;

		// for the entire bottom layer, train the upper layer!
		while (true) {

			first_flag = true;
			block_count = 1;

			// for an continous_block
			do {
				//cout << dataset_begin << " " << dataset_end << endl;

				current_trainingset = key_row.cols(dataset_begin, dataset_end);
				current_response = model_row.cols(dataset_begin, dataset_end);

				LinearRegression lr_poineer(current_trainingset, current_response);

				// Test the lr's accuracy
				current_accuracy = MeasureAccuracySingleLR(current_trainingset, current_response, lr_poineer);
				current_absolute_accuracy = MeasureAbsoluteAccuracySingleLR(current_trainingset, current_response, lr_poineer);
				//cout << current_accuracy << " " << current_absolute_accuracy << endl;

				// if the accuracy is below threshold, continous merge with the next block
				if (current_absolute_accuracy < threshold) {

					// check finished?
					if (dataset_end == TOTAL_SIZE_MODEL - 1) {
						lr_stable = lr_poineer;
						stage_model_BU[layer].push_back(lr_stable);
						dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1)));
						stable_accuracy = current_accuracy;
						stable_absolute_accuracy = current_absolute_accuracy;
						break;
					}

					dataset_end += step; // judge border!!!
					if (dataset_end >= TOTAL_SIZE_MODEL) {
						dataset_end = TOTAL_SIZE_MODEL - 1;
					}
					first_flag = false;
					block_count++;
					lr_stable = lr_poineer;
					stable_accuracy = current_accuracy;
					stable_absolute_accuracy = current_absolute_accuracy;
					continue;
				}
				else if (first_flag == true) {// if the first_accuracy is already too large, start a new block, except it is the first block, if it's the first block, divide the region into half?
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
					dataset_range[layer].push_back(pair<double, double>(current_trainingset.at(0, 0), current_trainingset.at(0, current_trainingset.n_cols - 1)));
					break;
				}

			} while (true);

			// reset the start and end for the next continous block(s)
			cout << model_index << " " << dataset_begin << " " << dataset_end << " " << stable_accuracy << " " << stable_absolute_accuracy << " " << block_count << endl;
			dataset_begin = dataset_end + 1;
			if (dataset_begin >= TOTAL_SIZE_MODEL) {
				break; // meet the end
			}
			else {
				dataset_end = dataset_begin + step;
				if (dataset_end >= TOTAL_SIZE_MODEL) {
					dataset_end = TOTAL_SIZE_MODEL - 1;
				}
				model_index++;
			}
		}

	}

	void BuildNonLeafLayerWithBtree() {
		bottom_layer_index.clear();
		int layer = this->level - 1; // bottom layer
		for (int i = 0; i < dataset_range[layer].size(); i++) {
			bottom_layer_index.insert(pair<double, int>(dataset_range[layer][i].second, i));
		}
	}

	//==========================================================================================================

	void PredictWithHybrid(vector<double> &queryset, vector<int> &results) {
		// upper layers are btree nodes, lower layers are linear regressions
		stx::btree<double, int>::iterator iter;
		results.clear();
		int model_index = 0;
		int layer = this->level - 1;
		double a, b;
		double result;
		int TOTAL_SIZE = 0;
		for (int k = 0; k < queryset.size(); k++) {
			/*if (i == 19) {
				cout << "debug here!" << endl;
			}*/
			iter = this->btree_model_index.lower_bound(queryset[k]);
			model_index = iter->second;

			for (int i = start_layer; i < stage_model_parameter.size(); i++) {
				a = stage_model_parameter[i][model_index][0];
				b = stage_model_parameter[i][model_index][1];
				result = a * queryset[k] + b;
				if (i < stage_model_parameter.size() - 1) {
					TOTAL_SIZE = stage_model_parameter[i + 1].size();
					model_index = (result / TOTAL_SIZE * stage_model_parameter[i + 1].size());
					//index *= stage_model_parameters[i + 1].size();
					//index = int(index);
					if (model_index < 0) {
						model_index = 0;
					}
					else if (model_index > stage_model_parameter[i + 1].size() - 1) {
						model_index = stage_model_parameter[i + 1].size() - 1;
					}
				}
			}
			results.push_back(result);
		}
	}

	// in the learned non-leaf layer, it will find the exact using scan
	void PredictWithHybridShift(vector<double> &queryset, vector<int> &results) {
		// upper layers are btree nodes, lower layers are linear regressions
		stx::btree<double, int>::iterator iter;
		results.clear();
		int model_index = 0;
		int layer = this->level - 1;
		double a, b;
		double result;
		int TOTAL_SIZE = 0;
		for (int k = 0; k < queryset.size(); k++) {

			iter = this->btree_model_index.lower_bound(queryset[k]);
			model_index = iter->second;

			for (int i = start_layer; i < stage_model_parameter.size(); i++) {

				a = stage_model_parameter[i][model_index][0];
				b = stage_model_parameter[i][model_index][1];
				result = a * queryset[k] + b;
				TOTAL_SIZE = stage_model_parameter[i + 1].size();
				model_index = (result / TOTAL_SIZE * stage_model_parameter[i + 1].size());
				if (model_index < 0) {
					model_index = 0;
				}
				else if (model_index > stage_model_parameter[i + 1].size() - 1) {
					model_index = stage_model_parameter[i + 1].size() - 1;
				}

				// scan until the true model is found
				if (i < stage_model_parameter.size() - 1) {
					while (true) {
						//int prior = index - 1;
						if (queryset[k] > dataset_range[i+1][model_index].second) {
							model_index++;
							//shift++;
						}
						else if (model_index - 1 >= 0 && queryset[k] < dataset_range[i+1][model_index - 1].second) {
							model_index--;
							//shift++;
						}
						else {
							break;
						}
					}
				}	
			}
			results.push_back(result);
		}
	}

	void PredictWithLR(vector<double> &queryset, vector<int> &results) {
		double result;
		results.clear();
		double a, b;
		int index = 0;
		int TOTAL_SIZE = 0;
		for (int k = 0; k < queryset.size(); k++) {
			index = 0;
			for (int i = start_layer; i < stage_model_parameter.size(); i++) {
				a = stage_model_parameter[i][index][0];
				b = stage_model_parameter[i][index][1];
				result = a * queryset[k] + b;
				if (i < stage_model_parameter.size() - 1) {
					TOTAL_SIZE = stage_model_parameter[i + 1].size();
					index = (result / TOTAL_SIZE * stage_model_parameter[i + 1].size());
					//index *= stage_model_parameters[i + 1].size();
					//index = int(index);
					if (index < 0) {
						index = 0;
					}
					else if (index > stage_model_parameter[i + 1].size() - 1) {
						index = stage_model_parameter[i + 1].size() - 1;
					}
				}
			}
			results.push_back(result);
		}
	}

	// using LR as the non-leaf index, but need to find the exact bottom layer index for prediction.
	void PredictWithLRShift(vector<double> &queryset, vector<int> &results) {
		double result=0;
		results.clear();
		double a=0, b=0;
		int model_index = 0;
		int TOTAL_SIZE = 0;

		//int max_shift = 0;
		//int shift = 0; // could be cancelled
		for (int k = 0; k < queryset.size(); k++) {
			model_index = 0;
			for (int i = start_layer; i < stage_model_parameter.size(); i++) {
				a = stage_model_parameter[i][model_index][0];
				b = stage_model_parameter[i][model_index][1];
				result = a * queryset[k] + b;
				TOTAL_SIZE = stage_model_parameter[i + 1].size();
				model_index = (result / TOTAL_SIZE * stage_model_parameter[i + 1].size());
				if (model_index < 0) {
					model_index = 0;
				}
				else if (model_index > stage_model_parameter[i + 1].size() - 1) {
					model_index = stage_model_parameter[i + 1].size() - 1;
				}

				// scan until the true model is found
				if (i < stage_model_parameter.size() - 1) {
					while (true) {
						//int prior = index - 1;
						if (queryset[k] > dataset_range[i + 1][model_index].second) {
							model_index+=1;
							//shift++;
						}
						else if (model_index - 1 >= 0 && queryset[k] < dataset_range[i + 1][model_index - 1].second) {
							model_index-=1;
							//shift++;
						}
						else {
							break;
						}
					}
				}
			}
			results.push_back(result);
		}
		//cout << "avg shift :" << shift/1000 << endl;
		//cout << "max shift: " << max_shift << endl;
	}

	void PredictWithLRShiftOptimized(vector<double> &queryset, vector<int> &results) {
		double result = 0;
		results.clear();
		double a = 0, b = 0;
		int model_index = 0;
		int TOTAL_SIZE = 0;

		//int max_shift = 0;
		//int shift = 0; // could be cancelled
		for (int k = 0; k < queryset.size(); k++) {
			model_index = 0;
			for (int i = start_layer; i < stage_model_parameter.size()-1; i++) {
				a = stage_model_parameter[i][model_index][0];
				b = stage_model_parameter[i][model_index][1];
				result = a * queryset[k] + b;
				TOTAL_SIZE = stage_model_parameter[i + 1].size();
				model_index = (result / TOTAL_SIZE * stage_model_parameter[i + 1].size());
				if (model_index < 0) {
					model_index = 0;
				}
				else if (model_index > stage_model_parameter[i + 1].size() - 1) {
					model_index = stage_model_parameter[i + 1].size() - 1;
				}

				// scan until the true model is found				
				while(queryset[k] > dataset_range[i + 1][model_index].second) {
					model_index += 1;
				}
				while(model_index - 1 >= 0 && queryset[k] < dataset_range[i + 1][model_index - 1].second) {
					model_index -= 1;
				}
			}
			a = stage_model_parameter[level-1][model_index][0];
			b = stage_model_parameter[level-1][model_index][1];
			result = a * queryset[k] + b;
			results.push_back(result);
		}
		//cout << "avg shift :" << shift/1000 << endl;
		//cout << "max shift: " << max_shift << endl;
	}

	void PredictWithLRBinary(vector<double> &queryset, vector<int> &results) {
		double result = 0;
		results.clear();
		double a = 0, b = 0;
		int model_index = 0;
		int TOTAL_SIZE = 0;

		int ERROR_THRESHOLD = 20; // should be auto configured later
		int middle = 0, begin = 0, end = 0;

		//int max_shift = 0;
		//int shift = 0; // could be cancelled
		for (int k = 0; k < queryset.size(); k++) {

			model_index = 0;
			for (int i = start_layer; i < stage_model_parameter.size() - 1; i++) {
				a = stage_model_parameter[i][model_index][0];
				b = stage_model_parameter[i][model_index][1];
				result = a * queryset[k] + b;
				TOTAL_SIZE = stage_model_parameter[i + 1].size();
				model_index = (result / TOTAL_SIZE * stage_model_parameter[i + 1].size());
				if (model_index < 0) {
					model_index = 0;
				}
				else if (model_index > stage_model_parameter[i + 1].size() - 1) {
					model_index = stage_model_parameter[i + 1].size() - 1;
				}

				// first check if it's outside the error threshold
				begin = model_index - ERROR_THRESHOLD >= 0 ? model_index - ERROR_THRESHOLD : 0;
				end = model_index + ERROR_THRESHOLD <= dataset_range[i + 1].size() - 1 ? model_index + ERROR_THRESHOLD : dataset_range[i + 1].size() - 1;
				
				// do binary search, about 30 ns
				middle = (begin + end) / 2;
				model_index = middle;
				while (true) {
					cout << begin << " " << middle << " " << end << endl;
					if (queryset[k] > dataset_range[i + 1][middle].second) {
						begin = middle + 1;
						middle = (begin + end) / 2;
						//cout << "middle: " << middle << "  begin: " << begin << " end: " << end  << " begin+end / 2" << (begin + end) / 2 << endl;
					}
					else if (queryset[k] < dataset_range[i + 1][middle - 1].second) {
						end = middle;
						middle = (begin + end) / 2;
					}
					else {
						model_index = middle;
						break;
					}
				}
			}
			a = stage_model_parameter[level - 1][model_index][0];
			b = stage_model_parameter[level - 1][model_index][1];
			result = a * queryset[k] + b;
			results.push_back(result);
		}
	}

	void PredictWithLRBinaryShiftOptimized(vector<double> &queryset, vector<int> &results) {
		double result = 0;
		results.clear();
		double a = 0, b = 0;
		int model_index = 0;
		int TOTAL_SIZE = 0;

		int ERROR_THRESHOLD = 20; // should be auto configured later
		int middle = 0, begin = 0, end = 0;

		//int max_shift = 0;
		//int shift = 0; // could be cancelled
		for (int k = 0; k < queryset.size(); k++) {

			model_index = 0;
			for (int i = start_layer; i < stage_model_parameter.size() - 1; i++) {
				a = stage_model_parameter[i][model_index][0];
				b = stage_model_parameter[i][model_index][1];
				result = a * queryset[k] + b;
				TOTAL_SIZE = stage_model_parameter[i + 1].size();
				model_index = (result / TOTAL_SIZE * stage_model_parameter[i + 1].size());
				if (model_index < 0) {
					model_index = 0;
				}
				else if (model_index > stage_model_parameter[i + 1].size() - 1) {
					model_index = stage_model_parameter[i + 1].size() - 1;
				}

				// first check if it's outside the error threshold
				begin = model_index - ERROR_THRESHOLD >= 0 ? model_index - ERROR_THRESHOLD : 0;
				end = model_index + ERROR_THRESHOLD <= dataset_range[i + 1].size() - 1 ? model_index + ERROR_THRESHOLD : dataset_range[i + 1].size() - 1;

				// scan until the true model is found, about 10 ns
				if (queryset[k] > dataset_range[i + 1][end].second) {
					model_index = end;
					while (queryset[k] > dataset_range[i + 1][model_index].second) {
						model_index += 1;
					}
				}
				else if (queryset[k] < dataset_range[i + 1][begin].second) {
					model_index = begin;
					while (model_index - 1 >= 0 && queryset[k] < dataset_range[i + 1][model_index - 1].second) {
						model_index -= 1;
					}
				}
				else {
					// do binary search, about 30 ns
					middle = (begin + end) / 2;
					model_index = middle;
					while (true) {
						//cout << begin << " " << middle << " " << end << endl;
						if (queryset[k] > dataset_range[i + 1][middle].second) {
							begin = middle+1;
							middle = (begin + end) / 2;
							//cout << "middle: " << middle << "  begin: " << begin << " end: " << end  << " begin+end / 2" << (begin + end) / 2 << endl;
						}
						else if (queryset[k] < dataset_range[i + 1][middle - 1].second) {
							end = middle;
							middle = (begin + end) / 2;
						}
						else {
							model_index = middle;
							break;
						}
					}
				}

			}
			a = stage_model_parameter[level - 1][model_index][0];
			b = stage_model_parameter[level - 1][model_index][1];
			result = a * queryset[k] + b;
			results.push_back(result);
		}
		//cout << "avg shift :" << shift/1000 << endl;
		//cout << "max shift: " << max_shift << endl;
	}


	// using LR as the non-leaf index, but need to find the exact bottom layer index for prediction.
	// using binary search instead of scan
	void PredictWithLRShift2(vector<double> &queryset, vector<int> &results) {
		double result = 0;
		results.clear();
		double a = 0, b = 0;
		int index = 0;
		int TOTAL_SIZE = 0;
		int TOTAL_MODEL = dataset_range[level-1].size();
		int middle = 0, begin = 0, end = 0;

		for (int k = 0; k < queryset.size(); k++) {
			index = 0;		
			// binary search untill meet the correct model
			begin = 0;
			end = TOTAL_MODEL - 1;
			middle = (begin+end) / 2;
			while (true) {
				//int prior = index - 1;
				if (queryset[k] > dataset_range[level - 1][middle].second) {
					begin = middle;
					middle = (begin + end) / 2;
				}
				else if (middle - 1 >= 0 && queryset[k] < dataset_range[level - 1][middle - 1].second) {
					end = middle;
					middle = (begin + end) / 2;
				}
				else {
					break;
				}
			}
			a = stage_model_parameter[level-1][middle][0];
			b = stage_model_parameter[level-1][middle][1];
			//cout << k << " " << middle << endl;
			result = a * queryset[k] + b;		
			results.push_back(result);
		}
	}

	void PredictWithStxBtree(vector<double> &queryset, vector<int> &results) {
		stx::btree<double, int>::iterator iter;
		results.clear();
		int model_index = 0;
		int layer = this->level - 1;
		double a, b;
		double result;
		for (int i = 0; i < queryset.size(); i++) {
			/*if (i == 0 || i==39 || i==63) {
 				cout << "debug here!" << endl;
			}*/
			iter = this->bottom_layer_index.lower_bound(queryset[i]);
			model_index = iter->second;
			//cout << queryset[i] << " " << model_index << endl;

			a = stage_model_parameter[layer][model_index][0];
			b = stage_model_parameter[layer][model_index][1];
			//cout << a << " " << b << " " << model_index << endl;
			result = a * queryset[i] + b;
			results.push_back(result);
		}
	}

	//==========================================================================================================

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

	double inline MeasureMaxAbsoluteError(arma::mat &testset, arma::rowvec &response, LinearRegression &lr) {
		// do the predicition
		arma::rowvec prediction;
		lr.Predict(testset, prediction);

		arma::rowvec absolute_error = abs(prediction - response);
		absolute_error.elem(find_nonfinite(absolute_error)).zeros(); // 'remove' ¡ÀInf or NaN
		int index = absolute_error.index_max();
		return absolute_error.at(index);
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
	int start_layer = 0;

	vector<vector<LinearRegression>> stage_model_BU;
	vector<vector<pair<double, double>>> dataset_range;
	vector<vector<vector<double>>> stage_model_parameter;

	stx::btree<double, int> bottom_layer_index; // for btree index

	stx::btree<double, int> btree_model_index; // for hybrid index

	vector<vector<arma::mat>> stage_dataset; // for top down learned index
	vector<vector<arma::rowvec>> stage_label; // for top down learned index
};