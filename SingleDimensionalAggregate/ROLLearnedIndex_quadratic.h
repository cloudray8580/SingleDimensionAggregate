#pragma once
#pragma once

//#define __ASSERT_MACROS_DEFINE_VERSIONS_WITHOUT_UNDERSCORES 0
#include "STXBtreeAggregate.h"
#include "mlpack/core.hpp"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <vector>

using namespace std;
using namespace mlpack::regression;

// descending
bool cmp_magnitude2(pair<int, double> &p1, pair<int, double> &p2) {
	return p1.second > p2.second;
}

class ROLLearnedIndex_quadratic {
public:
	ROLLearnedIndex_quadratic(int level = 4, int blocks = 1000, double threshold = 200) {
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

	// key_v still use single, current_trainingset should contains the quadratic term?
	void ApproximateMaxLossLinearRegression(int origin_index, int shift_index, vector<double> &key_v, vector<double> &position_v, const arma::mat& current_trainingset, const arma::rowvec&current_response, double sampling_control, double error_threshold, double sampling_percentage, double &slope_a, double &slope_b, double &intercept, double &maxloss) {

		auto t0 = chrono::steady_clock::now();

		// 1. first train an linear regression as the axes
		LinearRegression lr(current_trainingset, current_response);

		// 2. calculate the candidate max error points
		arma::vec paras = lr.Parameters();
		double a, b, c;
		a = paras[1]; // a
		b = paras[2]; // b
		c = paras[0]; // c
		//cout << "a: " << a << " b: " << b << " c: " << c << endl;

		// lower the LR to make the dataset above the line to ease the following calculation
		double signed_error = 0, max_negative_error = 0;
		double predicted_position;
		for (int i = origin_index; i <= shift_index; i++) {
			predicted_position = a * key_v[i] * key_v[i] + b * key_v[i] + c;
			signed_error = position_v[i] - predicted_position;
			if (signed_error < max_negative_error) {
				max_negative_error = signed_error;
				//cout << i << " " << max_negative_error << " " << signed_error << endl;
			}
		}
		c += max_negative_error;

		double local_maximum = 0, local_minimum = shift_index - origin_index;
		double pre_local_maximum = 0, pre_local_minimum = 0;
		int local_maximum_pos = 0, local_minimum_pos = 0;
		bool max_min_ready_flag = false, min_max_ready_flag = false;

		// scan the dataset to find those significant turning points
		int segment_size = shift_index - origin_index + 1;
		double error = 0;
		vector<pair<int, double>> segmented_pos; // record the segmentation points
		for (int i = 0; i < segment_size; i++) {
			predicted_position = a * key_v[i + origin_index] * key_v[i + origin_index] + b * key_v[i + origin_index] + c;
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
				//segmented_pos.push_back(pair<int, double>(origin_index + local_minimum_pos, pre_local_maximum - local_minimum));
				segmented_pos.push_back(pair<int, double>(local_minimum_pos, pre_local_maximum - local_minimum));
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
				//segmented_pos.push_back(pair<int, double>(origin_index + local_maximum_pos, local_maximum - pre_local_minimum));
				segmented_pos.push_back(pair<int, double>(local_maximum_pos, local_maximum - pre_local_minimum));
				min_max_ready_flag = false;
				local_minimum = error;
				local_minimum_pos = i;
				pre_local_maximum = local_maximum;
			}
		}

		// sort these segmented position according to its scope		
		std::sort(segmented_pos.begin(), segmented_pos.end(), cmp_magnitude2);

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
		b = paras[2]; // b
		c = paras[0]; // c

		// find the max positive error and the max negative error
		double max_pos_err = 0, max_neg_err = 0;
		for (int i = 0; i < segment_size; i++) {
			predicted_position = a * key_v[i + origin_index] * key_v[i + origin_index] + b * key_v[i + origin_index] + c;
			error = position_v[origin_index + i] - predicted_position;
			if (error > max_pos_err) {
				max_pos_err = error;
			}
			else if (error < max_neg_err) {
				max_neg_err = error;
			}
		}
		double adjusted_c = c + (max_pos_err - abs(max_neg_err)) / 2;
		double maxerr = (max_pos_err - max_neg_err) / 2;
		//cout << "max error for approx max loss:" << maxerr << endl;
		//cout << "a: " << a << " b: " << b << " c: " << c <<  endl;

		slope_a = a;
		slope_b = b;
		intercept = adjusted_c;
		maxloss = maxerr;

		//auto t1 = chrono::steady_clock::now();
		//cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	}

	// using arma format, include multi dimension
	void MyCplexSolverForMaxLossQuadratic(const arma::mat& current_trainingset, const arma::rowvec&current_response, double &a, double &b, double &c, double &loss) {
		IloEnv env;
		IloModel model(env);
		IloCplex cplex(model);
		IloObjective obj(env);
		IloNumVarArray vars(env);
		IloRangeArray ranges(env);

		cplex.setOut(env.getNullStream());

		// set variable type, IloNumVarArray starts from 0.
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // the weight, i.e., a for x^2
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // the weight, i.e., b for x
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT));      // the bias, i.e., c
		vars.add(IloNumVar(env, 0.0, INFINITY, ILOFLOAT)); // our target, the max loss

		cplex.setParam(IloCplex::RootAlg, IloCplex::Barrier); // set optimizer used interior point method
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Sifting); // set optimizer used interior point method

		// declare objective
		obj.setExpr(vars[3]);
		obj.setSense(IloObjective::Minimize);
		model.add(obj);


		// add constraint for each record
		for (int i = 0; i < current_trainingset.n_cols; i++) {
			model.add(vars[0] * current_trainingset(0,i) + vars[1] * current_trainingset(1, i) + vars[2] - current_response[i] <= vars[3]);
			model.add(vars[0] * current_trainingset(0,i) + vars[1] * current_trainingset(1, i) + vars[2] - current_response[i] >= -vars[3]);
		}

		IloNum starttime_ = cplex.getTime();
		cplex.solve();
		/*try {
			cplex.solve();
		}
		catch (IloException &e) {
			std::cerr << "IloException: " << e << endl;
		}
		catch (std::exception &e) {
			std::cerr << "standard exception: " << e.what() << endl;
		}
		catch (...) {
			std::cerr << "some other exception: " << endl;
		}*/

		IloNum endtime_ = cplex.getTime();
		double target = cplex.getObjValue();
		double slope1, slope2, bias;

		slope1 = cplex.getValue(vars[0]);
		slope2 = cplex.getValue(vars[1]);
		bias = cplex.getValue(vars[2]);

		//cout << "the variable a: " << cplex.getValue(vars[0]) << endl;
		//cout << "the variable b: " << cplex.getValue(vars[1]) << endl;
		//cout << "the variable max loss: " << cplex.getValue(vars[2]) << endl;
		//cout << "cplex solve time: " << endtime_ - starttime_ << endl;

		env.end();

		//return target;
		a = slope1;
		b = slope2;
		c = bias;
		loss = target;
	}

	void MyCplexSolverForMaxLossQuadraticOptimized(vector<double> &key_v, vector<double> &position_v, int lower_index, int higher_index, double &a, double &b, double &c, double &loss) {
		IloEnv env;
		IloModel model(env);
		IloCplex cplex(model);
		IloObjective obj(env);
		IloNumVarArray vars(env);
		IloRangeArray ranges(env);

		//cplex.setOut(env.getNullStream());

		cplex.setParam(IloCplex::RootAlg, IloCplex::Primal);
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Barrier); // set optimizer used interior point method
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Sifting); // set optimizer used interior point method


		// set variable type, IloNumVarArray starts from 0.
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // the weight, i.e., a for x^2
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // the weight, i.e., b for x
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT));      // the bias, i.e., c
		vars.add(IloNumVar(env, 0.0, 3000, ILOFLOAT)); // our target, the max loss

		// declare objective
		obj.setExpr(vars[3]);
		obj.setSense(IloObjective::Minimize);
		model.add(obj);

		// add constraint for each record
		for (int i = lower_index; i <= higher_index; i++) {
			model.add(vars[0] * key_v[i] * key_v[i] + vars[1] * key_v[i] + vars[2] - position_v[i] <= vars[3]);
			model.add(vars[0] * key_v[i] * key_v[i] + vars[1] * key_v[i] + vars[2] - position_v[i] >= -vars[3]);
		}

		IloNum starttime_ = cplex.getTime();
		cplex.solve();

		IloNum endtime_ = cplex.getTime();
		double target = cplex.getObjValue();
		double slope1, slope2, bias;

		slope1 = cplex.getValue(vars[0]);
		slope2 = cplex.getValue(vars[1]);
		bias = cplex.getValue(vars[2]);

		//cout << "the variable a: " << cplex.getValue(vars[0]) << endl;
		//cout << "the variable b: " << cplex.getValue(vars[1]) << endl;
		//cout << "the variable max loss: " << cplex.getValue(vars[2]) << endl;
		//cout << "cplex solve time: " << endtime_ - starttime_ << endl;

		env.end();

		//return target;
		a = slope1;
		b = slope2;
		c = bias;
		loss = target;
	}

	// do not need to dump parameter for the bottom layer anymore
	// the x^2 and x are inside the training set
	void TrainBottomLayerWithFastDetectForMaxLoss(const arma::mat& dataset, const arma::rowvec& labels, int step = 1000, double sampling_control = 2, double sampling_percentage = 0.1) {
		// Todo
		int layer = this->level - 1;

		int minimum_step = step;

		vector<double> key_v, position_v;
		RowvecToVector(dataset.row(1), key_v);
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
		double a, b, c; // slope and bias

		int segment_count = 0;

		while (origin_index < TOTAL_SIZE) {

			//cout << stage_model_BU[layer].size() << " " << origin_index << " " << shift_index << " " << step << endl;

			if (shift_index >= TOTAL_SIZE) {
				shift_index = TOTAL_SIZE - 1;
			}

			arma::mat current_trainingset = dataset.cols(origin_index, shift_index); // included [origin_index, shift_index]
			arma::rowvec current_response = labels.cols(origin_index, shift_index);

			//LinearRegression lr_poineer(current_trainingset, current_response);

			//ApproximateMaxLossLinearRegression(origin_index, shift_index, key_v, position_v, current_trainingset, current_response, sampling_control, threshold, sampling_percentage, a, b, c, current_absolute_accuracy);
			
			MyCplexSolverForMaxLossQuadratic(current_trainingset, current_response, a, b, c, current_absolute_accuracy);

			model.clear();
			model.push_back(a);
			model.push_back(b);
			model.push_back(c);

			if (current_absolute_accuracy > threshold) {
				if (lr_correction_tag) {
					// save the previous model
					stage_model_parameter_layer.push_back(temp_model);
					dataset_range[layer].push_back(temp_dataset_range);

					//cout << "segment: " << segment_count << " segment size: " << shift_index - origin_index << endl;
					segment_count++;

					// roll back index
					origin_index = pre_origin_index;
					shift_index = pre_shift_index;

					// update step
					step = shift_index - origin_index;
					step *= 2;

					if (step < minimum_step) {
						step = minimum_step;
					}

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
					predicted_pos = a * key_v[shift_index] * key_v[shift_index] + b * key_v[shift_index] + c;
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
					temp_dataset_range = pair<double, double>(key_v[origin_index], key_v[shift_index]);
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
					model.push_back(c);
					stage_model_parameter_layer.push_back(model);
					dataset_range[layer].push_back(pair<double, double>(key_v[origin_index], key_v[shift_index]));
					//cout << "segment: " << segment_count << " segment size: " << shift_index - origin_index << endl;
					segment_count++;
					// update the step
					step = shift_index - origin_index;
					step *= 2;

					if (step < minimum_step) {
						step = minimum_step;
					}

					// prepare for the next segment
					origin_index = shift_index + 1; // verify whether + 1 is necessary here !
					shift_index = origin_index + step;
					lr_correction_tag = false;
				}
			}
		}
		stage_model_parameter[layer] = stage_model_parameter_layer;
	}

	// using linear programming
	// dataset only contains the keys  with highest term 1
	void TrainBottomLayerWithSegmentOnTrainMaxLossOptimized(const arma::mat& dataset, const arma::rowvec& labels) {
		int layer = this->level - 1;
		index_range.clear();

		vector<double> key_v, position_v;
		RowvecToVector(dataset, key_v);
		RowvecToVector(labels, position_v);

		int TOTAL_SIZE = dataset.n_cols; // 1157570

		int origin_index = 0;
		int shift_index = origin_index;

		double current_absolute_accuracy;

		vector<double> model, temp_model;
		vector<vector<double>> stage_model_parameter_layer;
		double a, b, c; // slope and bias, ax^2 + bx + c
		int segment_count = 0;

		int cone_origin_index = 0;
		int cone_shift_index = 0;
		double upper_pos, lower_pos;
		double slope_high, slope_low;
		double current_slope;
		dataset_range.clear();
		bool exit_tag = false;

		while (origin_index < TOTAL_SIZE) {

			//if (index_range.size() == 24) {
			//	cout << "debug here!" << endl;
			//}

			// 1. segment initializer using ShrinkingCone, determine the shift_index

			// build the cone with first and second point
			cone_origin_index = origin_index;
			cone_shift_index = cone_origin_index + 1;

			if (cone_shift_index >= TOTAL_SIZE) {
				//dataset_range.push_back(pair<double, double>(key_v[cone_origin_index], key_v[cone_shift_index]));
				break;
			}

			while (key_v[cone_shift_index] == key_v[cone_origin_index]) {
				cone_shift_index++;
			}

			slope_high = (position_v[cone_shift_index] + threshold - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
			slope_low = (position_v[cone_shift_index] - threshold - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);

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
					if (position_v[cone_shift_index] + threshold < upper_pos) {
						// update slope_high
						slope_high = (position_v[cone_shift_index] + threshold - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
					}
					if (position_v[cone_shift_index] - threshold > lower_pos) {
						// update slope_low
						slope_low = (position_v[cone_shift_index] - threshold - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
					}
				}
				else {
					// outside the cone, start a new segement.
					exit_tag = true;
					break;
				}
			}

			// 2. segment exponential search

			shift_index = cone_shift_index;

			if (shift_index >= TOTAL_SIZE) {
				shift_index = TOTAL_SIZE - 1;
			}

			int Co = shift_index - origin_index;
			MyCplexSolverForMaxLossQuadraticOptimized(key_v, position_v, origin_index, shift_index, a, b, c, current_absolute_accuracy);
			//cout << "current max error: " << current_absolute_accuracy << endl;
			while (current_absolute_accuracy < threshold && shift_index < TOTAL_SIZE - 1) {
				Co *= 2;
				shift_index = origin_index + Co;
				if (shift_index >= TOTAL_SIZE) {
					shift_index = TOTAL_SIZE - 1;
				}
				MyCplexSolverForMaxLossQuadraticOptimized(key_v, position_v, origin_index, shift_index, a, b, c, current_absolute_accuracy);
				//cout << "current max error: " << current_absolute_accuracy << endl;
			}

			// 3. segment binary search (BETWEEN origin_index + Co/2 and origin_index + Co)

			//cout << "current max error: " << current_absolute_accuracy << endl;
			if (current_absolute_accuracy < threshold && shift_index == TOTAL_SIZE - 1) {
				// stop
			}
			else {
				int Ilow = origin_index + Co / 2, Ihigh = shift_index, Middle;
				while (Ihigh - Ilow > 1) {
					Middle = (Ilow + Ihigh) / 2;

					shift_index = Middle;

					MyCplexSolverForMaxLossQuadraticOptimized(key_v, position_v, origin_index, shift_index, a, b, c, current_absolute_accuracy);
					//cout << "current max error: " << current_absolute_accuracy << endl;
					if (current_absolute_accuracy > threshold) {
						Ihigh = Middle;
					}
					else {
						Ilow = Middle;
					}
				}
				if (current_absolute_accuracy > threshold) {
					shift_index = Ilow;
					MyCplexSolverForMaxLossQuadraticOptimized(key_v, position_v, origin_index, shift_index, a, b, c, current_absolute_accuracy);
				}
			}
			//cout << "current max error: " << current_absolute_accuracy << endl;
			
			model.clear();
			model.push_back(a);
			model.push_back(b);
			model.push_back(c);
			stage_model_parameter_layer.push_back(model);
			dataset_range[layer].push_back(pair<double, double>(key_v[origin_index], key_v[shift_index]));
			
			//cout << "segment: " << segment_count << " segment size: " << shift_index - origin_index << endl;
			segment_count++;
			index_range.push_back(pair<int, int>(origin_index, shift_index));

			// prepare for the next segment
			origin_index = shift_index + 1; // verify whether + 1 is necessary here !
		}
		stage_model_parameter[level - 1] = stage_model_parameter_layer;
	}

	//==========================================================================================================

	void BuildNonLeafLayerWithBtree() {
		bottom_layer_index.clear();
		int layer = this->level - 1; // bottom layer
		for (int i = 0; i < dataset_range[layer].size(); i++) {
			bottom_layer_index.insert(pair<double, int>(dataset_range[layer][i].first, i));
		}
	}

	//==========================================================================================================

	void PredictWithStxBtree(vector<double> &queryset, vector<int> &results) {
		stx::btree<double, int>::iterator iter;
		results.clear();
		int model_index = 0;
		int layer = this->level - 1;
		double a, b, c;
		double result;
		for (int i = 0; i < queryset.size(); i++) {
			/*if (i == 0 || i==39 || i==63) {
				cout << "debug here!" << endl;
			}*/
			iter = this->bottom_layer_index.upper_bound(queryset[i]);
			iter--;
			model_index = iter->second;
			//cout << queryset[i] << " " << model_index << endl;

			a = stage_model_parameter[layer][model_index][0];
			b = stage_model_parameter[layer][model_index][1];
			c = stage_model_parameter[layer][model_index][2];
			//cout << a << " " << b << " " << model_index << endl;
			result = a * queryset[i]* queryset[i] + b* queryset[i] + c;
			results.push_back(result);
		}
	}

	//==========================================================================================================

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
				model.push_back(paras[2]); // b
				model.push_back(paras[0]); // c check ?
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
	vector<pair<int, int>> index_range;

	stx::btree<double, int> bottom_layer_index; // for btree index
};