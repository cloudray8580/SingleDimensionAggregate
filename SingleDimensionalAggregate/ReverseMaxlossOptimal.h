#pragma once

#include<vector>
#include "STXBtreeAggregate.h"
#include "mlpack/core.hpp"
#include <ilcplex/ilocplex.h>

class ReverseMaxlossOptimal{
public:

	ReverseMaxlossOptimal(double t_abs = 100, double t_rel = 0.01, int highest_term = 1) {
		this->t_abs = t_abs;
		this->t_rel = t_rel;
		this->highest_term = highest_term;
	}

	void SolveMaxlossLP(const vector<double> &key_v, const vector<double> &position_v, int lower_index, int higher_index, vector<double> &paras, double &loss) {
		IloEnv env;
		IloModel model(env);
		IloCplex cplex(model);
		IloObjective obj(env);
		IloNumVarArray vars(env); // the coefficients
		IloRangeArray ranges(env);

		env.setOut(env.getNullStream());
		cplex.setOut(env.getNullStream());
		cplex.setParam(IloCplex::MIPInterval, 1000);

		cplex.setParam(IloCplex::NumericalEmphasis, CPX_ON);
		//cplex.setParam(IloCplex::Param::Advance, 0); // turnoff advanced start
		//cplex.setParam(IloCplex::Param::Preprocessing::Presolve, false); // turnoff presolve

		// set variable type, IloNumVarArray starts from 0.
		for (int i = 0; i <= this->highest_term; i++) {
			vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT));  // the weight, i.e., a2 for x^2, start from the lowest, i.e., a0, to an
		}
		//vars.add(IloNumVar(env, 0.0, INFINITY, ILOFLOAT)); // our target, the max loss

		IloNumVar target(env, 0.0, INFINITY, ILOFLOAT); // the max loss value

		//cplex.setParam(IloCplex::RootAlg, IloCplex::Dual); // using dual simplex
		cplex.setParam(IloCplex::RootAlg, IloCplex::Barrier); // set optimizer used interior point method
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Sifting); // set optimizer used sifting method
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Concurrent);

		cplex.setParam(IloCplex::Param::Barrier::Limits::Growth, 1e13);

		// declare objective
		obj.setExpr(target);
		obj.setSense(IloObjective::Minimize);
		model.add(obj);

		vector<double> key_term;
		double key = 1;

		// add constraint for each record
		for (int i = lower_index; i <= higher_index; i++) {
			IloExpr model_term(env);
			key = 1;
			key_term.clear();
			for (int j = 0; j <= highest_term; j++) {
				key_term.push_back(key);
				key *= key_v[i];
			}

			for (int j = highest_term; j >= 0; j--) {
				model_term += vars[j] * key_term[j];
				//cout << "i: " << i << endl;
				//key *= key_v[i];
			}
			model.add(model_term - position_v[i] <= target);
			model.add(model_term - position_v[i] >= -target);
			//model.add(vars[0] * key_v[i] * key_v[i] + vars[1] * key_v[i] + vars[2] - position_v[i] <= vars[3]);
			//model.add(vars[0] * key_v[i] * key_v[i] + vars[1] * key_v[i] + vars[2] - position_v[i] >= -vars[3]);
		}

		IloNum starttime_ = cplex.getTime();
		//cplex.solve();
		try {
			cplex.solve();
		}
		catch (IloException &e) {
			std::cerr << "iloexception: " << e << endl;
		}
		catch (std::exception &e) {
			std::cerr << "standard exception: " << e.what() << endl;
		}
		catch (...) {
			std::cerr << "some other exception: " << endl;
		}
		IloNum endtime_ = cplex.getTime();
		
		//cplex.exportModel("C:/Users/Cloud/Desktop/range392.sav");
		loss = cplex.getObjValue();

		paras.clear();
		for (int i = 0; i <= this->highest_term; i++) {
			paras.push_back(cplex.getValue(vars[i]));
		}

		//cout << "the variable a: " << cplex.getValue(vars[0]) << endl;
		//cout << "the variable b: " << cplex.getValue(vars[1]) << endl;
		//cout << "the variable max loss: " << cplex.getValue(vars[2]) << endl;
		//cout << "cplex solve time: " << endtime_ - starttime_ << endl;

		env.end();
	}

	// using linear programming
	// dataset only contains the keys with highest term 1
	void SegmentOnTrainMaxLossModel(const arma::mat& dataset, const arma::rowvec& labels) {

		model_parameters.clear();
		dataset_range.clear();
		index_range.clear();

		vector<double> key_v, position_v;
		RowvecToVector(dataset, key_v);
		RowvecToVector(labels, position_v);

		int TOTAL_SIZE = dataset.n_cols; // 1157570

		int origin_index = 0;
		int shift_index = origin_index;

		double current_absolute_accuracy;

		vector<double> model, temp_model;

		int segment_count = 0;

		int cone_origin_index = 0;
		int cone_shift_index = 0;
		double upper_pos, lower_pos;
		double slope_high, slope_low;
		double current_slope;
		dataset_range.clear();
		bool exit_tag = false;

		while (origin_index < TOTAL_SIZE) {

			// 1. segment initializer using ShrinkingCone, determine the shift_index

			// build the cone with first and second point
			cone_origin_index = origin_index;
			cone_shift_index = cone_origin_index + 1;

			if (cone_shift_index >= TOTAL_SIZE) {
				break;
			}

			while (key_v[cone_shift_index] == key_v[cone_origin_index]) {
				cone_shift_index++;
			}

			slope_high = (position_v[cone_shift_index] + t_abs - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
			slope_low = (position_v[cone_shift_index] - t_abs - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);

			exit_tag = false;

			// test if the following points are in the cone
			while (cone_shift_index < TOTAL_SIZE) {
				cone_shift_index++;

				// test if exceed  the border, if the first time, if the last segement

				upper_pos = slope_high * (key_v[cone_shift_index] - key_v[cone_origin_index]) + position_v[cone_origin_index];
				lower_pos = slope_low * (key_v[cone_shift_index] - key_v[cone_origin_index]) + position_v[cone_origin_index];

				if (position_v[cone_shift_index] < upper_pos && position_v[cone_shift_index] > lower_pos) {
					// inside the conde, update the slope
					if (position_v[cone_shift_index] + t_abs < upper_pos) {
						// update slope_high
						slope_high = (position_v[cone_shift_index] + t_abs - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
					}
					if (position_v[cone_shift_index] - t_abs > lower_pos) {
						// update slope_low
						slope_low = (position_v[cone_shift_index] - t_abs - position_v[cone_origin_index]) / (key_v[cone_shift_index] - key_v[cone_origin_index]);
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
			SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
			//cout << "current max error: " << current_absolute_accuracy << endl;
			while (current_absolute_accuracy < t_abs && shift_index < TOTAL_SIZE - 1) {
				Co *= 2;
				shift_index = origin_index + Co;
				if (shift_index >= TOTAL_SIZE) {
					shift_index = TOTAL_SIZE - 1;
				}
				SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
				//cout << "current max error: " << current_absolute_accuracy << endl;
			}

			// 3. segment binary search (BETWEEN origin_index + Co/2 and origin_index + Co)

			//cout << "current max error: " << current_absolute_accuracy << endl;
			if (current_absolute_accuracy < t_abs && shift_index == TOTAL_SIZE - 1) {
				// stop
			}
			else {
				int Ilow = origin_index + Co / 2, Ihigh = shift_index, Middle;
				while (Ihigh - Ilow > 1) {
					Middle = (Ilow + Ihigh) / 2;

					shift_index = Middle;

					SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
					//cout << "current max error: " << current_absolute_accuracy << endl;
					if (current_absolute_accuracy > t_abs) {
						Ihigh = Middle;
					}
					else {
						Ilow = Middle;
					}
				}
				if (current_absolute_accuracy > t_abs) {
					shift_index = Ilow;
					SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
				}
			}
			//cout << "current max error: " << current_absolute_accuracy << endl;

			/*model.clear();
			for (int i = 0; i <= highest_term; i++) {
				model.push_back(paras[i]);
			}*/
			model_parameters.push_back(model);
			dataset_range.push_back(pair<double, double>(key_v[origin_index], key_v[shift_index]));
			index_range.push_back(pair<int, int>(origin_index, shift_index));
			//cout << "segment: " << segment_count << " segment size: " << shift_index - origin_index << endl;
			segment_count++;

			// prepare for the next segment
			origin_index = shift_index + 1; // verify whether + 1 is necessary here !
		}
	}

	void BuildNonLeafLayerWithBtree() {
		bottom_layer_index.clear();
		for (int i = 0; i < dataset_range.size(); i++) {
			bottom_layer_index.insert(pair<double, int>(dataset_range[i].first, i));
		}
	}

	void PredictWithStxBtree(vector<double> &queryset, vector<int> &results) {
		stx::btree<double, int>::iterator iter;
		results.clear();
		int model_index = 0;
		double result, key;
		for (int i = 0; i < queryset.size(); i++) {
			if (i == 8 || i==19 || i==26) {
				cout << "debug here!" << endl;
			}
			iter = this->bottom_layer_index.upper_bound(queryset[i]);
			iter--;
			model_index = iter->second;
			cout << queryset[i] << " " << model_index << endl;
			result = 0;
			key = 1;
			for (int j = 0; j <= highest_term; j++) {
				result += model_parameters[model_index][j] * key;
				key *= queryset[i];
				cout << "result: " << result << "   key:" << key << endl;
			}
			results.push_back(result);
		}
	}

	double t_abs; // the absolute error threshold
	double t_rel; // the relative error threshold
	int highest_term = 1; // the highest term of the model
	vector<pair<double, double>> dataset_range; // the keys
	vector<vector<double>> model_parameters; // for an * x^n + an-1 * x^n-1 + ... + a1x + a0, store a0, a1, ...,an
	vector<pair<int, int>> index_range; // the range of keys' index
	stx::btree<double, int> bottom_layer_index; // using a btree to index model (key, model_index)
};