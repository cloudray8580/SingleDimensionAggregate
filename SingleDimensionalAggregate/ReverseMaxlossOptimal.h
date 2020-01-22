#pragma once

#include<vector>
#include "STXBtreeAggregate.h"
#include "mlpack/core.hpp"
#include <ilcplex/ilocplex.h>
#include "Utils.h"

using namespace std;

// sort according to their deviation, put this outside the class
bool cmp_magnitude1(pair<int, double> &p1, pair<int, double> &p2) {
	return p1.second > p2.second;
}

class ReverseMaxlossOptimal{
public:

	ReverseMaxlossOptimal(double t_abs = 100, double t_rel = 0.01, int highest_term = 1) {
		this->t_abs = t_abs;
		this->t_rel = t_rel;
		this->highest_term = highest_term;
	}

	// our proposed approximation algorithm, now only for d = 1, i.e., linear
	// for threshold: -1: do not perform thresholding; others: that specified threshold
	// for sampled points: option = 0: using LP; option = 1: using MSE
	void SketchFiting(const vector<double> &key_v, const vector<double> &position_v, int lower_index, int higher_index, vector<double> &paras, double &loss, int threshold = -1, int option=0) {
		
		arma::mat dataset;
		arma::rowvec responses; // count

		dataset.set_size(higher_index - lower_index + 1);
		int temp_index = lower_index;
		dataset.imbue([&]() { return key_v[temp_index++]; });

		responses.set_size(higher_index - lower_index + 1);
		temp_index = lower_index;
		responses.imbue([&]() { return position_v[temp_index++]; });

		dataset = dataset.t();

		/*cout << "dataset size: " << dataset.size() << " cols: " << dataset.n_cols << " rows: " << dataset.n_rows << endl;
		cout << "response size: " << responses.size() << endl;
		for (int i = 0; i < dataset.size(); i++) {
			cout << dataset[i] << " " << responses[i] << endl;
		}*/

		LinearRegression lr(dataset, responses);
		arma::vec lr_paras = lr.Parameters();
		double a, b;
		a = lr_paras[1]; // a
		b = lr_paras[0]; // b

		//cout << "lr.paras: " << lr.Parameters() << endl;

		// go through the segment to record the local maximum (in positive region) and local minimum (in negative region) points
		double predicted_position, signed_error;
		double negative_max_error = 0, positive_max_error = 0;
		double negative_max_index, positive_max_index;

		vector<pair<int, double>> sampled_records; // index, error

		for (int i = lower_index+1; i < higher_index; i++) {
			predicted_position = a * key_v[i] + b;
			signed_error = position_v[i] - predicted_position;
			
			if (signed_error < negative_max_error) {
				negative_max_error = signed_error;
				negative_max_index = i;
			} 
			else if(signed_error < 0) {
				// save the previous negative max
				sampled_records.push_back(pair<double, double>(negative_max_index, negative_max_error));
			}
			else if (signed_error > positive_max_error) {
				positive_max_error = signed_error;
				positive_max_index = i;
			}
			else if (signed_error > 0) {
				// save the previous  positive max
				sampled_records.push_back(pair<double, double>(positive_max_index, positive_max_error));
			}
		}

		cout << "lower index: " << lower_index << "    higher_index: " << higher_index << endl;
		cout << "positive max error: " << positive_max_error << "  negative max error: " << negative_max_error << endl;
		cout << "positive max index: " << positive_max_index << "  negative max index: " << negative_max_index << endl;
		cout << "a: " << a << " b: " << b << endl;
		cout << "number of points sampled: " << sampled_records.size() << endl;

		// filter the sampled points
		if (threshold != -1) { // if -1, do not do this thresholding
			std::sort(sampled_records.begin(), sampled_records.end(), cmp_magnitude1);
		}
		
		sampled_records.push_back(pair<double, double>(lower_index, 0)); // put the first point
		sampled_records.push_back(pair<double, double>(higher_index, 0)); // put the last point

		switch (option) {
		case 0:	// option1: try LP for sampled points

			SolveMaxlossLPForApproximation(key_v, position_v, sampled_records, threshold, paras, loss);

			break;
		case 1:	// option2: try MSE for sampled points

			dataset.clear();
			responses.clear();

			int total_size = sampled_records.size();
			if (threshold != -1) {
				total_size = threshold;
			}
			dataset.set_size(total_size);
			responses.set_size(total_size);
			temp_index = 0;
			dataset.imbue([&]() { return key_v[sampled_records[temp_index++].first]; });
			temp_index = 0;
			responses.imbue([&]() { return key_v[sampled_records[temp_index++].first]; });

			LinearRegression lr2(dataset, responses);
			lr_paras = lr2.Parameters();
			a = paras[1]; // a
			b = paras[0]; // b

			paras.clear();
			paras.push_back(a);
			paras.push_back(b);

			// find loss
			double max_error, absolute_error;
			for (int i = lower_index; i <= higher_index; i++) {
				predicted_position = a * key_v[i] + b;
				absolute_error = abs(position_v[i] - predicted_position);
				if (absolute_error > max_error) {
					max_error = absolute_error;
				}
			}
			loss = max_error;

			// perform a bias balance operation?

			break;
		}
	}

	// designed for the sketch sampling method
	void SolveMaxlossLPForApproximation(const vector<double> &key_v, const vector<double> &position_v, vector<pair<int, double>> &indexes, int threshold, vector<double> &paras, double &loss) {
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
		cplex.setParam(IloCplex::Param::Advance, 0); // turnoff advanced start
		cplex.setParam(IloCplex::Param::Preprocessing::Presolve, false); // turnoff presolve

		cplex.setParam(IloCplex::Param::Barrier::Limits::Growth, 1e6);
		cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility, 1e-9);
		cplex.setParam(IloCplex::Param::Barrier::ConvergeTol, 1e-12);
		cplex.setParam(IloCplex::Param::Read::Scale, 1);
		cplex.setParam(IloCplex::Param::Simplex::Tolerances::Markowitz, 0.99999);
		cplex.setParam(IloCplex::Param::MIP::Tolerances::Integrality, 0.0);

		cplex.setParam(IloCplex::RootAlg, IloCplex::Primal); // using simplex
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Dual); // using dual simplex
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Barrier); // set optimizer used interior point method
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Sifting); // set optimizer used sifting method
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Concurrent);


		// set variable type, IloNumVarArray starts from 0.
		for (int i = 0; i <= this->highest_term; i++) {
			vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT));  // the weight, i.e., a2 for x^2, start from the lowest, i.e., a0, to an
		}
		IloNumVar target(env, 0.0, INFINITY, ILOFLOAT); // the max loss value

		// declare objective
		obj.setExpr(target);
		obj.setSense(IloObjective::Minimize);
		model.add(obj);

		double key = 1;

		// add constraint for each record

		int total_size = indexes.size();
		if (threshold != -1) {
			total_size = threshold;
		}
		int current_index;
		for (int i = 0; i < total_size; i++) {
			current_index = indexes[i].first;
			IloExpr model_term(env);
			key = 1;
			for (int j = 0; j <= highest_term; j++) {
				model_term += vars[j] * key;
				key *= key_v[current_index];
			}
			model.add(model_term - position_v[current_index] <= target);
			model.add(model_term - position_v[current_index] >= -target);
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

		env.end();
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
		cplex.setParam(IloCplex::Param::Advance, 0); // turnoff advanced start
		cplex.setParam(IloCplex::Param::Preprocessing::Presolve, false); // turnoff presolve

		cplex.setParam(IloCplex::Param::Barrier::Limits::Growth, 1e6);
		cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility, 1e-9);
		cplex.setParam(IloCplex::Param::Barrier::ConvergeTol, 1e-12);
		cplex.setParam(IloCplex::Param::Read::Scale, 1);
		cplex.setParam(IloCplex::Param::Simplex::Tolerances::Markowitz, 0.99999);
		cplex.setParam(IloCplex::Param::MIP::Tolerances::Integrality, 0.0);

		cplex.setParam(IloCplex::RootAlg, IloCplex::Primal); // using simplex
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Dual); // using dual simplex
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Barrier); // set optimizer used interior point method
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Sifting); // set optimizer used sifting method
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Concurrent);


		// set variable type, IloNumVarArray starts from 0.
		for (int i = 0; i <= this->highest_term; i++) {
			vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT));  // the weight, i.e., a2 for x^2, start from the lowest, i.e., a0, to an
		}
		IloNumVar target(env, 0.0, INFINITY, ILOFLOAT); // the max loss value


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
	// option = 0: using LP; option = 1: using sketch fiting
	void SegmentOnTrainMaxLossModel(const arma::mat& dataset, const arma::rowvec& labels, int option = 0) {

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
			if (option == 0) {
				SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
			}
			else if (option == 1) {
				SketchFiting(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
			}
			
			//cout << "current max error: " << current_absolute_accuracy << endl;
			while (current_absolute_accuracy < t_abs && shift_index < TOTAL_SIZE - 1) {
				Co *= 2;
				shift_index = origin_index + Co;
				if (shift_index >= TOTAL_SIZE) {
					shift_index = TOTAL_SIZE - 1;
				}
				if (option == 0) {
					SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
				}
				else if (option == 1) {
					SketchFiting(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
				}
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

					if (option == 0) {
						SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
					}
					else if (option == 1) {
						SketchFiting(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
					}
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
					if (option == 0) {
						SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
					}
					else if (option == 1) {
						SketchFiting(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
					}
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

	void SegmentOnTrainMaxLossModel(vector<double> &key_v, vector<double> &position_v, int option = 0) {

		model_parameters.clear();
		dataset_range.clear();
		index_range.clear();

		/*vector<double> key_v, position_v;
		RowvecToVector(dataset, key_v);
		RowvecToVector(labels, position_v);*/

		int TOTAL_SIZE = key_v.size(); // 1157570

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
			if (option == 0) {
				SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
			}
			else if (option == 1) {
				SketchFiting(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
			}

			//cout << "current max error: " << current_absolute_accuracy << endl;
			while (current_absolute_accuracy < t_abs && shift_index < TOTAL_SIZE - 1) {
				Co *= 2;
				shift_index = origin_index + Co;
				if (shift_index >= TOTAL_SIZE) {
					shift_index = TOTAL_SIZE - 1;
				}
				if (option == 0) {
					SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
				}
				else if (option == 1) {
					SketchFiting(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
				}
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

					if (option == 0) {
						SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
					}
					else if (option == 1) {
						SketchFiting(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
					}
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
					if (option == 0) {
						SolveMaxlossLP(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
					}
					else if (option == 1) {
						SketchFiting(key_v, position_v, origin_index, shift_index, model, current_absolute_accuracy);
					}
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

	// need to provide the histogram segmentation first, splits are split positions, the index
	// using Polyfit model, i.e., the Linear prgramming model
	void SegmentWithHistogram(const arma::mat& dataset, const arma::rowvec& labels, vector<int>& splits) {
		
		model_parameters.clear();
		dataset_range.clear();
		index_range.clear();
		absolute_errors.clear();

		vector<double> key_v, position_v;
		RowvecToVector(dataset, key_v);
		RowvecToVector(labels, position_v);

		int start_index = 0;
		int end_index = 0;
		vector<double> model;
		double current_absolute_accuracy;
		double max_abs_error = 0;

		for (int i = 0; i < splits.size(); i++) {

			// start and end index
			start_index = splits[i];
			if (i + 1 < splits.size()) {
				end_index = splits[i + 1];
			}
			else {
				end_index = key_v.size() - 1;
			}

			// call polyfit on the segment
			SolveMaxlossLP(key_v, position_v, start_index, end_index, model, current_absolute_accuracy);
			if (current_absolute_accuracy > max_abs_error) {
				max_abs_error = current_absolute_accuracy;
			}
			absolute_errors.push_back(current_absolute_accuracy);

			// add the model info
			model_parameters.push_back(model);
			dataset_range.push_back(pair<double, double>(key_v[start_index], key_v[end_index]));
			index_range.push_back(pair<int, int>(start_index, end_index));
		}

		this->max_absolute_error = max_abs_error;

		cout << "#splits: " << splits.size() << "  max_abs_err: " << max_abs_error << endl;
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
			/*if (i == 8 || i==19 || i==26) {
				cout << "debug here!" << endl;
			}*/
			iter = this->bottom_layer_index.upper_bound(queryset[i]);
			iter--;
			model_index = iter->second;
			cout << queryset[i] << " " << model_index << endl;
			result = 0;
			key = 1;
			/*for (int j = 0; j <= highest_term; j++) {
				result += model_parameters[model_index][j] * key;
				key *= queryset[i];
				cout << "result: " << result << "   key:" << key << endl;
			}*/
			results.push_back(result);
		}
	}

	void PrepareMaxAggregateTree() {
		// generate the aggregate_max of the tree
		// also generate the max min key boundary
		bottom_layer_index.generate_max_aggregate();

		// but the max aggregate value is not correct, need to change it to
	}

	QueryResult MaxPredictionWithoutRefinement(vector<double> &queryset_low, vector<double> &queryset_up, vector<double> &results, vector<double> &key_v) {
		results.clear();
		
		int model_index = 0;
		double result, result_low, result_up, key_low, key_up;

		// retrieve all the slopes, assume only in 1-dimensional data
		vector<double> slopes;
		for (int i = 0; i < model_parameters.size(); i++) {
			slopes.push_back(model_parameters[i][1]); // only consider 1-dimension
		}

		bottom_layer_index.AssignSlopeToLeafNodes(slopes);

		stx::btree<double, int>::iterator iter;

		auto t0 = chrono::steady_clock::now();

		double max_lower, max_upper, side_max;
		double left_boundary, right_boundary;

		// for each range query pair
		for (int i = 0; i < queryset_low.size(); i++) {

			// handle left boundary, this part could be actually embed into the tree, and hence the time for a traverse with tree height n-1 could be deducted
			iter = this->bottom_layer_index.upper_bound(queryset_low[i]);
			iter--;
			model_index = iter->second;
			left_boundary = dataset_range[model_index].second;
			if (slopes[model_index] > 0) {
				// prediction at right boundary
				max_lower = dataset_range[model_index].second * model_parameters[model_index][1] + model_parameters[model_index][0];
			}
			else {
				// prediction at query key
				max_lower = queryset_low[i] * model_parameters[model_index][1] + model_parameters[model_index][0];
			}

			// handle right boundary, this part could be actually embed into the tree, and hence the time for a traverse with tree height n-1 could be deducted
			iter = this->bottom_layer_index.upper_bound(queryset_low[i]);
			iter--;
			model_index = iter->second;
			right_boundary = dataset_range[model_index].first;
			if (slopes[model_index] > 0) {
				// prediction at query key
				max_upper = queryset_low[i] * model_parameters[model_index][1] + model_parameters[model_index][0];
			}
			else {
				// prediction at left boundary
				max_upper = dataset_range[model_index].first * model_parameters[model_index][1] + model_parameters[model_index][0];
			}
			
			side_max = max_lower > max_upper ? max_lower : max_upper;

			// handle middle
			result = bottom_layer_index.our_max_query(left_boundary, right_boundary); // such that the query do not handle both sides

			result = side_max > result ? side_max : result;
			results.push_back(result);
		}
		auto t1 = chrono::steady_clock::now();
		/*cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
		cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset_low.size()) << " ns" << endl;*/

		auto average_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / queryset_low.size();
		auto total_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();

		double MEabs, MErel;
		MeasureAccuracy(results, "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/HKI_MAX.csv", MEabs, MErel);

		QueryResult query_result;
		query_result.average_query_time = average_time;
		query_result.total_query_time = total_time;
		query_result.measured_absolute_error = MEabs;
		query_result.measured_relative_error = MErel;

		return query_result;

		// record experiment result;
		ofstream outfile_exp;
		outfile_exp.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RunResults/MAX.csv", std::ios_base::app);
		outfile_exp << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset_low.size()) << endl;
		outfile_exp << endl;
		outfile_exp.close();
	}

	// without refinement
	void CountPredictionWithoutRefinement(vector<double> &queryset_low, vector<double> &queryset_up, vector<int> &results, vector<double> &key_v, string recordfilepath = "C:/Users/Cloud/Desktop/LearnedAggregateData/experiment_result.csv") {

		results.clear();
		stx::btree<double, int>::iterator iter;
		int model_index = 0;
		double result, result_low, result_up, key_low, key_up;

		auto t0 = chrono::steady_clock::now();
		// for each range query pair
		for (int i = 0; i < queryset_low.size(); i++) {

			// calculate the lower key position
			iter = this->bottom_layer_index.upper_bound(queryset_low[i]);
			iter--;
			model_index = iter->second;
			//cout << queryset_low[i] << " " << model_index << endl;
			result_low = 0;
			key_low = 1;
			for (int j = 0; j <= highest_term; j++) {
				result_low += model_parameters[model_index][j] * key_low;
				key_low *= queryset_low[i];
				//cout << "result: " << result << "   key:" << key << endl;
			}

			// calculate the upper key position
			iter = this->bottom_layer_index.upper_bound(queryset_up[i]);
			iter--;
			model_index = iter->second;
			//cout << queryset_up[i] << " " << model_index << endl;
			result_up = 0;
			key_up = 1;
			for (int j = 0; j <= highest_term; j++) {
				result_up += model_parameters[model_index][j] * key_up;
				key_up *= queryset_up[i];
				//cout << "result: " << result << "   key:" << key << endl;
			}

			// calculate the COUNT
			result = result_up - result_low;
			results.push_back(result);
		}
		auto t1 = chrono::steady_clock::now();
		cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
		cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset_low.size()) << " ns" << endl;

		// record experiment result;
		ofstream outfile_exp;
		outfile_exp.open(recordfilepath, std::ios_base::app);
		outfile_exp << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset_low.size()) << endl;
		outfile_exp.close();
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

		auto t0 = chrono::steady_clock::now();
		// for each range query pair
		for (int i = 0; i < queryset_low.size(); i++) {

			// calculate the lower key position
			iter = this->bottom_layer_index.upper_bound(queryset_low[i]);
			iter--;
			model_index = iter->second;
			//cout << queryset_low[i] << " " << model_index << endl;
			result_low = 0;
			key_low = 1;
			for (int j = 0; j <= highest_term; j++) {
				result_low += model_parameters[model_index][j] * key_low;
				key_low *= queryset_low[i];
				//cout << "result: " << result << "   key:" << key << endl;
			}
			
			// calculate the upper key position
			iter = this->bottom_layer_index.upper_bound(queryset_up[i]);
			iter--;
			model_index = iter->second;
			//cout << queryset_up[i] << " " << model_index << endl;
			result_up = 0;
			key_up = 1;
			for (int j = 0; j <= highest_term; j++) {
				result_up += model_parameters[model_index][j] * key_up;
				key_up *= queryset_up[i];
				//cout << "result: " << result << "   key:" << key << endl;
			}

			// calculate the COUNT
			result = result_up - result_low;

			// analysis estimated maximum relative error:
			max_err_rel = (2 * t_abs) / (result - 2 * t_abs);
			if (max_err_rel > t_rel || max_err_rel < 0) {
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
		ofstream outfile_exp;
		outfile_exp.open(recordfilepath, std::ios_base::app);
		outfile_exp << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset_low.size()) << "," << 1000-count_refinement << endl;
		outfile_exp.close();
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

		auto t0 = chrono::steady_clock::now();
		// for each range query pair
		for (int i = 0; i < queryset_low.size(); i++) {

			// calculate the lower key position
			iter = this->bottom_layer_index.upper_bound(queryset_low[i]);
			iter--;
			model_index = iter->second;
			//cout << queryset_low[i] << " " << model_index << endl;
			result_low = 0;
			key_low = 1;
			for (int j = 0; j <= highest_term; j++) {
				result_low += model_parameters[model_index][j] * key_low;
				key_low *= queryset_low[i];
				//cout << "result: " << result << "   key:" << key << endl;
			}

			// calculate the upper key position
			iter = this->bottom_layer_index.upper_bound(queryset_up[i]);
			iter--;
			model_index = iter->second;
			//cout << queryset_up[i] << " " << model_index << endl;
			result_up = 0;
			key_up = 1;
			for (int j = 0; j <= highest_term; j++) {
				result_up += model_parameters[model_index][j] * key_up;
				key_up *= queryset_up[i];
				//cout << "result: " << result << "   key:" << key << endl;
			}

			// calculate the COUNT
			result = result_up - result_low;

			// analysis estimated maximum relative error:
			max_err_rel = (2 * t_abs) / (result - 2 * t_abs);
			if (max_err_rel > t_rel || max_err_rel < 0) {
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
		cout << "hit probability: " << 1000 - count_refinement << " / 1000" << endl;*/

		auto average_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / queryset_low.size();
		auto total_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();

		double MEabs, MErel;
		MeasureAccuracy(results, "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/TWEET_1D.csv", MEabs, MErel);

		QueryResult query_result;
		query_result.average_query_time = average_time;
		query_result.total_query_time = total_time;
		query_result.measured_absolute_error = MEabs;
		query_result.measured_relative_error = MErel;

		return query_result;
	}

	// using max absolute error as T.abs for all bins
	void CountPredictionHist(vector<double> &queryset_low, vector<double> &queryset_up, vector<int> &results, vector<double> &key_v, string recordfilepath = "C:/Users/Cloud/Desktop/LearnedAggregateData/experiment_result.csv") {

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

		auto t0 = chrono::steady_clock::now();
		// for each range query pair
		for (int i = 0; i < queryset_low.size(); i++) {

			// calculate the lower key position
			iter = this->bottom_layer_index.upper_bound(queryset_low[i]);
			iter--;
			model_index = iter->second;
			//cout << queryset_low[i] << " " << model_index << endl;
			result_low = 0;
			key_low = 1;
			for (int j = 0; j <= highest_term; j++) {
				result_low += model_parameters[model_index][j] * key_low;
				key_low *= queryset_low[i];
				//cout << "result: " << result << "   key:" << key << endl;
			}

			// calculate the upper key position
			iter = this->bottom_layer_index.upper_bound(queryset_up[i]);
			iter--;
			model_index = iter->second;
			//cout << queryset_up[i] << " " << model_index << endl;
			result_up = 0;
			key_up = 1;
			for (int j = 0; j <= highest_term; j++) {
				result_up += model_parameters[model_index][j] * key_up;
				key_up *= queryset_up[i];
				//cout << "result: " << result << "   key:" << key << endl;
			}

			// calculate the COUNT
			result = result_up - result_low;

			// analysis estimated maximum relative error:
			max_err_rel = (2 * max_absolute_error) / (result - 2 * max_absolute_error);
			if (max_err_rel > t_rel || max_err_rel < 0) {
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
		ofstream outfile_exp;
		outfile_exp.open(recordfilepath, std::ios_base::app);
		outfile_exp << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset_low.size()) << "," << 1000 - count_refinement << endl;
		outfile_exp.close();
	}

	// using each bin's absolute error for relative error condition check
	void CountPredictionHistOptimized(vector<double> &queryset_low, vector<double> &queryset_up, vector<int> &results, vector<double> &key_v, string recordfilepath = "C:/Users/Cloud/Desktop/LearnedAggregateData/experiment_result.csv") {

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
		double abs_err_l, abs_err_u;

		auto t0 = chrono::steady_clock::now();
		// for each range query pair
		for (int i = 0; i < queryset_low.size(); i++) {

			// calculate the lower key position
			iter = this->bottom_layer_index.upper_bound(queryset_low[i]);
			iter--;
			model_index = iter->second;
			//cout << queryset_low[i] << " " << model_index << endl;
			result_low = 0;
			key_low = 1;
			for (int j = 0; j <= highest_term; j++) {
				result_low += model_parameters[model_index][j] * key_low;
				key_low *= queryset_low[i];
				//cout << "result: " << result << "   key:" << key << endl;
			}
			abs_err_l = absolute_errors[model_index];

			// calculate the upper key position
			iter = this->bottom_layer_index.upper_bound(queryset_up[i]);
			iter--;
			model_index = iter->second;
			//cout << queryset_up[i] << " " << model_index << endl;
			result_up = 0;
			key_up = 1;
			for (int j = 0; j <= highest_term; j++) {
				result_up += model_parameters[model_index][j] * key_up;
				key_up *= queryset_up[i];
				//cout << "result: " << result << "   key:" << key << endl;
			}
			abs_err_u = absolute_errors[model_index];

			// calculate the COUNT
			result = result_up - result_low;

			// analysis estimated maximum relative error:
			max_err_rel = (abs_err_l + abs_err_u) / (result - abs_err_l - abs_err_u);
			//max_err_rel = (2 * max_absolute_error) / (result - 2 * max_absolute_error);
			if (max_err_rel > t_rel || max_err_rel < 0) {
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
		ofstream outfile_exp;
		outfile_exp.open(recordfilepath, std::ios_base::app);
		outfile_exp << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset_low.size()) << "," << 1000 - count_refinement << endl;
		outfile_exp.close();
	}

	// the following is the min max support
	void FastMaxSupport() {
		// 1. build aggregate max tree
		// 2. change the bottom node into linkers
	}

	double t_abs; // the absolute error threshold
	double t_rel; // the relative error threshold
	int highest_term = 1; // the highest term of the model
	vector<pair<double, double>> dataset_range; // the keys
	vector<vector<double>> model_parameters; // for an * x^n + an-1 * x^n-1 + ... + a1x + a0, store a0, a1, ...,an
	vector<pair<int, int>> index_range; // the range of keys' index
	stx::btree<double, int> bottom_layer_index; // using a btree to index model (key, model_index)
	
	vector<double> absolute_errors; // the absolute error for each histogram segment
 	double max_absolute_error; // for histogram segmentation
};