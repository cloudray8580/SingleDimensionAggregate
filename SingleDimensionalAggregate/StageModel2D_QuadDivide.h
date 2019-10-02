#pragma once
#include <vector>
#include <ilcplex/ilocplex.h> 
using namespace std;

class StageModel2D_QuadDivide {
public:

	StageModel2D_QuadDivide(double error_threshold) {
		this->error_threshold = error_threshold;
	}

	// the higher index is included
	// @keys_v: firsr: key value of first dimension; second:key value of second dimension
	// @accumulation_v: accumulation value of the target dimension (count or sum)

	void MyCplexSolverForMaxLossOptimized2D(vector<pair<double, double>> &keys_v, vector<double> &accumulation_v, int lower_index, int higher_index, double &a, double &b, double &loss) {
		IloEnv env;
		IloModel model(env);
		IloCplex cplex(model);
		IloObjective obj(env);
		IloNumVarArray vars(env);
		IloRangeArray ranges(env);

		cplex.setOut(env.getNullStream());     

		// set variable type, IloNumVarArray starts from 0.
		vars.add(IloNumVar(env, 0.0, INFINITY, ILOFLOAT)); // the weight, i.e., a
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT));      // the bias, i.e., b
		vars.add(IloNumVar(env, 0.0, INFINITY, ILOFLOAT)); // our target, the max loss

		cplex.setParam(IloCplex::RootAlg, IloCplex::Barrier); // set optimizer used interior point method
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Sifting); // set optimizer used interior point method

		// declare objective
		obj.setExpr(vars[2]);
		obj.setSense(IloObjective::Minimize);
		model.add(obj);


		// add constraint for each record
		for (int i = lower_index; i <= higher_index; i++) {
			model.add(vars[0] * key_v[i] + vars[1] - position_v[i] <= vars[2]);
			model.add(vars[0] * key_v[i] + vars[1] - position_v[i] >= -vars[2]);
		}

		IloNum starttime_ = cplex.getTime();
		cplex.solve();

		IloNum endtime_ = cplex.getTime();
		double target = cplex.getObjValue();
		double slope, bias;

		slope = cplex.getValue(vars[0]);
		bias = cplex.getValue(vars[1]);

		//cout << "the variable a: " << cplex.getValue(vars[0]) << endl;
		//cout << "the variable b: " << cplex.getValue(vars[1]) << endl;
		//cout << "the variable max loss: " << cplex.getValue(vars[2]) << endl;
		//cout << "cplex solve time: " << endtime_ - starttime_ << endl;

		env.end();

		//return target;
		a = slope;
		b = bias;
		loss = target;
	}

	// using equal width sampling,
	void TrainModel(vector<vector<double>> &sampled_accu_count) {

	}


	double error_threshold;
	vector<vector<vector<double>>> model_parameters;  // the ith 1 dimension and jth 2 dimension model's parameter, 
	vector<vector<vector<double>>> model_domain_range; // the ith 1 dimension and jth 2 dimension model's domain range 
}