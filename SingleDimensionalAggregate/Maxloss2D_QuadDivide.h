#pragma once
#include <iostream>
#include <vector>
//#include <ilcplex/ilocplex.h>
#include "mlpack/core.hpp"

#include <boost/geometry.hpp> // annotate this for ambiguous rank, or adjust the initilization list with normal initilization way
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/geometry/geometries/register/box.hpp> // to register box, remember to include this!!!
#include <boost/function_output_iterator.hpp>

#include "RTree.h"
#include "Utils.h"

using namespace std;
using namespace mlpack;

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
namespace bgid = boost::geometry::index::detail;

typedef bg::model::d2::point_xy<double> MyPoint; // need to include point_xy.hpp
typedef bg::model::box<MyPoint> Box;
//typedef std::pair<Box, unsigned> value;

struct ModelBox{
public:
	MyPoint P_low, P_up; // the lower left and upper right corner
	ModelBox(double x1, double y1, double x2, double y2) : P_low(x1, y1),P_up(x2, y2) {}
	double d1_lower, d1_upper, d2_lower, d2_upper;
	double a2, a1, b2, b1, c, bias;
	double loss;
	double level;
};

BOOST_GEOMETRY_REGISTER_BOX(ModelBox, MyPoint, P_low, P_up)


class Maxloss2D_QuadDivide {
public:

	Maxloss2D_QuadDivide(double error_threshold, double relative_error_threshold, double min_d1 = -90, double max_d1 = 90, double min_d2 = -180, double max_d2 = 180) {
		this->error_threshold = error_threshold;
		this->relative_error_threshold = relative_error_threshold;
		this->domain_min_d1 = min_d1;
		this->domain_max_d1 = max_d1;
		this->domain_min_d2 = min_d2;
		this->domain_max_d2 = max_d2;
	}

	// the higher index is included
	// @keys_v: firsr: key value of first dimension; second:key value of second dimension
	// @accumulation_v: accumulation value of the target dimension (count or sum)

	// M = a2x^2 + a1x + b2y^2 + b1y + cxy + d
	void SolveMaxlossLP2D(vector<pair<double, double>> &keys_v, vector<double> &accumulation_v, double lower_D1, double upper_D1, double lower_D2, double upper_D2, double &a2, double &a1, double &b2, double &b1, double &c, double &d, double &loss) {
		IloEnv env;
		IloModel model(env);
		IloCplex cplex(model);
		IloObjective obj(env);
		IloNumVarArray vars(env);
		IloRangeArray ranges(env);

		//cplex.setOut(env.getNullStream());     

		cplex.setParam(IloCplex::NumericalEmphasis, CPX_ON);
		/*cplex.setParam(IloCplex::Param::Barrier::Limits::Growth, 1e6);
		cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility, 1e-9);
		cplex.setParam(IloCplex::Param::Barrier::ConvergeTol, 1e-12);
		cplex.setParam(IloCplex::Param::Read::Scale, 1);
		cplex.setParam(IloCplex::Param::Simplex::Tolerances::Markowitz, 0.99999);
		cplex.setParam(IloCplex::Param::MIP::Tolerances::Integrality, 0.0);*/

		// set variable type, IloNumVarArray starts from 0.
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // 0, the weight, i.e., a2 for x^2
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // 1, the weight, i.e., a1 for x
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // 2, the weight, i.e., b2 for y^2
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // 3, the weight, i.e., b1 for y
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // 4, the weight, i.e., c for xy
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // 5, the bias, i.e., d
		vars.add(IloNumVar(env, 0.0, INFINITY, ILOFLOAT)); // 6, our target, the max loss

		//cplex.setParam(IloCplex::RootAlg, IloCplex::Primal); // using simplex
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Barrier); // set optimizer used interior point method
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Sifting); // set optimizer used interior point method

		// declare objective
		obj.setExpr(vars[6]);
		obj.setSense(IloObjective::Minimize);
		model.add(obj);

		int count_constraint = 0;

		// add constraint for each record
		for (int i = 0; i < keys_v.size(); i++) {
			if (keys_v[i].first >= lower_D1 && keys_v[i].first <= upper_D1 && keys_v[i].second >= lower_D2 && keys_v[i].second <= upper_D2) {
				count_constraint++;
				model.add(vars[0] * keys_v[i].first * keys_v[i].first + vars[1] * keys_v[i].first + vars[2] * keys_v[i].second * keys_v[i].second + vars[3] * keys_v[i].second + vars[4] * keys_v[i].first * keys_v[i].second + vars[5] - accumulation_v[i] <= vars[6]);
				model.add(vars[0] * keys_v[i].first * keys_v[i].first + vars[1] * keys_v[i].first + vars[2] * keys_v[i].second * keys_v[i].second + vars[3] * keys_v[i].second + vars[4] * keys_v[i].first * keys_v[i].second + vars[5] - accumulation_v[i] >= -vars[6]);
			}
			else {
				cout << "invalid point " << i << ": " << keys_v[i].first << " " << keys_v[i].second << endl;
			}
		}

		IloNum starttime_ = cplex.getTime();
		cplex.solve();
		IloNum endtime_ = cplex.getTime();
		
		cplex.exportModel("C:/Users/Cloud/Desktop/2D.lp");

		try {
			loss = cplex.getObjValue();
			a2 = cplex.getValue(vars[0]);
			a1 = cplex.getValue(vars[1]);
			b2 = cplex.getValue(vars[2]);
			b1 = cplex.getValue(vars[3]);
			c = cplex.getValue(vars[4]);
			d = cplex.getValue(vars[5]);
		}
		catch (IloException &e) {
			std::cerr << "iloexception: " << e << endl;
			loss = -1; //  indicate the model is invalid
			a2 = 0;
			a1 = 0;
			b2 = 0;
			b1 = 0;
			c = 0;
			d = 0;
			return;
		}
		catch (IloAlgorithm::NotExtractedException &e) {
			loss = -1; //  indicate the model is invalid
			a2 = 0;
			a1 = 0;
			b2 = 0;
			b1 = 0;
			c = 0;
			d = 0;
			return;
		}

		//cout << "the variable a2: " << cplex.getValue(vars[0]) << endl;
		//cout << "the variable a1: " << cplex.getValue(vars[1]) << endl;
		//cout << "the variable b2: " << cplex.getValue(vars[2]) << endl;
		//cout << "the variable b1: " << cplex.getValue(vars[3]) << endl;
		//cout << "the variable c: " << cplex.getValue(vars[4]) << endl;
		//cout << "the variable bias: " << cplex.getValue(vars[5]) << endl;
		//cout << "the variable max loss: " << cplex.getValue(vars[6]) << endl;
		//cout << "the max loss: " << cplex.getValue(vars[6]) << endl;
		env.end();
	}

	// M = a2x^2 + a1x + b2y^2 + b1y + c (optimized for calculating derivative)
	void SolveMaxlossLP2DSimplified(vector<pair<double, double>> &keys_v, vector<double> &accumulation_v, double lower_D1, double upper_D1, double lower_D2, double upper_D2, double &a2, double &a1, double &b2, double &b1, double &c, double &loss) {
		IloEnv env;
		IloModel model(env);
		IloCplex cplex(model);
		IloObjective obj(env);
		IloNumVarArray vars(env);
		IloRangeArray ranges(env);

		cplex.setOut(env.getNullStream());

		/*cplex.setParam(IloCplex::NumericalEmphasis, CPX_ON);
		cplex.setParam(IloCplex::Param::Barrier::Limits::Growth, 1e6);
		cplex.setParam(IloCplex::Param::Simplex::Tolerances::Feasibility, 1e-9);
		cplex.setParam(IloCplex::Param::Barrier::ConvergeTol, 1e-12);
		cplex.setParam(IloCplex::Param::Read::Scale, 1);
		cplex.setParam(IloCplex::Param::Simplex::Tolerances::Markowitz, 0.99999);
		cplex.setParam(IloCplex::Param::MIP::Tolerances::Integrality, 0.0);*/

		cplex.setParam(IloCplex::RootAlg, IloCplex::AutoAlg);
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Primal); // using simplex
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Barrier); // set optimizer used interior point method
		//cplex.setParam(IloCplex::RootAlg, IloCplex::Sifting); // set optimizer used interior point method

		// set variable type, IloNumVarArray starts from 0.
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // 0, the weight, i.e., a2 for x^2
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // 1, the weight, i.e., a1 for x
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // 2, the weight, i.e., b2 for y^2
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // 3, the weight, i.e., b1 for y
		vars.add(IloNumVar(env, -INFINITY, INFINITY, ILOFLOAT)); // 4, the bias, i.e., c
		vars.add(IloNumVar(env, 0.0, INFINITY, ILOFLOAT)); // 5, our target, the max loss

		// declare objective
		obj.setExpr(vars[5]);
		obj.setSense(IloObjective::Minimize);
		model.add(obj);

		// add constraint for each record
		for (int i = 0; i <= keys_v.size(); i++) {
			if (keys_v[i].first >= lower_D1 && keys_v[i].first <= upper_D1 && keys_v[i].second >= lower_D2 && keys_v[i].second <= upper_D2) {
				model.add(vars[0] * keys_v[i].first * keys_v[i].first + vars[1] * keys_v[i].first + vars[2] * keys_v[i].second * keys_v[i].second + vars[3] * keys_v[i].second + vars[4] - accumulation_v[i] <= vars[5]);
				model.add(vars[0] * keys_v[i].first * keys_v[i].first + vars[1] * keys_v[i].first + vars[2] * keys_v[i].second * keys_v[i].second + vars[3] * keys_v[i].second + vars[4] - accumulation_v[i] >= -vars[5]);
			}
		}

		IloNum starttime_ = cplex.getTime();
		cplex.solve();
		IloNum endtime_ = cplex.getTime();

		//cplex.exportModel("C:/Users/Cloud/Desktop/NotExtractedException.lp");

		try {
			loss = cplex.getObjValue();
			a2 = cplex.getValue(vars[0]);
			a1 = cplex.getValue(vars[1]);
			b2 = cplex.getValue(vars[2]);
			b1 = cplex.getValue(vars[3]);
			c = cplex.getValue(vars[4]);
		}
		catch (IloException &e) {
			std::cerr << "iloexception: " << e << endl;
			loss = -1; //  indicate the model is invalid
			a2 = 0;
			a1 = 0;
			b2 = 0;
			b1 = 0;
			c = 0;
			return;
		}
		catch (IloAlgorithm::NotExtractedException &e) {
			loss = -1; //  indicate the model is invalid
			a2 = 0;
			a1 = 0;
			b2 = 0;
			b1 = 0;
			c = 0;
			return;
		}

		//cout << "the variable a2: " << cplex.getValue(vars[0]) << endl;
		//cout << "the variable a1: " << cplex.getValue(vars[1]) << endl;
		//cout << "the variable b2: " << cplex.getValue(vars[2]) << endl;
		//cout << "the variable b1: " << cplex.getValue(vars[3]) << endl;
		//cout << "the variable bias: " << cplex.getValue(vars[4]) << endl;
		//cout << "the variable max loss: " << cplex.getValue(vars[5]) << endl;
		//cout << "the max loss: " << cplex.getValue(vars[5]) << endl;
		env.end();
	}

	// using equal width sampling,
	void TrainModel(){
		//double a2, a1, b2, b1, c, bias, loss;
		//SolveMaxlossLP2D(keys_v, accumulation_v, -90, 90, -180, 180, a2, a1, b2, b1, c, bias, loss);
		
		cout << "start training models" << endl;
		model_rtree.clear();
		temp_models.clear();
		TrainModelSubRegion(domain_min_d1, domain_max_d1, domain_min_d2, domain_max_d2, 1);

		for (int i = 0; i < temp_models.size(); i++) {
			model_rtree.insert(std::make_pair(temp_models[i], i));
		}
		cout << "exit train models" << endl;
	}

	// binary
	void WriteTrainedModelsToFile(string filepath) {

		cout << "write models.." << endl;

		ofstream outfile;
		outfile.open(filepath, ios::binary);
		for (int i = 0; i < temp_models.size(); i++) {

			outfile.write((char*)&temp_models[i].d1_lower, sizeof(double));
			outfile.write((char*)&temp_models[i].d1_upper, sizeof(double));
			outfile.write((char*)&temp_models[i].d2_lower, sizeof(double));
			outfile.write((char*)&temp_models[i].d2_upper, sizeof(double));

			outfile.write((char*)&temp_models[i].a2, sizeof(double));
			outfile.write((char*)&temp_models[i].a1, sizeof(double));
			outfile.write((char*)&temp_models[i].b2, sizeof(double));
			outfile.write((char*)&temp_models[i].b1, sizeof(double));
			outfile.write((char*)&temp_models[i].c, sizeof(double));
			outfile.write((char*)&temp_models[i].bias, sizeof(double));

			outfile.write((char*)&temp_models[i].loss, sizeof(double));
			outfile.write((char*)&temp_models[i].level, sizeof(double));
		}

		cout << "finish write models.." << endl;
	}

	// binary
	void ReadTrainedModelsFromFile(string filepath) {

		temp_models.clear();

		ifstream infile;
		infile.open(filepath, ios::binary);
		double d1_lower, d2_lower, d1_upper, d2_upper;

		while (true) {

			infile.read((char*)&d1_lower, sizeof(double));
			infile.read((char*)&d1_upper, sizeof(double));
			infile.read((char*)&d2_lower, sizeof(double));
			infile.read((char*)&d2_upper, sizeof(double));

			ModelBox mb(d1_lower, d2_lower, d1_upper, d2_upper);
			mb.d1_lower = d1_lower;
			mb.d2_lower = d2_lower;
			mb.d1_upper = d1_upper;
			mb.d2_upper = d2_upper;

			infile.read((char*)&mb.a2, sizeof(double));
			infile.read((char*)&mb.a1, sizeof(double));
			infile.read((char*)&mb.b2, sizeof(double));
			infile.read((char*)&mb.b1, sizeof(double));
			infile.read((char*)&mb.c, sizeof(double));
			infile.read((char*)&mb.bias, sizeof(double));

			infile.read((char*)&mb.loss, sizeof(double));
			infile.read((char*)&mb.level, sizeof(double));

			temp_models.push_back(mb);
			if (!infile) {
				break;
			}
		}
	}

	// after read models from file, the boost Rtree
	void LoadRtree() {
		model_rtree.clear();
		cout << "model size: " << temp_models.size() << endl;
		for (int i = 0; i < temp_models.size(); i++) {
			model_rtree.insert(std::make_pair(temp_models[i], i));
		}
	}

	// level start from 1
	void TrainModelSubRegion(double d1_lower, double d1_upper, double d2_lower, double d2_upper, int level) {
		
		//cout << "level: " << level << endl;
		//cout << "range L1:" << d1_lower << "  L2: " << d2_lower << "  U1: " << d1_upper << "  U2: " << d2_upper << endl;
		double a2, a1, b2, b1, c, bias, loss;
		SolveMaxlossLP2D(keys_v, accumulation_v, d1_lower, d1_upper, d2_lower, d2_upper, a2, a1, b2, b1, c, bias, loss);

		//double a2, a1, b2, b1, bias, loss;
		//SolveMaxlossLP2DSimplified(keys_v, accumulation_v, d1_lower, d1_upper, d2_lower, d2_upper, a2, a1, b2, b1, bias, loss);
		// cout << "loss: " << loss << endl;

		// check the max loss
		// if greater, divide into 4 sub region and continue
		if (loss > error_threshold) {
			double d1_middle = (d1_lower + d1_upper) / 2;
			double d2_middle = (d2_lower + d2_upper) / 2;

			TrainModelSubRegion(d1_lower, d1_middle, d2_middle, d2_upper, level + 1); // upper left
			TrainModelSubRegion(d1_middle, d1_upper, d2_middle, d2_upper, level + 1); // upper right
			TrainModelSubRegion(d1_middle, d1_upper, d2_lower, d2_middle, level + 1); // lower right
			TrainModelSubRegion(d1_lower, d1_middle, d2_lower, d2_middle, level + 1); // lower left
		}
		else {
			// finish this sub region and save the model
			ModelBox mb(d1_lower, d2_lower, d1_upper, d2_upper);
			mb.d1_lower = d1_lower;
			mb.d1_upper = d1_upper;
			mb.d2_lower = d2_lower;
			mb.d2_upper = d2_upper;
			mb.a2 = a2;
			mb.a1 = a1;
			mb.b2 = b2;
			mb.b1 = b1;
			mb.c = c;
			mb.bias = bias;
			mb.loss = loss;
			mb.level = level; 
			temp_models.push_back(mb);
			cout << "current model amount: " << temp_models.size() << endl;
			//model_rtree.insert(mb); // insert it in TrainModel()
		}
	}

	// using full model (i.e., with xy term)
	void QueryPrediction(vector<double> &d1_low, vector<double> &d2_low, vector<double> &d1_up, vector<double> &d2_up, vector<double> &results) {
		
		results.clear();

		vector<pair<ModelBox, int>> result_add1;
		vector<pair<ModelBox, int>> result_add2;
		vector<pair<ModelBox, int>> result_minus1;
		vector<pair<ModelBox, int>> result_minus2;

		double pred1, pred2, pred3, pred4;

		for (int i = 0; i < d1_low.size(); i++) {
			MyPoint lower_left(d1_low[i], d2_low[i]); // add 1 
			MyPoint upper_right(d1_up[i], d2_up[i]); // add 2
			MyPoint upper_left(d1_low[i], d2_up[i]); // minus 1 
			MyPoint lower_right(d1_up[i], d2_low[i]); // minus 2

			model_rtree.query(bgi::intersects(lower_left), std::back_inserter(result_add1));
			model_rtree.query(bgi::intersects(upper_right), std::back_inserter(result_add2));
			model_rtree.query(bgi::intersects(upper_left), std::back_inserter(result_minus1));
			model_rtree.query(bgi::intersects(lower_right), std::back_inserter(result_minus2));

			pred1 = result_add1[0].first.a2 * d1_low[i] * d1_low[i] + result_add1[0].first.a1 * d1_low[i] + result_add1[0].first.b2 * d2_low[i] * d2_low[i] + result_add1[0].first.b1 * d2_low[i] + result_add1[0].first.c * d1_low[i] * d2_low[i] + result_add1[0].first.bias;

			pred2 = result_add2[0].first.a2 * d1_up[i] * d1_up[i] + result_add2[0].first.a1 * d1_up[i] + result_add2[0].first.b2 * d2_up[i] * d2_up[i] + result_add2[0].first.b1 * d2_up[i] + result_add2[0].first.c * d1_up[i] * d2_up[i] + result_add2[0].first.bias;

			pred3 = result_minus1[0].first.a2 * d1_low[i] * d1_low[i] + result_minus1[0].first.a1 * d1_low[i] + result_minus1[0].first.b2 * d2_up[i] * d2_up[i] + result_minus1[0].first.b1 * d2_up[i] + result_minus1[0].first.c * d1_low[i] * d2_up[i] + result_minus1[0].first.bias;

			pred4 = result_minus2[0].first.a2 * d1_up[i] * d1_up[i] + result_minus2[0].first.a1 * d1_up[i] + result_minus2[0].first.b2 * d2_low[i] * d2_low[i] + result_minus2[0].first.b1 * d2_low[i] + result_minus2[0].first.c * d1_up[i] * d2_low[i] + result_minus2[0].first.bias;

			results.push_back(pred1 + pred2 - pred3 - pred4);

			result_add1.clear();
			result_add2.clear();
			result_minus1.clear();
			result_minus2.clear();
		}
	}

	struct Rect
	{
		Rect() {}
		Rect(double a_minX, double a_minY, double a_maxX, double a_maxY)
		{
			min[0] = a_minX;
			min[1] = a_minY;
			max[0] = a_maxX;
			max[1] = a_maxY;
		}
		double min[2];
		double max[2];
	};

	// using simple aggregate R tree
	// with refinement
	// filepath, the dataset of the 2D data
	// using full model (i.e., with xy term)
	void CountPrediction(vector<double> &d1_low, vector<double> &d2_low, vector<double> &d1_up, vector<double> &d2_up, vector<double> &results, string filepath= "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv") {

		// prepare the aggregate Rtree
		mat dataset;
		bool loaded = mlpack::data::Load(filepath, dataset);
		arma::rowvec x = dataset.row(0); // x
		arma::rowvec y = dataset.row(1); // y
		vector<double> x_v, y_v;
		RowvecToVector(x, x_v);
		RowvecToVector(y, y_v);

		// construct Rtree
		RTree<int, double, 2, float> tree; // try to update this !!!!!!
		//RTree<int, int, 2, int> tree;
		for (int i = 0; i < x_v.size(); i++) {
			Rect rect(x_v[i], y_v[i], x_v[i], y_v[i]);
			tree.Insert(rect.min, rect.max, i);
		}
		cout << "finsih inserting data to Simple RTree" << endl;

		double cardinality = 0; // number of matches in set
		tree.GenerateCountAggregate(tree.m_root); // generate Aggregate value first

		// the main part
		results.clear();

		vector<pair<ModelBox, int>> result_add1;
		vector<pair<ModelBox, int>> result_add2;
		vector<pair<ModelBox, int>> result_minus1;
		vector<pair<ModelBox, int>> result_minus2;

		double pred1, pred2, pred3, pred4;
		double result;
		double max_err_rel;
		int count_refinement;
		int leafcount = 0;
		
		auto t0 = chrono::steady_clock::now();

		for (int i = 0; i < d1_low.size(); i++) {
			MyPoint lower_left(d1_low[i], d2_low[i]); // add 1 
			MyPoint upper_right(d1_up[i], d2_up[i]); // add 2
			MyPoint upper_left(d1_low[i], d2_up[i]); // minus 1 
			MyPoint lower_right(d1_up[i], d2_low[i]); // minus 2

			model_rtree.query(bgi::intersects(lower_left), std::back_inserter(result_add1));
			model_rtree.query(bgi::intersects(upper_right), std::back_inserter(result_add2));
			model_rtree.query(bgi::intersects(upper_left), std::back_inserter(result_minus1));
			model_rtree.query(bgi::intersects(lower_right), std::back_inserter(result_minus2));

			pred1 = result_add1[0].first.a2 * d1_low[i] * d1_low[i] + result_add1[0].first.a1 * d1_low[i] + result_add1[0].first.b2 * d2_low[i] * d2_low[i] + result_add1[0].first.b1 * d2_low[i] + result_add1[0].first.c * d1_low[i] * d2_low[i] + result_add1[0].first.bias;

			pred2 = result_add2[0].first.a2 * d1_up[i] * d1_up[i] + result_add2[0].first.a1 * d1_up[i] + result_add2[0].first.b2 * d2_up[i] * d2_up[i] + result_add2[0].first.b1 * d2_up[i] + result_add2[0].first.c * d1_up[i] * d2_up[i] + result_add2[0].first.bias;

			pred3 = result_minus1[0].first.a2 * d1_low[i] * d1_low[i] + result_minus1[0].first.a1 * d1_low[i] + result_minus1[0].first.b2 * d2_up[i] * d2_up[i] + result_minus1[0].first.b1 * d2_up[i] + result_minus1[0].first.c * d1_low[i] * d2_up[i] + result_minus1[0].first.bias;

			pred4 = result_minus2[0].first.a2 * d1_up[i] * d1_up[i] + result_minus2[0].first.a1 * d1_up[i] + result_minus2[0].first.b2 * d2_low[i] * d2_low[i] + result_minus2[0].first.b1 * d2_low[i] + result_minus2[0].first.c * d1_up[i] * d2_low[i] + result_minus2[0].first.bias;

			result = pred1 + pred2 - pred3 - pred4;

			// check if satisfy relative error threshold
			max_err_rel = 4 * error_threshold / (result - 4 * error_threshold);
			//cout << "estimate relative error: " << max_err_rel << endl;
			if (max_err_rel > relative_error_threshold || max_err_rel < 0) {
				count_refinement++;
				// need to do refinement, directly use an aggregate Rtree
				Rect query_region(d1_low[i], d2_low[i], d1_up[i], d2_up[i]);
				cardinality = tree.Aggregate(query_region.min, query_region.max, leafcount);
				results.push_back(cardinality);
			}
			else {
				results.push_back(result);
			}

			result_add1.clear();
			result_add2.clear();
			result_minus1.clear();
			result_minus2.clear();
		}

		auto t1 = chrono::steady_clock::now();

		// get COUNT average
		double sum = 0;
		for (int i = 0; i < results.size(); i++) {
			sum += results[i];
		}
		sum /= 1000;

		cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
		cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (d1_low.size()) << " ns" << endl;
		cout << "refinement count: " << count_refinement << endl;
		cout << "hit probability: " << 1000 - count_refinement << " / 1000" << endl;
		cout << "average count: " << sum << endl;
		cout << "total models: " << temp_models.size() << endl;
	}


	QueryResult CountPrediction2(vector<double> &d1_low, vector<double> &d2_low, vector<double> &d1_up, vector<double> &d2_up, vector<double> &results) {

		cout << "start querying.." << endl;

		// the main part
		results.clear();

		vector<pair<ModelBox, int>> result_add1;
		vector<pair<ModelBox, int>> result_add2;
		vector<pair<ModelBox, int>> result_minus1;
		vector<pair<ModelBox, int>> result_minus2;

		double pred1, pred2, pred3, pred4;
		double result;
		double max_err_rel;
		int count_refinement;
		double cardinality   = 0;
		int leafcount = 0;

		auto t0 = chrono::steady_clock::now();

		for (int i = 0; i < d1_low.size(); i++) {

			/*if (i == 3) {
				cout << "here" << endl;
			}*/

			MyPoint lower_left(d1_low[i], d2_low[i]); // add 1 
			MyPoint upper_right(d1_up[i], d2_up[i]); // add 2
			MyPoint upper_left(d1_low[i], d2_up[i]); // minus 1 
			MyPoint lower_right(d1_up[i], d2_low[i]); // minus 2

			//cout << "query set: " << d1_low[i] << " " << d2_low[i] << " " << d1_up[i] << " " << d2_up[i] << endl;

			model_rtree.query(bgi::intersects(lower_left), std::back_inserter(result_add1));
			model_rtree.query(bgi::intersects(upper_right), std::back_inserter(result_add2));
			model_rtree.query(bgi::intersects(upper_left), std::back_inserter(result_minus1));
			model_rtree.query(bgi::intersects(lower_right), std::back_inserter(result_minus2));

			pred1 = result_add1[0].first.a2 * d1_low[i] * d1_low[i] + result_add1[0].first.a1 * d1_low[i] + result_add1[0].first.b2 * d2_low[i] * d2_low[i] + result_add1[0].first.b1 * d2_low[i] + result_add1[0].first.c * d1_low[i] * d2_low[i] + result_add1[0].first.bias;

			pred2 = result_add2[0].first.a2 * d1_up[i] * d1_up[i] + result_add2[0].first.a1 * d1_up[i] + result_add2[0].first.b2 * d2_up[i] * d2_up[i] + result_add2[0].first.b1 * d2_up[i] + result_add2[0].first.c * d1_up[i] * d2_up[i] + result_add2[0].first.bias;

			pred3 = result_minus1[0].first.a2 * d1_low[i] * d1_low[i] + result_minus1[0].first.a1 * d1_low[i] + result_minus1[0].first.b2 * d2_up[i] * d2_up[i] + result_minus1[0].first.b1 * d2_up[i] + result_minus1[0].first.c * d1_low[i] * d2_up[i] + result_minus1[0].first.bias;

			pred4 = result_minus2[0].first.a2 * d1_up[i] * d1_up[i] + result_minus2[0].first.a1 * d1_up[i] + result_minus2[0].first.b2 * d2_low[i] * d2_low[i] + result_minus2[0].first.b1 * d2_low[i] + result_minus2[0].first.c * d1_up[i] * d2_low[i] + result_minus2[0].first.bias;

			result = pred1 + pred2 - pred3 - pred4;

			/*cout << "result_add1.size: " << result_add1.size() << endl;
			cout << "result_add2.size: " << result_add2.size() << endl;
			cout << "result_minus1.size: " << result_minus1.size() << endl;
			cout << "result_minus2.size: " << result_minus2.size() << endl;

			cout << "pred1 lower left: " << pred1 << " " << d1_low[i] << " " << d2_low[i] << endl;
			cout << "pred2 upper right: " << pred2 << " " << d1_up[i] << " " << d2_up[i] << endl;
			cout << "pred3 upper left: " << pred3 << " " << d1_low[i] << " " << d2_up[i] << endl;
			cout << "pred4 lower right: " << pred4 << " " << d1_up[i] << " " << d2_low[i] << endl;

			cout << "result_add1 parameters: " << endl;
			cout << "a2: " << result_add1[0].first.a2 << endl;
			cout << "a1: " << result_add1[0].first.a1 << endl;
			cout << "b2: " << result_add1[0].first.b2 << endl;
			cout << "b1: " << result_add1[0].first.b1 << endl;
			cout << "c: " << result_add1[0].first.c << endl;
			cout << "L1: " << result_add1[0].first.d1_lower << endl;
			cout << "L2: " << result_add1[0].first.d2_lower << endl;
			cout << "U1: " << result_add1[0].first.d1_upper << endl;
			cout << "U2: " << result_add1[0].first.d2_upper << endl;

			cout << "predicted result using our models: " << result << endl;*/

			// check if satisfy relative error threshold
			max_err_rel = 4 * error_threshold / (result - 4 * error_threshold);
			//cout << "estimate relative error: " << max_err_rel << endl;
			if (max_err_rel > relative_error_threshold || max_err_rel < 0) {
				count_refinement++;
				// need to do refinement, directly use an aggregate Rtree
				Rect query_region(d1_low[i], d2_low[i], d1_up[i], d2_up[i]);
				cardinality = aRtree.Aggregate(query_region.min, query_region.max, leafcount);
				//cout << "predicted result using refinement: " << cardinality << endl;
				results.push_back(cardinality);
			}
			else {
				results.push_back(result);
			}

			result_add1.clear();
			result_add2.clear();
			result_minus1.clear();
			result_minus2.clear();

			//cout << "========================" << endl;
		}

		auto t1 = chrono::steady_clock::now();

		// get COUNT average
		double sum = 0;
		for (int i = 0; i < results.size(); i++) {
			sum += results[i];
		}
		sum /= 1000;
		cout << "average count: " << sum << endl;

		auto average_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / d1_low.size();
		auto total_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();

		double MEabs, MErel;
		MeasureAccuracy(results, "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/OSM_2D.csv", MEabs, MErel);

		QueryResult query_result;
		query_result.average_query_time = average_time;
		query_result.total_query_time = total_time;
		query_result.measured_absolute_error = MEabs;
		query_result.measured_relative_error = MErel;
		query_result.hit_count = d1_low.size() - count_refinement;
		query_result.model_amount = temp_models.size();
		query_result.total_paras = this->temp_models.size() * 8; // only for models, without the index

		// export query result to file
		ofstream outfile_result;
		outfile_result.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RunResults/COUNT2D_QueryResult.csv");
		for (int i = 0; i < results.size(); i++) {
			outfile_result << results[i] << endl;
		}

		return query_result;
	}

	// using simple Rtree
	void PrepareExactAggregateRtree(vector<double> &key1, vector<double> &key2) {

		cout << "prepare aggregate Rtree.." << endl;
		// aRtree.clear(); // no such function

		for (int i = 0; i < key1.size(); i++) {
			Rect rect(key1[i], key2[i], key1[i], key2[i]);
			aRtree.Insert(rect.min, rect.max, i);
		}
		cout << "finsih inserting data to Simple RTree" << endl;

		double total_count;
		aRtree.GenerateCountAggregate(aRtree.m_root);

		cout << "finish generating aggregate Rtree.." << endl;
	}

	// using simple Rtree, adjusted it to aggregate max tree
	// with refinement
	// filepath, the dataset of the 2D data
	// using simiplified model (i.e., without xy term)
	void MaxPrediction(vector<double> &d1_low, vector<double> &d2_low, vector<double> &d1_up, vector<double> &d2_up, vector<double> &results, string filepath = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv") {

		// prepare the aggregate Rtree
		mat dataset;
		bool loaded = mlpack::data::Load(filepath, dataset);
		arma::rowvec x = dataset.row(0); // x
		arma::rowvec y = dataset.row(1); // y
		vector<double> x_v, y_v;
		RowvecToVector(x, x_v);
		RowvecToVector(y, y_v);

		// the main part
		results.clear();

		vector<pair<ModelBox, int>> result_add1;
		vector<pair<ModelBox, int>> result_add2;
		vector<pair<ModelBox, int>> result_minus1;
		vector<pair<ModelBox, int>> result_minus2;

		double pred1, pred2, pred3, pred4;
		double result;
		double max_err_rel;
		int count_refinement;
		double max_result;

		auto t0 = chrono::steady_clock::now();

		for (int i = 0; i < d1_low.size(); i++) {
			MyPoint lower_left(d1_low[i], d2_low[i]); // add 1 
			MyPoint upper_right(d1_up[i], d2_up[i]); // add 2
			MyPoint upper_left(d1_low[i], d2_up[i]); // minus 1 
			MyPoint lower_right(d1_up[i], d2_low[i]); // minus 2

			model_rtree.query(bgi::intersects(lower_left), std::back_inserter(result_add1));
			model_rtree.query(bgi::intersects(upper_right), std::back_inserter(result_add2));
			model_rtree.query(bgi::intersects(upper_left), std::back_inserter(result_minus1));
			model_rtree.query(bgi::intersects(lower_right), std::back_inserter(result_minus2));

			// using derivative 

			/*pred1 = result_add1[0].first.a2 * d1_low[i] * d1_low[i] + result_add1[0].first.a1 * d1_low[i] + result_add1[0].first.b2 * d2_low[i] * d2_low[i] + result_add1[0].first.b1 * d2_low[i] + result_add1[0].first.c * d1_low[i] * d2_low[i] + result_add1[0].first.bias;

			pred2 = result_add2[0].first.a2 * d1_up[i] * d1_up[i] + result_add2[0].first.a1 * d1_up[i] + result_add2[0].first.b2 * d2_up[i] * d2_up[i] + result_add2[0].first.b1 * d2_up[i] + result_add2[0].first.c * d1_up[i] * d2_up[i] + result_add2[0].first.bias;

			pred3 = result_minus1[0].first.a2 * d1_low[i] * d1_low[i] + result_minus1[0].first.a1 * d1_low[i] + result_minus1[0].first.b2 * d2_up[i] * d2_up[i] + result_minus1[0].first.b1 * d2_up[i] + result_minus1[0].first.c * d1_low[i] * d2_up[i] + result_minus1[0].first.bias;

			pred4 = result_minus2[0].first.a2 * d1_up[i] * d1_up[i] + result_minus2[0].first.a1 * d1_up[i] + result_minus2[0].first.b2 * d2_low[i] * d2_low[i] + result_minus2[0].first.b1 * d2_low[i] + result_minus2[0].first.c * d1_up[i] * d2_low[i] + result_minus2[0].first.bias;

			result = pred1 + pred2 - pred3 - pred4;*/
			result = 0;

			// check if satisfy relative error threshold
			max_err_rel = 4 * error_threshold / (result - 4 * error_threshold);
			//cout << "estimate relative error: " << max_err_rel << endl;
			if (max_err_rel > relative_error_threshold || max_err_rel < 0) {
				count_refinement++;
				// need to do refinement, directly use an aggregate Rtree
				Rect query_region(d1_low[i], d2_low[i], d1_up[i], d2_up[i]);
				max_result = aRtree.MaxAggregate(query_region.min, query_region.max);
				results.push_back(max_result);
			}
			else {
				results.push_back(result);
			}

			result_add1.clear();
			result_add2.clear();
			result_minus1.clear();
			result_minus2.clear();
		}

		auto t1 = chrono::steady_clock::now();

		// get COUNT average
		double sum = 0;
		for (int i = 0; i < results.size(); i++) {
			sum += results[i];
		}
		sum /= 1000;

		cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
		cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (d1_low.size()) << " ns" << endl;
		cout << "refinement count: " << count_refinement << endl;
		cout << "hit probability: " << 1000 - count_refinement << " / 1000" << endl;
		cout << "average count: " << sum << endl;
	}

	// using the sample surface file to generate
	void GenerateKeysAndAccuFromFile(string filepath = "C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimTrainingSet1000_1000.csv") {
		arma::mat dataset;  
		mlpack::data::Load(filepath, dataset);
		arma::mat trainingset = dataset.rows(0, 1); // x and y
		arma::rowvec responses = dataset.row(2); // count

		keys_v.clear();
		accumulation_v.clear();
		for (int i = 0; i < trainingset.n_cols; i++) {
			keys_v.push_back(pair<double, double>(trainingset(0,i), trainingset(1,i)));
			accumulation_v.push_back(responses[i]);
		}
	}

	// loop the input keys to predicte its value
	void GeneratePredictionSurface(string filepath= "C:/Users/Cloud/Desktop/LearnIndex/data/PredictionSurface_1K_1K.csv") {
		
		prediction_surface.clear();
		vector<pair<ModelBox, int>> result;
		
		double pred;
		for (int i = 0; i < keys_v.size(); i++) {
			MyPoint p(keys_v[i].first, keys_v[i].second);
			model_rtree.query(bgi::intersects(p), std::back_inserter(result));
			pred = result[0].first.a2 * keys_v[i].first * keys_v[i].first + result[0].first.a1 * keys_v[i].first + result[0].first.b2 * keys_v[i].second * keys_v[i].second + result[0].first.b1 * keys_v[i].second + result[0].first.c * keys_v[i].first * keys_v[i].second + result[0].first.bias;

			prediction_surface.push_back(pred);
			result.clear();
		}

		// save the result to a file, transfer it to a rowvec and save to file
		arma::rowvec pred_surface;
		pred_surface.clear();
		pred_surface.set_size(prediction_surface.size());
		int count = 0;
		pred_surface.imbue([&]() { return prediction_surface[count++]; });

		pred_surface.save(filepath, csv_ascii);
	}

	// using boost Rtree
	static void TestRtreePerformance() {
		mat dataset;
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
		arma::rowvec x = dataset.row(0); // x
		arma::rowvec y = dataset.row(1); // y
		vector<double> x_v, y_v;
		RowvecToVector(x, x_v);
		RowvecToVector(y, y_v);

		// construct Rtree
		bgi::rtree<MyPoint, bgi::linear<16>> rtree;
		for (int i = 0; i < x_v.size(); i++) {
			MyPoint point(x_v[i], y_v[i]);
			rtree.insert(point);
		}
		cout << "finsih inserting data to Rtree" << endl;

		size_t cardinality = 0; // number of matches in set
		auto count_only = boost::make_function_output_iterator([&cardinality](bgi::rtree<MyPoint, bgi::linear<16>>::value_type const&) { ++cardinality; });
		Box query_region(MyPoint(-91, -181), MyPoint(91, 181));

		auto t0 = chrono::steady_clock::now();
		for (int i = 0; i < 1000; i++) {
			cardinality = 0;
			rtree.query(bgi::intersects(query_region), count_only);
		}
		auto t1 = chrono::steady_clock::now();
		cout << "Test Aggregate Rtree Time in chrono (average): " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / 1000 << " ns" << endl;
		cout << "count: " << cardinality << endl;
	}

	// using Simple RTree
	static void TestSimpleAggregateRtree(vector<double> &d1_low, vector<double> &d2_low, vector<double> &d1_up, vector<double> &d2_up, vector<double> &results, string datasetpath) {
		mat dataset;
		bool loaded = mlpack::data::Load(datasetpath, dataset);
		arma::rowvec x = dataset.row(0); // x
		arma::rowvec y = dataset.row(1); // y
		vector<double> x_v, y_v;
		RowvecToVector(x, x_v);
		RowvecToVector(y, y_v);

		// construct Rtree

		RTree<int, double, 2, float> tree; // try to update this !!!!!!
		//RTree<int, int, 2, int> tree;
		for (int i = 0; i < x_v.size(); i++) {
			Rect rect(x_v[i], y_v[i], x_v[i], y_v[i]);
			tree.Insert(rect.min, rect.max, i);
		}
		cout << "finsih inserting data to Simple RTree" << endl;

		double cardinality = 0; // number of matches in set
		tree.GenerateCountAggregate(tree.m_root); // generate Aggregate value first

		results.clear();

		int leafcount = 0, totalleafcount = 0;

		auto t0 = chrono::steady_clock::now();
		for (int i = 0; i < d1_low.size(); i++) {
			Rect query_region(d1_low[i], d2_low[i], d1_up[i], d2_up[i]);
			cardinality = tree.Aggregate(query_region.min, query_region.max, leafcount);
			totalleafcount += leafcount;
			results.push_back(cardinality);
		}
		auto t1 = chrono::steady_clock::now();
		cout << "Test Simple Aggregate RTree Time in chrono (average): " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / 1000 << " ns" << endl;
		double total_count = 0;
		for (int i = 0; i < results.size(); i++) {
			/*if (i < 10) {
				cout << results[i] << endl;
			}*/
			total_count += results[i];
		}
		cout << "total count: " << total_count << endl;
		cout << "average count: " << total_count / 1000 << endl;
		cout << "average leaf node visited: " << totalleafcount / results.size() << endl;
		cout << "total elements: " << tree.Count() << endl; 
		cout << "total leaf nodes in the aggregate Rtree: " << tree.CountLeafnode() << endl;
	}

	// using Simple RTree
	static void  TestSimpleApproximateAggregateRtree(vector<double> &d1_low, vector<double> &d2_low, vector<double> &d1_up, vector<double> &d2_up, vector<double> &results, string datasetpath, double Trel=0.01) {
		mat dataset;
		bool loaded = mlpack::data::Load(datasetpath, dataset);
		arma::rowvec x = dataset.row(0); // x
		arma::rowvec y = dataset.row(1); // y
		vector<double> x_v, y_v;
		RowvecToVector(x, x_v);
		RowvecToVector(y, y_v);

		// construct Rtree

		RTree<int, double, 2, float> tree; // try to update this !!!!!!
		//RTree<int, int, 2, int> tree;
		for (int i = 0; i < x_v.size(); i++) {
			Rect rect(x_v[i], y_v[i], x_v[i], y_v[i]);
			tree.Insert(rect.min, rect.max, i);
		}
		cout << "finsih inserting data to Simple RTree" << endl;

		double cardinality = 0; // number of matches in set
		tree.GenerateCountAggregate(tree.m_root); // generate Aggregate value first

		results.clear();
		auto t0 = chrono::steady_clock::now(); 
		for (int i = 0; i < d1_low.size(); i++) {
			Rect query_region(d1_low[i], d2_low[i], d1_up[i], d2_up[i]);
   			cardinality = tree.AggregateApprox(query_region.min, query_region.max, Trel);
			results.push_back(cardinality);
		}
		auto t1 = chrono::steady_clock::now();
		cout << "Test Simple Aggregate RTree Time in chrono (average): " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / 1000 << " ns" << endl;

		double total_count = 0;
		for (int i = 0; i < results.size(); i++) {
			if (i < 10) {
				cout << results[i] << endl;
			}
			total_count += results[i];
		}
		cout << "total count: " << total_count << endl;
		cout << "average count: " << total_count / 1000 << endl;
	}

	double error_threshold;
	double relative_error_threshold;
	double domain_min_d1, domain_max_d1, domain_min_d2, domain_max_d2; // the domain of the dataset
	vector<pair<double, double>> keys_v;
	vector<double> accumulation_v;

	vector<double> prediction_surface; // for model visualization only, predicted the training keys

	vector<ModelBox> temp_models; // the max loss lp models
	bgi::rtree<std::pair<ModelBox, int>, bgi::quadratic<16>> model_rtree; // used to index the 2D max loss lp models

	RTree<int, double, 2, float> aRtree; // the exact aggreate Rtree
};