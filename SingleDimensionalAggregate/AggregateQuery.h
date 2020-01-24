#pragma once
#include "LearnIndexAggregate.h"
#include "STXBtreeAggregate.h"
#include "StageModel.h"
#include "Hilbert.h"
#include "StageModel2D.h"
#include "StageModel2D_2.h"
#include "StageModelBottomUp.h"
#include "Atree.h"
#include "Douglas.h"
#include "EntropyHist.h"
#include "Histogram.h"
#include <stack>
#include "ROLLearnedIndex_quadratic.h"
#include "ROLLearnedIndex_cubic.h"
#include "ROLLearnedIndex_quartic.h"
#include "ReverseMaxlossOptimal.h"
#include <ilcplex/ilocplex.h>
#include "Maxloss2D_QuadDivide.h"
#include "RTree.h"
#include <cmath>

using namespace std;

void Approximate1DMax(){
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/SampledFinancial.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	vector<double> keys, values;
	RowvecToVector(trainingset, keys);
	RowvecToVector(responses, values);

	ReverseMaxlossOptimal RMLO(100, 0.01, 3);

	vector<double> paras;
	double loss;

	auto t0 = chrono::steady_clock::now();
	RMLO.SolveMaxlossLP(x_v, y_v, 0, y_v.size(), paras, loss);
	auto t1 = chrono::steady_clock::now();
	cout << "total query time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (1000 * 1000 * 1000) << "in s" << endl;

	cout << "loss: " << loss << endl;
	// start from a0 TO a3
	for (int i = 0; i < paras.size(); i++) {
		cout << paras[i] << endl;
	}
	system("pause");
}

// https://www.johndcook.com/blog/cpp_phi_inverse/
// https://www.johndcook.com/blog/normal_cdf_inverse/
double RationalApproximation(double t) {
	double c0 = 2.515517;
	double c1 = 0.802853;
	double c2 = 0.010328;
	double d1 = 1.432788;
	double d2 = 0.189269;
	double d3 = 0.001308;

	return t - ((c2 * t + c1)*t + c0) / (((d3 * t + d2)*t + d1)*t + 1.0);
}

// for normal distribution
double InverseCDF(double p) {
	if (p < 0.5)
	{
		// F^-1(p) = - G^-1(p)
		return -RationalApproximation(sqrt(-2.0*log(p)));
	}
	else
	{
		// F^-1(p) = G^-1(1-p)
		return RationalApproximation(sqrt(-2.0*log(1 - p)));
	}
}

// If using this one, two sample is 0 will terminate this algorithm
double SequentialSampling(vector<double> keys, double lower, double upper, double p = 0.9, double Trel=0.01, double Tabs=100) {

	double Vn = 0; // used to substitue sigma^2
	double Xn_sum = 0;
	double Xn_average = 0; // used to substitue Mu
	double sampled_value = 0;
	double n = 1;
	double square_sum = 0;
	double Zp;
	srand((unsigned)time(NULL));

	double d = Tabs / keys.size(); // the absolute error of each partition

	// calculate Zp
	Zp = InverseCDF((1 + p) / 2);

	// perform the first sample
	double index = (rand() % (keys.size() - 1));
	if (keys[index] >= lower && keys[index] <= upper) {
		sampled_value = 1;
	}
	else {
		sampled_value = 0;
	}
	Xn_sum += sampled_value;
	n = 1;

	double pre = sampled_value;
	int max_index = 0;

	// used for large random number generation
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<unsigned long long> dis(0, keys.size());

	// start from n>=2
	while (true) {

		// perform sampling
		// random take 1 key
		//double index = (rand() % (keys.size()-1)); // by default, rand generate values from 0 to rand_max, rand max is equal or larger than 32767, in my pc is 32767
		int index = dis(gen) % (keys.size() - 1);
		/*if (index > max_index)
			max_index = index;*/

		if (keys[index] >= lower && keys[index] <= upper) {
			sampled_value = 1;
		}
		else {
			sampled_value = 0;
		}

		n++;
		Xn_sum += sampled_value;
		Xn_average = Xn_sum/n;

		// calculate Vn
		if (n == 2) {
			square_sum += (pre - Xn_average)*(pre - Xn_average);
		}
		square_sum += (sampled_value - Xn_average)*(sampled_value - Xn_average);
		Vn = square_sum / (n - 1);

		// check stop condition:

		double left_part = Xn_sum;
		if (n*d >= Xn_sum) {
			left_part = n * d;
		}

		double diff = Trel * left_part - Zp * sqrt(n*Vn);
		//cout << "diff: " << diff << endl; 
		if(diff >= 0 && Vn > 0){
			break;
		}
		/*if (int(n) % 10000 == 0) {
			cout <<"current n: " << n  << "  diff: " << diff <<  "  Vn: " << Vn  << " index: " << index  << "  sum: " << Xn_sum << " max index: " << max_index << endl;
		}*/
	}
	//cout << "number of samplling: " << n << endl;
	double estimated_result = keys.size() * Xn_sum / n;
	return estimated_result;
}

// the 2D version
// keys1 has the same length as keys2, which is the same record's two keys
double SequentialSampling2D(vector<double> &keys1, vector<double> &keys2, double lower1, double lower2, double upper1, double upper2, double p = 0.9, double Trel = 0.01, double Tabs = 100) {

	double Vn = 0; // used to substitue sigma^2
	double Xn_sum = 0;
	double Xn_average = 0; // used to substitue Mu
	double sampled_value = 0;
	double n = 1;
	double square_sum = 0;
	double Zp;
	srand((unsigned)time(NULL));

	double d = Tabs / keys1.size(); // the absolute error of each partition

	// calculate Zp
	Zp = InverseCDF((1 + p) / 2);

	// perform the first sample
	double index = (rand() % (keys1.size() - 1));
	if (keys1[index] >= lower1 && keys1[index] <= upper1 && keys2[index] >= lower2 && keys2[index] <= upper2) {
		sampled_value = 1;
	}
	else {
		sampled_value = 0;
	}
	Xn_sum += sampled_value;
	n = 1;

	double pre = sampled_value;
	int max_index = 0;

	// used for large random number generation
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_int_distribution<unsigned long long> dis(0, keys1.size());

	// start from n>=2
	while (true) {

		// perform sampling
		// random take 1 key
		//double index = (rand() % (keys.size()-1)); // by default, rand generate values from 0 to rand_max, rand max is equal or larger than 32767, in my pc is 32767
		int index = dis(gen) % (keys1.size() - 1);
		/*if (index > max_index)
			max_index = index;*/

		if (keys1[index] >= lower1 && keys1[index] <= upper1 && keys2[index] >= lower2 && keys2[index] <= upper2) {
			sampled_value = 1;
		}
		else {
			sampled_value = 0;
		}

		n++;
		Xn_sum += sampled_value;
		Xn_average = Xn_sum / n;

		// calculate Vn
		if (n == 2) {
			square_sum += (pre - Xn_average)*(pre - Xn_average);
		}
		square_sum += (sampled_value - Xn_average)*(sampled_value - Xn_average);
		Vn = square_sum / (n - 1);

		// check stop condition:

		double left_part = Xn_sum;
		if (n*d >= Xn_sum) {
			left_part = n * d;
		}

		double diff = Trel * left_part - Zp * sqrt(n*Vn);
		//cout << "diff: " << diff << endl; 
		if (diff >= 0 && Vn > 0) {
			break;
		}
		/*if (int(n) % 10000 == 0) {
			cout <<"current n: " << n  << "  diff: " << diff <<  "  Vn: " << Vn  << " index: " << index  << "  sum: " << Xn_sum << " max index: " << max_index << endl;
		}*/
	}
	//cout << "number of samplling: " << n << endl;
	double estimated_result = keys1.size() * Xn_sum / n;
	return estimated_result;
}


// The same, abandoned
double S2Sampling(vector<double> keys, double lower, double upper, double p = 0.9, double Trel = 0.01) {
	int n = 1;
	double s = 0;
	double sample_value = 0;
	double w = 0;

	// perform the first sample
	double index = (rand() % (keys.size() - 1));
	if (keys[index] >= lower && keys[index] <= upper) {
		sample_value = 1;
	}
	else {
		sample_value = 0;
	}
	s = sample_value;

	double Zp = InverseCDF((1 + p) / 2);

	while (!(w >= 0) && !(Trel*s >= Zp * sqrt(n*w / (n - 1)))) {
		double index = (rand() % (keys.size() - 1));
		if (keys[index] >= lower && keys[index] <= upper) {
			sample_value = 1;
		}
		else {
			sample_value = 0;
		}

		w += (s - n * sample_value)*(s - n * sample_value) / (n*(n - 1));
		s += sample_value;
		n += 1;
	}
	cout << "number of samplling: " << n << endl;
	return keys.size() * s / n;
}

void ErrorGuaranteedSampling() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	vector<double> keys;
	RowvecToVector(trainingset, keys);

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results, real_results;
	vector<double> key_v;
	RowvecToVector(trainingset, key_v);

	double result;
	auto t00 = chrono::steady_clock::now();
	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		result = SequentialSampling(keys, queryset_x_low_v[i], queryset_x_up_v[i]);
		//result = S2Sampling(keys, queryset_x_low_v[i], queryset_x_up_v[i]);
		predicted_results.push_back(int(result));
	}
	auto t11 = chrono::steady_clock::now();
	cout << "total query time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / (1000 * 1000 * 1000) << "in s" << endl;
	cout << "average query time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / queryset_x_up_v.size() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / queryset_x_up_v.size() / (1000 * 1000 * 1000) << "in s" << endl;

	// check correctness
	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);

	system("pause");
}

void Test2DLeafVisited() {
	arma::mat queryset; // d1_low, d1_up, d2_low, d2_up
	mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/Queries2D_sigma_36_18.csv", queryset);
	vector<double> d1_low, d1_up, d2_low, d2_up, results;
	RowvecToVector(queryset.row(0), d1_low);
	RowvecToVector(queryset.row(1), d1_up);
	RowvecToVector(queryset.row(2), d2_low);
	RowvecToVector(queryset.row(3), d2_up);
	Maxloss2D_QuadDivide::TestSimpleAggregateRtree(d1_low, d2_low, d1_up, d2_up, results, "C:/Users/Cloud/Desktop/LearnedAggregateData/Sorted2DimTrainingSet_Long_Lat_100M.csv");
	system("pause");
}

void FinancialDataset1Model() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/ProcessedFinancialData.csv", dataset); // the financial dataset
	arma::rowvec key = dataset.row(0);
	//arma::rowvec longitude = dataset.row(1);
	arma::rowvec price = dataset.row(2); // use position as target dimension

	vector<double> x_v, y_v;
	RowvecToVector(key, x_v);
	RowvecToVector(price, y_v);

	ReverseMaxlossOptimal RMLO(100, 0.01, 2 );

	vector<double> paras;
	double loss;

	auto t00 = chrono::steady_clock::now();
	RMLO.SolveMaxlossLP(x_v, y_v, 0, y_v.size(), paras, loss);
	auto t11 = chrono::steady_clock::now();
	cout << "total query time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / (1000 * 1000 * 1000) << "in s" << endl;

	cout << "loss: " << loss << endl;
	// start from a0
	for (int i = 0; i < paras.size(); i++) {
		cout << paras[i] << endl;
	}
	system("pause");
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

// for verification
void Test2DSingleQueryMax() {

	mat dataset;
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimTrainingSet1000_1000.csv", dataset);
	
	arma::rowvec x = dataset.row(0); // x
	arma::rowvec y = dataset.row(1); // y
	arma::rowvec z = dataset.row(2); // y
	vector<double> x_v, y_v, z_v;
	RowvecToVector(x, x_v);
	RowvecToVector(y, y_v);
	RowvecToVector(z, z_v);

	// construct Rtree
	RTree<int, double, 2, float> tree; // try to update this !!!!!!
	for (int i = 0; i < x_v.size(); i++) {
		Rect rect(x_v[i], y_v[i], x_v[i], y_v[i]);
		tree.Insert(rect.min, rect.max, z_v[i]);
	}
	cout << "finsih inserting data to Simple RTree" << endl;

	double max_result = 0; // number of matches in set
	tree.GenerateMaxAggregate(tree.m_root); // generate Aggregate value first

	Rect query_region(-90, -180, 90, 180);
	max_result = tree.MaxAggregate(query_region.min, query_region.max);

	cout << "max result: " << max_result << endl;
}

void Test2DMinMax() {
	
	Maxloss2D_QuadDivide model2d(1000, 0.01, -180.0, 180.0, -90.0, 90.0);
	model2d.GenerateKeysAndAccuFromFile("C:/Users/Cloud/Desktop/LearnedAggregateData/Sampled2D_100M_1000_1000.csv");
	model2d.TrainModel();
	cout << "Bottom model size: " << model2d.model_rtree.size() << endl;
	cout << "Bottom model size: " << model2d.temp_models.size() << endl;

	// try to save models to file
	model2d.WriteTrainedModelsToFile("C:/Users/Cloud/Desktop/LearnedAggregateData/2D_LP_models_SIMPLIFIED_100M_1000_1000.csv"); // without xy term
	// try to read models from file
	model2d.ReadTrainedModelsFromFile("C:/Users/Cloud/Desktop/LearnedAggregateData/2D_LP_models_SIMPLIFIED_100M_1000_1000.csv"); // without xy term
	model2d.LoadRtree();

	// ============= generate simplified models ================


	//Maxloss2D_QuadDivide model2d(1000, 0.001, -180.0, 180.0, -90.0, 90.0);
	////model2d.GenerateKeysAndAccuFromFile("C:/Users/Cloud/Desktop/LearnedAggregateData/Sampled2D_100M_1000_1000.csv");
	////model2d.TrainModel();
	////cout << "Bottom model size: " << model2d.model_rtree.size() << endl;
	////cout << "Bottom model size: " << model2d.temp_models.size() << endl;

	//// try to save models to file
	////model2d.WriteTrainedModelsToFile("C:/Users/Cloud/Desktop/LearnedAggregateData/2D_LP_models_100M_1000_1000.csv");
	//// try to read models from file
	//model2d.ReadTrainedModelsFromFile("C:/Users/Cloud/Desktop/LearnedAggregateData/2D_LP_models_100M_1000_1000.csv");
	//model2d.LoadRtree();

	//arma::mat queryset;
	//mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/Queries2D_sigma_36_18.csv", queryset); // d1_low, d2_low, d1_up, d2_up

	//vector<double> d1_low, d1_up, d2_low, d2_up, results;
	//RowvecToVector(queryset.row(0), d1_low);
	//RowvecToVector(queryset.row(2), d1_up);
	//RowvecToVector(queryset.row(1), d2_low);
	//RowvecToVector(queryset.row(3), d2_up);

	//// change it to min max prediction
	////model2d.CountPrediction(d1_low, d2_low, d1_up, d2_up, results); // the min max queries
	//model2d.MaxPrediction(d1_low, d2_low, d1_up, d2_up, results); // the min max queries
}

void GenHist() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	vector<double> keys;
	RowvecToVector(trainingset, keys);
	EntropyHist(10000, keys);
}

void HistSegmentationWithPolyfit(int bins=500) {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	//vector<double> keys;
	//RowvecToVector(trainingset, keys);
	//EntropyHist(500, keys);

	arma::mat splits_mat;
	splits_mat.load("C:/Users/Cloud/Desktop/LearnIndex/data/EntropyHistSplits500.bin");
	vector<int> splits;
	for (int i = 0; i < splits_mat.n_rows; i++) {
		splits.push_back(splits_mat[i]);
	}

	ReverseMaxlossOptimal RMLO(100, 0.01, 1);
	RMLO.SegmentWithHistogram(trainingset, responses, splits); // using histogram for split
	RMLO.BuildNonLeafLayerWithBtree(); // btree

	mat queryset; 
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results, real_results;
	vector<double> key_v;
	RowvecToVector(trainingset, key_v);

	 RMLO.CountPredictionHist(queryset_x_low_v, queryset_x_up_v, predicted_results, key_v);
	//RMLO.CountPredictionHistOptimized(queryset_x_low_v, queryset_x_up_v, predicted_results, key_v);
}

void CompareMinMaxFinancial() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/ProcessedFinancialData.csv", dataset); // the financial dataset
	arma::rowvec key = dataset.row(0);
	//arma::rowvec longitude = dataset.row(1);
	arma::rowvec price = dataset.row(2); // use position as target dimension

	vector<double> key_attribute, target_attribute;
	RowvecToVector(key, key_attribute);
	RowvecToVector(price, target_attribute);

	stx::btree<double, double> aggregate_max_tree;
	aggregate_max_tree.clear();
	for (int i = 0; i < key_attribute.size(); i++) {
		aggregate_max_tree.insert(pair<double, double>(key_attribute[i], target_attribute[i]));
	}
	// generate max values for the aggregate max tree
	aggregate_max_tree.generate_max_aggregate();

	// query set
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/ProcessedFinancialQuery.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);

	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	vector<double> predicted_results, real_results;
	double max_value;

	auto t00 = chrono::steady_clock::now();
	for (int i = 0; i < queryset_x_up_v.size(); i++) {
		max_value = aggregate_max_tree.max_query(queryset_x_low_v[i], queryset_x_up_v[i]);
		real_results.push_back(max_value);
	}
	auto t11 = chrono::steady_clock::now();
	cout << "total query time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / (1000 * 1000 * 1000) << "in s" << endl;
	cout << "average query time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / queryset_x_up_v.size() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / queryset_x_up_v.size() / (1000 * 1000 * 1000) << "in s" << endl;


	ReverseMaxlossOptimal RMLO(100, 0.01, 1);
	RMLO.SegmentOnTrainMaxLossModel(key, price, 0);
	RMLO.BuildNonLeafLayerWithBtree(); // btree
	RMLO.PrepareMaxAggregateTree();
	/*RMLO.MaxPredictionWithoutRefinement(queryset_x_low_v, queryset_x_up_v, predicted_results, key_attribute);
	cout << "model amount: " << RMLO.bottom_layer_index.size() << " " << RMLO.dataset_range.size() << endl;*/
	system("pause");
}

void CompareMinMax() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec latitude = dataset.row(0);
	//arma::rowvec longitude = dataset.row(1);
	arma::rowvec longitude = dataset.row(2); // use position as target dimension

	vector<double> key_attribute, target_attribute;
	RowvecToVector(latitude, key_attribute);
	RowvecToVector(longitude, target_attribute);

	stx::btree<double, double> aggregate_max_tree;
	aggregate_max_tree.clear();
	for (int i = 0; i < key_attribute.size(); i++) {
		aggregate_max_tree.insert(pair<double, double>(key_attribute[i], target_attribute[i]));
	}
	// generate max values for the aggregate max tree
	aggregate_max_tree.generate_max_aggregate();

	// query set
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);

	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	vector<double> predicted_results, real_results;
	double max_value;

	auto t00 = chrono::steady_clock::now();
	for (int i = 0; i < queryset_x_up_v.size(); i++) {
		max_value = aggregate_max_tree.max_query(queryset_x_low_v[i], queryset_x_up_v[i]);
		real_results.push_back(max_value);
	}
	auto t11 = chrono::steady_clock::now();
	cout << "total query time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / (1000 * 1000 * 1000) << "in s" << endl;
	cout << "average query time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / queryset_x_up_v.size() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / queryset_x_up_v.size() / (1000 * 1000 * 1000) << "in s" << endl;
	

	ReverseMaxlossOptimal RMLO(1000, 0.01, 1);
	RMLO.SegmentOnTrainMaxLossModel(latitude, longitude, 0);
	RMLO.BuildNonLeafLayerWithBtree(); // btree
	RMLO.PrepareMaxAggregateTree();
	/*RMLO.MaxPredictionWithoutRefinement(queryset_x_low_v, queryset_x_up_v, predicted_results, key_attribute);
	cout << "model amount: " << RMLO.bottom_layer_index.size() << " " << RMLO.dataset_range.size() << endl;*/
	system("pause");
}

void TestMaxAggregate() {
	ReverseMaxlossOptimal RMLO(100, 0.01, 1);
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset); // the 1M tweet dataset

	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	RMLO.SegmentOnTrainMaxLossModel(trainingset, responses, 0);
	RMLO.BuildNonLeafLayerWithBtree(); // btree

	cout << RMLO.bottom_layer_index.size() << endl;
	double aggregate_max = RMLO.bottom_layer_index.generate_max_aggregate();
	cout << RMLO.bottom_layer_index.size() << endl;


	// perform max query 

	vector<double> max_values;
	double max_value;
	auto t00 = chrono::steady_clock::now();
	max_value = RMLO.bottom_layer_index.max_query(-50, 30);
	max_values.push_back(max_value);
	max_value = RMLO.bottom_layer_index.max_query(-40, 30);
	max_values.push_back(max_value);
	max_value = RMLO.bottom_layer_index.max_query(-30, 30);
	max_values.push_back(max_value);
	max_value = RMLO.bottom_layer_index.max_query(-20, 30);
	max_values.push_back(max_value);
	max_value = RMLO.bottom_layer_index.max_query(-10, 30);
	max_values.push_back(max_value);

	max_value = RMLO.bottom_layer_index.max_query(-40, 30);
	max_values.push_back(max_value);
	max_value = RMLO.bottom_layer_index.max_query(-40, 20);
	max_values.push_back(max_value);
	max_value = RMLO.bottom_layer_index.max_query(-40, 10);
	max_values.push_back(max_value);
	max_value = RMLO.bottom_layer_index.max_query(-40, 0);
	max_values.push_back(max_value);
	max_value = RMLO.bottom_layer_index.max_query(-40, -10);
	max_values.push_back(max_value);

	auto t11 = chrono::steady_clock::now();
	cout << "total query time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / (1000 * 1000 * 1000) << "in s" << endl;

	cout << "total max: " << aggregate_max << endl;

	for (int i = 0; i < 10; i++) {
		cout << max_values[i] << endl;
	}

	system("pause");
}

void TestApproximateAggregateRTree(){
	
	arma::mat queryset; // d1_low, d1_up, d2_low, d2_up
	mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/Queries2D_sigma_36_18.csv", queryset);
	vector<double> d1_low, d1_up, d2_low, d2_up, results;
	RowvecToVector(queryset.row(0), d1_low);
	RowvecToVector(queryset.row(1), d1_up);
	RowvecToVector(queryset.row(2), d2_low);
	RowvecToVector(queryset.row(3), d2_up);
	Maxloss2D_QuadDivide::TestSimpleApproximateAggregateRtree(d1_low, d2_low, d1_up, d2_up, results, "C:/Users/Cloud/Desktop/LearnedAggregateData/Sorted2DimTrainingSet_Long_Lat_100M.csv");
	Maxloss2D_QuadDivide::TestSimpleAggregateRtree(d1_low, d2_low, d1_up, d2_up, results, "C:/Users/Cloud/Desktop/LearnedAggregateData/Sorted2DimTrainingSet_Long_Lat_100M.csv");

	system("pause");
}

void TestRMLO(int highest_term = 1, double Tabs = 100, double Trel = 0.01, string recordfilepath = "C:/Users/Cloud/Desktop/LearnedAggregateData/experiment_result.csv");
void TestAtree(double Tabs = 100, double Trel = 0.01, string recordfilepath = "C:/Users/Cloud/Desktop/LearnedAggregateData/experiment_result.csv");

// record results into .csv file
void experimentVerifyTrel() {
	vector<double> Trels = { 0.001, 0.005, 0.01, 0.05, 0.1, 0.2 };
	string filepath = "C:/Users/Cloud/Desktop/LearnedAggregateData/experiment_result.csv";
	//for (int i = 0; i < Trels.size(); i++) {
	//	/*TestRMLO(1, 100, Trels[i], filepath);
	//	TestRMLO(2, 100, Trels[i], filepath);
	//	TestRMLO(3, 100, Trels[i], filepath);*/
	//	TestAtree(100, Trels[i], filepath);
	//	ofstream outfile_exp;
	//	outfile_exp.open(filepath, std::ios_base::app);
	//	outfile_exp << endl;
	//	outfile_exp.close();
	//}

	/*TestAtree(100, 0.05, filepath);
	TestAtree(100, 0.05, filepath);
	TestAtree(100, 0.05, filepath);*/

	TestRMLO(1, 100, 0.01, filepath);
	TestRMLO(1, 100, 0.01, filepath);
	TestRMLO(1, 100, 0.01, filepath);

}

void TestRMLOApproximation() {
	ReverseMaxlossOptimal RMLO(100, 0.01, 1);
	mat dataset;

	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset); // the 1M tweet dataset

	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	auto t00 = chrono::steady_clock::now();
	RMLO.SegmentOnTrainMaxLossModel(trainingset, responses, 1);
	auto t11 = chrono::steady_clock::now();
	cout << "Construction time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / (1000 * 1000 * 1000) << "in s" << endl;

	RMLO.BuildNonLeafLayerWithBtree(); // btree

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);

	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);

	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	vector<int> predicted_results, real_results;

	vector<double> key_v;
	RowvecToVector(trainingset, key_v);

	RMLO.CountPrediction(queryset_x_low_v, queryset_x_up_v, predicted_results, key_v);

	double treeparas = RMLO.bottom_layer_index.CountParametersPrimary();
	double treesize = treeparas * 8;
	double modelsize = RMLO.dataset_range.size() * 8 * (3 + RMLO.highest_term);
	double totalsize = treesize + modelsize;

	cout << "bottom model count: " << RMLO.dataset_range.size() << endl;
	cout << "Total nodes in btree index: " << RMLO.bottom_layer_index.CountNodesPrimary() << endl;
	cout << "Total parameters in btree index: " << treeparas << endl;
	cout << "Btree size (Bytes): " << treesize << endl;
	cout << "Model size (Bytes): " << modelsize << endl;
	cout << "Total structure size (Bytes): " << totalsize << "in Bytes    " << totalsize / 1024 << " in KB    " << totalsize / (1024 * 1024) << " in MB" << endl;
}

void TestX1(int lower, int upper);
void TestX2(int lower, int upper);
void TestX3(int lower, int upper);
void TestX4(int lower, int upper);

bool MySearchCallback(int id, void* arg)
{
	printf("Hit data rect %d\n", id);
	return true; // keep going
}

void TestSimpleRTree() {
	struct Rect
	{
		Rect() {}
		Rect(int a_minX, int a_minY, int a_maxX, int a_maxY)
		{
			min[0] = a_minX;
			min[1] = a_minY;
			max[0] = a_maxX;
			max[1] = a_maxY;
		}
		int min[2];
		int max[2];
	};

	struct Rect rects[] =
	{
	  Rect(0, 0, 2, 2), // xmin, ymin, xmax, ymax (for 2 dimensional RTree)
	  Rect(5, 5, 7, 7),
	  Rect(8, 5, 9, 6),
	  Rect(7, 1, 9, 2),
	};

	int nrects = sizeof(rects) / sizeof(rects[0]);

	Rect search_rect(6, 4, 10, 6); // search will find above rects that this one overlaps
	RTree<int, int, 2, float> tree;


	for (int i = 0; i < 4; i++)
	{
		tree.Insert(rects[i].min, rects[i].max, i); // Note, all values including zero are fine in this version
	}

	tree.GenerateCountAggregate(tree.m_root);
	int leafcount = 0;
	double aggregate = tree.Aggregate(search_rect.min, search_rect.max, leafcount);
	//nhits = tree.Search(search_rect.min, search_rect.max, MySearchCallback, NULL);

	//printf("Search resulted in %d hits\n", nhits);
	cout << "aggregate value: " << aggregate << endl;
	
}

void TestMaxloss2D() {

	//Maxloss2D_QuadDivide::TestRtreePerformance();
	//Maxloss2D_QuadDivide::TestSimpleAggregateRtree();

	//Maxloss2D_QuadDivide model2d(1000, 0.01, -90.0, 90.0, -180.0, 180.0);
	//model2d.GenerateKeysAndAccuFromFile();

	Maxloss2D_QuadDivide model2d(1000, 0.01, -180.0, 180.0, -90.0, 90.0);
	//model2d.GenerateKeysAndAccuFromFile("C:/Users/Cloud/Desktop/LearnedAggregateData/Sampled2D_100M_1000_1000.csv");
	//model2d.TrainModel();
	//cout << "Bottom model size: " << model2d.model_rtree.size() << endl;
	//cout << "Bottom model size: " << model2d.temp_models.size() << endl;

	// try to save models to file
	//model2d.WriteTrainedModelsToFile("C:/Users/Cloud/Desktop/LearnedAggregateData/2D_LP_models_100M_1000_1000.csv");
	// try to read models from file
	model2d.ReadTrainedModelsFromFile("C:/Users/Cloud/Desktop/LearnedAggregateData/2D_LP_models_100M_1000_1000.csv");
	model2d.LoadRtree();

	//// go through the models to find how many invalid models;
	//int count_invalid = 0;
	//double max_error = 0;
	//for (int i = 0; i < model2d.temp_models.size(); i++) {
	//	if (model2d.temp_models[i].loss == -1) {
	//		count_invalid++;
	//	}
	//	if (model2d.temp_models[i].loss > max_error) {
	//		max_error = model2d.temp_models[i].loss;
	//	}
	//}
	//cout << "invalid models: " << count_invalid << endl;
	//cout << "real max errors: " << max_error << endl;
	// 
	
	arma::mat queryset;
	//mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection100m.csv", queryset); // d1_low, d1_up, d2_low, d2_up
	mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/Queries2D_sigma_36_18.csv", queryset); // d1_low, d2_low, d1_up, d2_up

	//vector<double> d1_low, d1_up, d2_low, d2_up, results;
	////RowvecToVector(queryset.row(0), d1_low);
	////RowvecToVector(queryset.row(1), d1_up);
	////RowvecToVector(queryset.row(2), d2_low);
	////RowvecToVector(queryset.row(3), d2_up);

	//for (int i = 1; i <= 10; i++) {
	//	cout << "========== queryset: " << i << " =========="<< endl;
	//	switch (i) {
	//	case 1:
	//		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection100m.csv", queryset);
	//		break;
	//	case 2:
	//		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection200m.csv", queryset);
	//		break;
	//	case 3:
	//		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection500m.csv", queryset);
	//		break;
	//	case 4:
	//		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection1km.csv", queryset);
	//		break;
	//	case 5:
	//		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection2km.csv", queryset);
	//		break;
	//	case 6:
	//		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection5km.csv", queryset);
	//		break;
	//	case 7:
	//		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection10km.csv", queryset);
	//		break;
	//	case 8:
	//		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection20km.csv", queryset);
	//		break;
	//	case 9:
	//		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection50km.csv", queryset);
	//		break;
	//	case 10:
	//		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection100km.csv", queryset);
	//		break;
	//	}

	//=========================
		vector<double> d1_low, d1_up, d2_low, d2_up, results;
		RowvecToVector(queryset.row(0), d1_low);
		RowvecToVector(queryset.row(2), d1_up);
		RowvecToVector(queryset.row(1), d2_low);
		RowvecToVector(queryset.row(3), d2_up);

		model2d.CountPrediction(d1_low, d2_low, d1_up, d2_up, results);

		//===========================
	//}

	/*auto t0 = chrono::steady_clock::now();
	model2d.QueryPrediction(d1_low, d2_low, d1_up, d2_up, results);
	auto t1 = chrono::steady_clock::now();
	
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns  " << chrono::duration_cast<chrono::nano seconds>(t1 - t0).count() / 1000 / (queryset.size() / queryset.n_rows) << "  us" << endl;*/

	////model2d.GeneratePredictionSurface(); // generate prediction surface

	//arma::mat queryset; // d1_low, d1_up, d2_low, d2_up
	//mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/Queries2D_sigma_36_18.csv", queryset);
	//vector<double> d1_low, d1_up, d2_low, d2_up, results;
	//RowvecToVector(queryset.row(0), d1_low);
	//RowvecToVector(queryset.row(1), d1_up);
	//RowvecToVector(queryset.row(2), d2_low);
	//RowvecToVector(queryset.row(3), d2_up);
	//Maxloss2D_QuadDivide::TestSimpleAggregateRtree(d1_low, d2_low, d1_up, d2_up, results, "C:/Users/Cloud/Desktop/LearnedAggregateData/Sorted2DimTrainingSet_Long_Lat_100M.csv");

	system("pause");
}

void TestLPHighestTerm() {

	double loss;
	vector<double> paras;
	
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(trainingset, key_v);
	RowvecToVector(responses, position_v);

	ReverseMaxlossOptimal RMLO(100, 0.01, 1);
	//RMLO.SolveMaxlossLP(key_v, position_v, 628762, 632418, paras, loss);
	//RMLO.SolveMaxlossLP(key_v, position_v, 1084084, 1089591, paras, loss);

	/*cout.precision(11);
	cout << "max loss: " << loss << endl;
	for (int i = 0; i < paras.size(); i++) {
		cout << paras[i] << " ";
	}

	cout << endl;
	cout <<"============================="<< endl;*/

	//ROLLearnedIndex_cubic learnedindex(9, 1000, 100);
	//double a, b, c, d, e;
	//RowvecToVector(trainingset, key_v);
	//RowvecToVector(responses, position_v);
	//learnedindex.MyCplexSolverForMaxLossQuadraticOptimized(key_v, position_v, 628762, 632418, a, b, c, d, e);
	//learnedindex.MyCplexSolverForMaxLossQuadraticOptimized(key_v, position_v, 1084084, 1089591, a, b, c, d, e);
	//cout << d << "  " << c << "  " << b << "  " << a << "  " << e << endl;
	//cout << a << "  " << b << "  " << c << "  " << d << "  " << e << endl;

	//cout << "=============================" << endl;
	//cout << "import sav file and run again:" << endl;

	//IloEnv new_env2;
	//IloModel new_model2(new_env2);
	//IloCplex new_cplex2(new_model2);
	////new_cplex2.setParam(IloCplex::RootAlg, IloCplex::Barrier);
	//new_cplex2.setParam(IloCplex::Param::Preprocessing::Presolve, false);
	//new_cplex2.setParam(IloCplex::Param::Advance, 0);

	//new_cplex2.importModel(new_model2, "C:/Users/Cloud/Desktop/range392_2.sav");
	//new_cplex2.solve();
	//cout << "max loss: " << new_cplex2.getObjValue() << endl;
}


// This function is used to test the measured relative error and the query response time (without any refinement) of polyfit
// we generate this curve by variating the absolute error threshold of polyfit
void PolyfitExperimentRelativeErrorAndQueryTime(int highest_term=1, double Tabs=100, double Trel=0.01, string recordfilepath="C:/Users/Cloud/Desktop/LearnedAggregateData/experiment_result.csv") {

	cout << "========================  " << Tabs << "  =====================" << endl;

	ReverseMaxlossOptimal RMLO(Tabs, Trel, highest_term);
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset); // the 1M tweet dataset
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	//cout << "using highest term: " << highest_term << endl;

	auto t00 = chrono::steady_clock::now();
	RMLO.SegmentOnTrainMaxLossModel(trainingset, responses);
	auto t11 = chrono::steady_clock::now();
	//cout << "Construction time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() / (1000 * 1000 * 1000) << "in s" << endl;

	RMLO.BuildNonLeafLayerWithBtree(); // btree

	// for query
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Query_1D.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Query_1D.csv", queryset);

	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;
	vector<double> key_v;
	RowvecToVector(trainingset, key_v);

	RMLO.CountPredictionWithoutRefinement(queryset_x_low_v, queryset_x_up_v, predicted_results, key_v, recordfilepath);

	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results); // need to adjust the error threshold by hand!!!!
	//cout << "===============================================" << endl;
	cout << endl;
	//system("pause");
}

void TestMultipleTimes() {
	PolyfitExperimentRelativeErrorAndQueryTime(1, 50);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 100);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 200);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 300);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 400);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 500);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 600);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 700);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 800);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 900);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 1000);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 2000);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 5000);
	PolyfitExperimentRelativeErrorAndQueryTime(1, 10000);
}

// int highest_term = 1, double Tabs=100, double Trel=0.01, string recordfilepath="C:/Users/Cloud/Desktop/LearnedAggregateData/experiment_result.csv"
void TestRMLO(int highest_term, double Tabs, double Trel, string recordfilepath) {
	
	ReverseMaxlossOptimal RMLO(Tabs, Trel, highest_term);
	mat dataset;

	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_SUM_2.csv", dataset); // this is for sum query

	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset); // the 1M tweet dataset
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_1M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Sorted_Value.csv", dataset);
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv", dataset);

	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	cout << "using highest term: " << highest_term << endl;

	auto t00 = chrono::steady_clock::now();
	RMLO.SegmentOnTrainMaxLossModel(trainingset, responses);
	auto t11 = chrono::steady_clock::now();
	cout << "Construction time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() << " in ns    " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count()/(1000*1000*1000) << "in s" << endl;

	RMLO.BuildNonLeafLayerWithBtree(); // btree

	vector<double> trainingset_v, responses_v;
	RowvecToVector(trainingset, trainingset_v);
	RowvecToVector(responses, responses_v);
	//stage_model_bottom_up.SelfChecking(trainingset_v, responses_v);

	//cout.precision(11);
	/*cout << "dataset_range[i].first" << " " << "dataset_range[i].second" << " " << "parameters[i].first" << " " << "parameters[i].second" << " " << "y2" << endl;
	for (int i = 0; i < stage_model_bottom_up.stage_model_parameter[8].size(); i++) {
		cout << stage_model_bottom_up.dataset_range[8][i].first << " " << stage_model_bottom_up.dataset_range[8][i].second << " " << stage_model_bottom_up.stage_model_parameter[8][i][0] << " " << stage_model_bottom_up.stage_model_parameter[8][i][1] << "  y1:" << stage_model_bottom_up.dataset_range[8][i].first * stage_model_bottom_up.stage_model_parameter[8][i][0] + stage_model_bottom_up.stage_model_parameter[8][i][1] << "  y2: " << stage_model_bottom_up.dataset_range[8][i].second * stage_model_bottom_up.stage_model_parameter[8][i][0] + stage_model_bottom_up.stage_model_parameter[8][i][1] << endl;
	}*/

	mat queryset;
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_1M_Query_1D.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Query_1D.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Query_1D.csv", queryset);
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Query_1D.csv", queryset);

	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;
	vector<double> key_v;
	RowvecToVector(trainingset, key_v);

	RMLO.CountPrediction(queryset_x_low_v, queryset_x_up_v, predicted_results, key_v, recordfilepath);
	RMLO.CountPrediction(queryset_x_low_v, queryset_x_up_v, predicted_results, key_v, recordfilepath); // do it again

	//auto t0 = chrono::steady_clock::now();

	//RMLO.PredictWithStxBtree(queryset_x_up_v, predicted_results_x_up);
	//RMLO.PredictWithStxBtree(queryset_x_low_v, predicted_results_x_low);

	//double t_abs = RMLO.t_abs;
	//double t_rel = RMLO.t_rel;
	//int count_over = 0, count_negative = 0;
	//for (int i = 0; i < predicted_results_x_up.size(); i++) {
	//	predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);

	//	// analysis estimated maximum relative error:
	//	double max_err_rel = (2 * t_abs) / (predicted_results[i] - 2 * t_abs);
	//	if (max_err_rel > t_rel) {
	//		count_over++;
	//	}
	//	else if (max_err_rel < 0) {
	//		count_negative++;
	//	}
	//	//cout << "maximum relative error: " << max_err_rel * 100 << "%" << endl;
	//}

	//auto t1 = chrono::steady_clock::now();

	double treeparas = RMLO.bottom_layer_index.CountParametersPrimary();
	double treesize = treeparas * 8;
	double modelsize = RMLO.dataset_range.size() * 8 * (3 + RMLO.highest_term);
	double totalsize = treesize + modelsize;

	//cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	//cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "bottom model count: " << RMLO.dataset_range.size() << endl;
	cout << "Total nodes in btree index: " << RMLO.bottom_layer_index.CountNodesPrimary() << endl;
	cout << "Total parameters in btree index: " << treeparas << endl;
	cout << "Btree size (Bytes): " << treesize << endl;
	cout << "Model size (Bytes): " << modelsize << endl;
	cout << "Total structure size (Bytes): " << totalsize << "in Bytes    " << totalsize / 1024 << " in KB    " << totalsize / (1024 * 1024) << " in MB" << endl;
	//cout << "over threshold relative error: " << count_over << "   negative relative error: " << count_negative << endl;
	//cout << "hit probability: " << 1000 - count_over - count_negative << " / 1000" << endl;

	//// save results to file
	//ofstream outfile;
	//outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted1DimResults_LEARN.csv");
	//for (int i = 0; i < predicted_results.size(); i++) {
	//	outfile << predicted_results[i] << endl;
	//}
	//outfile.close();

	////CalculateRealCountWithScan1D(queryset, real_results, "C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Sorted_Value.csv");
	//CalculateRealCountWithScan1D(queryset, real_results);
	//MeasureAccuracy(predicted_results, real_results); // need to adjust the error threshold by hand!!!!
	//cout << "===============================================" << endl;

	system("pause");
}

void TestRMLOHighestTerm() {
	for (int i = 0; i <= 10; i++) {
		TestRMLO(i);
	}
	system("pause");
}

int inline MyBinarySearch(vector<double> &data, double key) {
	int TOTAL_SIZE = data.size();
	int low = 0, high = TOTAL_SIZE, middle;
	while (high-low>1) {
		middle = (low + high) / 2;
		if (data[middle] > key) {
			high = middle;
		}
		else if (data[middle] < key) {
			low = middle;
		}
		else {
			break;
		}
	}
	return middle;
}

void TestBinarySearch() {
	mat dataset;
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Sorted_Value.csv", dataset);
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv", dataset);

	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> trainingset_v, responses_v;

	RowvecToVector(trainingset, trainingset_v);
	RowvecToVector(responses, responses_v);

	mat queryset;
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Query_1D.csv", queryset);
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Query_1D.csv", queryset);

	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);

	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	auto t0 = chrono::steady_clock::now();

	// using binary search
	int pos;
	for (int i = 0; i < queryset_x_up_v.size(); i++) {
		pos = MyBinarySearch(trainingset_v, queryset_x_up_v[i]);
		predicted_results_x_up.push_back(pos);
		pos = MyBinarySearch(trainingset_v, queryset_x_low_v[i]);
		predicted_results_x_low.push_back(pos);
	}

	for (int i = 0; i < predicted_results_x_up.size(); i++) {
		predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);
	}

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
}

void MyCplexSolverForMaxLoss(const arma::mat& current_trainingset, const arma::rowvec&current_response, double &a, double &b, double &loss) {
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
	for (int i = 0; i < current_trainingset.size(); i++) {
		model.add(vars[0] * current_trainingset[i] + vars[1] - current_response[i] <= vars[2]);
		model.add(vars[0] * current_trainingset[i] + vars[1] - current_response[i] >= -vars[2]);
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

void TestEqualDepthSegmentation() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec labels = dataset.row(dataset.n_rows - 1);

	int TOTAL_SIZE = trainingset.size();

	//cout << "domain_min: " << domain_min << "  domain_max: " << domain_max << endl;

	double abs_err = 10000;

	// try different number of segments
	int n_segments = 1;
	while (true) {
		cout << "current segment size: " << n_segments << endl;
		bool all_success = true;
		int segment_step = TOTAL_SIZE / n_segments;
		int sub_segment_low = 0;
		int sub_segment_high = sub_segment_low;

		for (int i = 0; i < n_segments; i++) {

			sub_segment_low = sub_segment_high;
			sub_segment_high = sub_segment_low + segment_step;
			if (sub_segment_high >= TOTAL_SIZE) {
				sub_segment_high = TOTAL_SIZE - 1;
			}

			// determint the segment data
			arma::rowvec sub_trainingset = trainingset.cols(sub_segment_low, sub_segment_high);
			arma::rowvec sub_labels = labels.cols(sub_segment_low, sub_segment_high);

			// cal linear programming solver
			double a, b, loss;
			MyCplexSolverForMaxLoss(sub_trainingset, sub_labels, a, b, loss);

			if (loss > abs_err) {
				all_success = false;
				break;
			}
		}

		if (all_success) {
			break;
		}
		else {
			n_segments += 1;
		}
	}
}

void TestEqualWidthSegmentation() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec labels = dataset.row(dataset.n_rows-1);
	int domain_min_index = trainingset.index_min();
	int domain_max_index = trainingset.index_max();
	double domain_min = trainingset[domain_min_index];
	double domain_max = trainingset[domain_max_index];

	//cout << "domain_min: " << domain_min << "  domain_max: " << domain_max << endl;

	double abs_err = 100000;

	// try different number of segments
	int n_segments = 1 ;
	while (true) {
		cout << "current segment size: " << n_segments << endl;
		bool all_success = true;
		double segment_step = (domain_max - domain_min) / n_segments;
		double sub_segment_min = domain_min;
		double sub_segment_max = sub_segment_min;

		for (int i = 0; i < n_segments; i++) {

			sub_segment_min = sub_segment_max;
			sub_segment_max = sub_segment_min + segment_step;

			// determint the segment data
			uvec seg_index = find(trainingset >= sub_segment_min && trainingset < sub_segment_max);
			cout << "segment.size: " << seg_index.size() << endl;
			arma::rowvec sub_trainingset = trainingset.cols(seg_index);
			arma::rowvec sub_labels = labels.cols(seg_index);

			if (sub_trainingset.size() == 0) {
				continue;
			}

			// cal linear programming solver
			double a, b, loss;
			MyCplexSolverForMaxLoss(sub_trainingset, sub_labels, a, b, loss);

			if (loss > abs_err) {
				all_success = false;
				break;
			}
		}

		if (all_success) {
			break;
		}
		else {
			n_segments += 1;
		}
	}
}

void TestCplex() {

	StageModelBottomUp learnedindex(9, 1000, 100); // level, step, error

	mat dataset_mse;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset_mse);
	LinearRegression lr(dataset_mse.row(0), dataset_mse.row(dataset_mse.n_rows-1));
	arma::vec paras = lr.Parameters();
	double a_mse, b_mse;
	a_mse = paras[1]; // a
	b_mse = paras[0]; // b
	cout << "a mse: " << a_mse << "  b mse: " << b_mse << endl;

	arma::mat dataset;
	dataset.load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv");

	arma::mat trainingset = dataset.col(0);
	arma::mat responses = dataset.col(dataset.n_cols - 1);

	cout << dataset.n_cols << " " << dataset.n_rows << endl;
	//cout << trainingset.size() << endl;

	//trainingset = trainingset.cols(0, 1000);
	//responses = responses.cols(0, 1000);

	vector<double> key_v, position_v;

	//trainingset.size()
	for (int i = 0; i < trainingset.size(); i++) {
		key_v.push_back(trainingset[i]);
		position_v.push_back(responses[i]);
	}

	//RowvecToVector(trainingset, key_v);
	//RowvecToVector(responses, position_v);

	double a = 1, b = 1, current_absolute_accuracy = 0;

	learnedindex.MyCplexSolverForMaxLoss(key_v, position_v, a, b, current_absolute_accuracy);
}

void MeasureTimeWithoutPow() {
	mat queryset1, queryset2, queryset3;
	bool loaded1 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset1);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Query_1D.csv", queryset2);
	//bool loaded3 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Query_1D.csv", queryset3);

	arma::rowvec query_x_low = queryset1.row(0);
	arma::rowvec query_x_up = queryset1.row(1);

	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	double x, x2, x3, x4;
	double a1 = 1.1, a2 = 2.2, a3 = 3.3, a4 = 4.4, b = 5.5;
	double result;
	vector<double> results;


	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		x = queryset_x_low_v[i];
		result = a1 * x + b;
		results.push_back(result);
		result = a1 * x + b;
		results.push_back(result);
	}
	results.clear();


	//query_x_low = queryset2.row(0);
	//query_x_up = queryset2.row(1);
	//queryset_x_up_v, queryset_x_low_v;
	//RowvecToVector(query_x_up, queryset_x_up_v);
	//RowvecToVector(query_x_low, queryset_x_low_v);

	auto t0 = chrono::steady_clock::now();

	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		x = queryset_x_low_v[i];
		result = a1 * x + b;
		results.push_back(result);
		x = queryset_x_up_v[i];
		result = a1 * x + b;
		results.push_back(result);
	}

	auto t1 = chrono::steady_clock::now();
	cout << "total time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count(  ) << " ns" << endl;
	cout << "average time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset1.size() / queryset1.n_rows) << " ns" << endl;
	results.clear();

	//query_x_low = queryset3.row(0);
	//query_x_up = queryset3.row(1);
	//queryset_x_up_v, queryset_x_low_v;
	//RowvecToVector(query_x_up, queryset_x_up_v);
	//RowvecToVector(query_x_low, queryset_x_low_v);

	t0 = chrono::steady_clock::now();

	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		x = queryset_x_low_v[i];
		result = a1 * x + a2 * x * x + b;
		results.push_back(result);
		x = queryset_x_up_v[i];
		result = a1 * x + a2 * x * x + b;
		results.push_back(result);
	}

	t1 = chrono::steady_clock::now();
	cout << "total time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "average time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset1.size() / queryset1.n_rows) << " ns" << endl;
	results.clear();

	//query_x_low = queryset1.row(0);
	//query_x_up = queryset1.row(1);
	//queryset_x_up_v, queryset_x_low_v;
	//RowvecToVector(query_x_up, queryset_x_up_v);
	//RowvecToVector(query_x_low, queryset_x_low_v);

	t0 = chrono::steady_clock::now();

	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		x = queryset_x_low_v[i];
		result = a1 * x + a2 * x * x + a3 * x * x * x + b;
		results.push_back(result);
		x = queryset_x_up_v[i];
		result = a1 * x + a2 * x * x + a3 * x * x * x + b;
		results.push_back(result);
	}

	t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset1.size() / queryset1.n_rows) << " ns" << endl;
	results.clear();
	t0 = chrono::steady_clock::now();

	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		x = queryset_x_low_v[i];
		result = a1 * x + a2 * x * x + a3 * x * x * x + a3 * x * x * x * x + b;
		results.push_back(result);
		x = queryset_x_up_v[i];
		result = a1 * x + a2 * x * x + a3 * x * x * x + a3 * x * x * x * x + b;
		results.push_back(result);
	}

	t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset1.size() / queryset1.n_rows) << " ns" << endl;
	results.clear();
}

void measuretime() {
	mat queryset1, queryset2, queryset3;
	bool loaded1 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset1);
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Query_1D.csv", queryset2);
	bool loaded3 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Query_1D.csv", queryset3);

	arma::rowvec query_x_low = queryset1.row(0);
	arma::rowvec query_x_up = queryset1.row(1);

	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	double x, x2, x3, x4;
	double a1 = 1.1, a2 = 2.2, a3 = 3.3, a4 = 4.4, b = 5.5;
	double result;
	vector<double> results;


	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		result = a1 * queryset_x_low_v[i] + b;
		results.push_back(result);
		result = a1 * queryset_x_up_v[i] + b;
		results.push_back(result);
	}
	results.clear();


	query_x_low = queryset2.row(0);
	query_x_up = queryset2.row(1);
	queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	auto t0 = chrono::steady_clock::now();

	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		result = a1 * queryset_x_low_v[i] + b;
		results.push_back(result);
		result = a1 * queryset_x_up_v[i] + b;
		results.push_back(result);
	}

	auto t1 = chrono::steady_clock::now();
	cout << "total time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "average time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset1.size() / queryset1.n_rows) << " ns" << endl;
	results.clear();

	query_x_low = queryset3.row(0);
	query_x_up = queryset3.row(1);
	queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	t0 = chrono::steady_clock::now();

	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		result = a1 * queryset_x_low_v[i] + a2 * pow(queryset_x_low_v[i],2) +  b;
		results.push_back(result);
		result = a1 * queryset_x_up_v[i] + a2 * pow(queryset_x_up_v[i], 2) + b;
		results.push_back(result);
	}

	t1 = chrono::steady_clock::now();
	cout << "total time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "average time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset1.size() / queryset1.n_rows) << " ns" << endl;
	results.clear();

	query_x_low = queryset1.row(0);
	query_x_up = queryset1.row(1);
	queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	t0 = chrono::steady_clock::now();

	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		result = a1 * queryset_x_low_v[i] + a2 * pow(queryset_x_low_v[i], 2) + a3 * pow(queryset_x_low_v[i], 3) + b;
		results.push_back(result);
		result = a1 * queryset_x_up_v[i] + a2 * pow(queryset_x_up_v[i], 2) + a3 * pow(queryset_x_up_v[i], 3) + b;
		results.push_back(result);
	}

	t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset1.size() / queryset1.n_rows) << " ns" << endl;
	results.clear();
	t0 = chrono::steady_clock::now();

	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		result = a1 * queryset_x_low_v[i] + a2 * pow(queryset_x_low_v[i], 2) + a3 *  pow(queryset_x_low_v[i], 3) + a3 * pow(queryset_x_low_v[i], 4) + b;
		results.push_back(result);
		result = a1 * queryset_x_up_v[i] + a2 * pow(queryset_x_up_v[i], 2) + a3 * pow(queryset_x_up_v[i], 3) + a3 * pow(queryset_x_up_v[i], 4) + b;
		results.push_back(result);
	}

	t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset1.size() / queryset1.n_rows) << " ns" << endl;
	results.clear();
}

void TestX(int lower, int upper) {

	TestX1(lower, upper);
	TestX2(lower, upper);
	TestX3(lower, upper);
	TestX4(lower, upper);
}

void TestX4_Complete() {
	ROLLearnedIndex_quartic learnedindex(9, 1000, 100000); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x4.csv", dataset);

	arma::mat trainingset = dataset.rows(0, 3);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	learnedindex.TrainBottomLayerWithFastDetectForMaxLoss(trainingset, responses, 200000, 2, 0.1);
	learnedindex.BuildNonLeafLayerWithBtree(); // btree


	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);

	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	auto t0 = chrono::steady_clock::now();


	learnedindex.PredictWithStxBtree(queryset_x_up_v, predicted_results_x_up);
	learnedindex.PredictWithStxBtree(queryset_x_low_v, predicted_results_x_low);

	for (int i = 0; i < predicted_results_x_up.size(); i++) {
		predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);
	}

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "bottom model count: " << learnedindex.dataset_range[learnedindex.level - 1].size() << endl;

	cout << "Total nodes in btree index: " << learnedindex.bottom_layer_index.CountNodesPrimary() << endl;
	cout << "Total parameters in btree index: " << learnedindex.bottom_layer_index.CountParametersPrimary() << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted1DimResults_LEARN.csv");
	for (int i = 0; i < predicted_results.size(); i++) {
		outfile << predicted_results[i] << endl;
	}
	outfile.close();

	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

void TestX4(int lower = 0, int upper = 1000) {
	ROLLearnedIndex_quartic learnedindex(9, 1000, 100); // level, step, error

	mat dataset;
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x4.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/ProcessedFinancialData.csv", dataset); // the financial dataset
	bool loaded = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/ProcessedFinancialData_X4.csv", dataset); // the financial dataset

	dataset = dataset.cols(lower, upper);

	arma::mat trainingset = dataset.rows(0, 3);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(dataset.row(3), key_v);
	RowvecToVector(responses, position_v);

	double a, b, c, d, e, current_absolute_accuracy;

	learnedindex.ApproximateMaxLossLinearRegression(0, key_v.size() - 1, key_v, position_v, trainingset, responses, 100, 500, 0.1, a, b, c, d, e, current_absolute_accuracy);

	cout << "a4: " << a << endl;
	cout << "a3: " << b << endl;
	cout << "a2: " << c << endl;
	cout << "a1: " << d << endl;
	cout << "a0: " << e << endl;
}

void TestX3_Complete() {
	ROLLearnedIndex_cubic learnedindex(9, 1000, 100); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x3.csv", dataset);

	arma::mat trainingset = dataset.rows(0, 2);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	learnedindex.TrainBottomLayerWithFastDetectForMaxLoss(trainingset, responses, 10000, 4, 10); // step , control, rep points
	learnedindex.BuildNonLeafLayerWithBtree(); // btree


	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);

	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	auto t0 = chrono::steady_clock::now();


	learnedindex.PredictWithStxBtree(queryset_x_up_v, predicted_results_x_up);
	learnedindex.PredictWithStxBtree(queryset_x_low_v, predicted_results_x_low);

	for (int i = 0; i < predicted_results_x_up.size(); i++) {
		predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);
	}

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "bottom model count: " << learnedindex.dataset_range[learnedindex.level - 1].size() << endl;

	cout << "Total nodes in btree index: " << learnedindex.bottom_layer_index.CountNodesPrimary() << endl;
	cout << "Total parameters in btree index: " << learnedindex.bottom_layer_index.CountParametersPrimary() << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted1DimResults_LEARN.csv");
	for (int i = 0; i < predicted_results.size(); i++) {
		outfile << predicted_results[i] << endl;
	}
	outfile.close();

	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

void TestX3(int lower = 0, int upper = 1000) {
	ROLLearnedIndex_cubic learnedindex(9, 1000, 100); // level, step, error

	mat dataset;
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x3.csv", dataset);
	bool loaded = mlpack::data::Load("C:/Users/Cloud/iCloudDrive/ProcessedFinancialData_X4.csv", dataset); // the financial dataset

	dataset = dataset.cols(lower, upper);

	arma::mat trainingset = dataset.rows(1, 3);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(dataset.row(2), key_v);
	RowvecToVector(responses, position_v);

	double a, b, c, d, current_absolute_accuracy;

	learnedindex.ApproximateMaxLossLinearRegression(0, key_v.size()-1, key_v, position_v, trainingset, responses, 20, 500, 0.1, a, b, c, d, current_absolute_accuracy);
	cout << "a3: " << a << endl;
	cout << "a2: " << b << endl;
	cout << "a1: " << c << endl;
	cout << "a0: " << d << endl;
}

void TestX2(int lower = 0, int upper = 1000) {
	ROLLearnedIndex_quadratic learnedindex(9, 1000, 100); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x2.csv", dataset);

	dataset = dataset.cols(lower, upper);

	arma::mat trainingset = dataset.rows(0, 1);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(dataset.row(1), key_v);
	RowvecToVector(responses, position_v);

	double a, b, c, current_absolute_accuracy;

	learnedindex.ApproximateMaxLossLinearRegression(0, key_v.size() - 1, key_v, position_v, trainingset, responses, 2, 100, 0.1, a, b, c, current_absolute_accuracy);
}

void TestX1(int lower = 0, int upper = 1000) {
	StageModelBottomUp learnedindex(9, 1000, 100); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x2.csv", dataset);

	dataset = dataset.cols(lower, upper);

	arma::mat trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	//trainingset = trainingset.cols(0, 1000);
	//responses = responses.cols(0, 1000);

	vector<double> key_v, position_v;
	RowvecToVector(trainingset, key_v);
	RowvecToVector(responses, position_v);

	double a, b, current_absolute_accuracy=0;

	learnedindex.ApproximateMaxLossLinearRegression(0, key_v.size() - 1, key_v, position_v, trainingset, responses, 2, 10, 0.1, a, b, current_absolute_accuracy);
}

void TestROLQuadratic() {
	ROLLearnedIndex_quadratic learnedindex_quadratic(9, 10000, 100); // level, step, error

	mat dataset;
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_SUM_2.csv", dataset); // this is for sum query
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset); // the 1M tweet dataset
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv", dataset);
	arma::mat trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);


	/*arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x2.csv", dataset);
	arma::mat trainingset = dataset.rows(0,1);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);*/

	//trainingset = trainingset.cols(0, 10000);
	//responses = responses.cols(0, 10000);

	auto t00 = chrono::steady_clock::now();
	//learnedindex_quadratic.TrainBottomLayerWithFastDetectForMaxLoss(trainingset, responses, 1000, 2, 0.1);
	learnedindex_quadratic.TrainBottomLayerWithSegmentOnTrainMaxLossOptimized(trainingset, responses);
	auto t11 = chrono::steady_clock::now();
	cout << "Construction Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() << " ns" << endl;

	learnedindex_quadratic.BuildNonLeafLayerWithBtree(); // btree

	//learnedindex_quadratic.DumpParameter();

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);

	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	auto t0 = chrono::steady_clock::now();

	
	learnedindex_quadratic.PredictWithStxBtree(queryset_x_up_v, predicted_results_x_up);
	learnedindex_quadratic.PredictWithStxBtree(queryset_x_low_v, predicted_results_x_low);

	for (int i = 0; i < predicted_results_x_up.size(); i++) {
		predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);
	}

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "bottom model count: " << learnedindex_quadratic.dataset_range[learnedindex_quadratic.level - 1].size() << endl;
	
	cout << "Total nodes in btree index: " << learnedindex_quadratic.bottom_layer_index.CountNodesPrimary() << endl;
	cout << "Total parameters in btree index: " << learnedindex_quadratic.bottom_layer_index.CountParametersPrimary() << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted1DimResults_LEARN.csv");
	for (int i = 0; i < predicted_results.size(); i++) {
		outfile << predicted_results[i] << endl;
	}
	outfile.close();

	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

void TestQuadraticTime() {

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(trainingset, key_v);
	double a = 10.7, b = 12.5, c = 19.1;
	double result;

	auto t0 = chrono::steady_clock::now();

	for (int i = 0; i < 1000; i++) {
		//result = a * key_v[i] * key_v[i] + b * key_v[i] + c;
		result = a * key_v[i] + b;
	}

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
}

struct Point
{
	double x, y;
};

// A global point needed for  sorting points with reference 
// to  the first point Used in compare function of qsort() 
Point p0;

Point nextToTop(stack<Point> &S)
{
	Point p = S.top();
	S.pop();
	Point res = S.top();
	S.push(p);
	return res;
}

// A utility function to swap two points 
void swap(Point &p1, Point &p2)
{
	Point temp = p1;
	p1 = p2;
	p2 = temp;
}

// A utility function to return square of distance 
// between p1 and p2 
int distSq(Point p1, Point p2)
{
	return (p1.x - p2.x)*(p1.x - p2.x) +
		(p1.y - p2.y)*(p1.y - p2.y);
}

int orientation(Point p, Point q, Point r)
{
	int val = (q.y - p.y) * (r.x - q.x) -
		(q.x - p.x) * (r.y - q.y);

	if (val == 0) return 0;  // colinear 
	return (val > 0) ? 1 : 2; // clock or counterclock wise 
}

// A function used by library function qsort() to sort an array of 
// points with respect to the first point 
int compare(const void *vp1, const void *vp2)
{
	Point *p1 = (Point *)vp1;
	Point *p2 = (Point *)vp2;

	// Find orientation 
	int o = orientation(p0, *p1, *p2);
	if (o == 0)
		return (distSq(p0, *p2) >= distSq(p0, *p1)) ? -1 : 1;

	return (o == 2) ? -1 : 1;
}

void TestConvexHullMethod() {
	
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(trainingset, key_v);
	RowvecToVector(responses, position_v);

	auto t0 = chrono::steady_clock::now();
	//======================================

	// form points
	vector<Point> points;
	for (int i = 0; i < position_v.size(); i++) {
		Point p;
		p.x = key_v[i];
		p.y = position_v[i];
		points.push_back(p);
	}

	// Find the bottommost point 
	int ymin = points[0].y, min = 0;
	for (int i = 1; i < points.size(); i++)
	{
		int y = points[i].y;

		// Pick the bottom-most or chose the left 
		// most point in case of tie 
		if ((y < ymin) || (ymin == y &&
			points[i].x < points[min].x))
			ymin = points[i].y, min = i;
	}

	// Place the bottom-most point at first position 
	swap(points[0], points[min]);

	// Sort n-1 points with respect to the first point. 
	// A point p1 comes before p2 in sorted output if p2 
	// has larger polar angle (in counterclockwise 
	// direction) than p1 
	p0 = points[0];
	qsort(&points[1], points.size()-1, sizeof(Point), compare);

	// If two or more points make same angle with p0, 
	// Remove all but the one that is farthest from p0 
	// Remember that, in above sorting, our criteria was 
	// to keep the farthest point at the end when more than 
	// one points have same angle. 
	int m = 1; // Initialize size of modified array 
	for (int i = 1; i < points.size(); i++)
	{
		// Keep removing i while angle of i and i+1 is same 
		// with respect to p0 
		while (i < points.size()-1 && orientation(p0, points[i],
			points[i + 1]) == 0)
			i++;


		points[m] = points[i];
		m++;  // Update size of modified array 
	}

	// If modified array of points has less than 3 points, 
	// convex hull is not possible 
	if (m < 3) return;

	// Create an empty stack and push first three points 
	// to it. 
	stack<Point> S;
	S.push(points[0]);
	S.push(points[1]);
	S.push(points[2]);

	// Process remaining n-3 points 
	for (int i = 3; i < m; i++)
	{
		// Keep removing top while the angle formed by 
		// points next-to-top, top, and points[i] makes 
		// a non-left turn 
		while (orientation(nextToTop(S), S.top(), points[i]) != 2)
			S.pop();
		S.push(points[i]);
	}

	// Now stack has the output points, print contents of stack 
	//while (!S.empty())
	//{
	//	Point p = S.top();
	//	cout << "(" << p.x << ", " << p.y << ")" << endl;
	//	S.pop();
	//}

	//======================================
	vector<double> target_key_v, target_pos_v;
	while (!S.empty())
	{
		Point p = S.top();
		target_key_v.push_back(p.x);
		target_pos_v.push_back(p.y);
		S.pop();
	}

	arma::rowvec current_trainingset, current_response;

	VectorToRowvec(current_trainingset, target_key_v);
	VectorToRowvec(current_response, target_pos_v);

	LinearRegression lr(current_trainingset, current_response);

	arma::vec paras = lr.Parameters();
	double a = paras[1]; // a
	double b = paras[0]; // b

	// find the max positive error and the max negative error
	double max_pos_err = 0, max_neg_err = 0, maximum_error = 0, abs_error;
	double predicted_position, error;
	for (int i = 0; i < key_v.size(); i++) {
		predicted_position = a * key_v[i] + b;
		error = position_v[i] - predicted_position;
		abs_error = abs(error);
		if (abs_error > maximum_error) {
			maximum_error = abs_error;
		}
		/*if (error > max_pos_err) {
			max_pos_err = error;
		}
		else if (error < max_neg_err) {
			max_neg_err = error;
		}*/
	}
	double adjusted_b = b + (max_pos_err - abs(max_neg_err)) / 2;
	double maxerr = (max_pos_err - max_neg_err) / 2;


	auto t1 = chrono::steady_clock::now();
	cout << maxerr << " " << maximum_error << endl;
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
}

void TestEqualWidthSampling() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(trainingset, key_v);
	RowvecToVector(responses, position_v);

	double step = (key_v[key_v.size() - 1] - key_v[0]) / 100;
	double target_key = key_v[0];
	vector<double> target_key_v, target_pos_v;

	auto t0 = chrono::steady_clock::now();

	for(int i = 0; i < key_v.size(); i++){
		if (key_v[i] >= target_key) {
			target_key_v.push_back(key_v[i]);
			target_pos_v.push_back(position_v[i]);
			target_key += step;
		}
	}

	arma::rowvec current_trainingset, current_response;

	VectorToRowvec(current_trainingset, target_key_v);
	VectorToRowvec(current_response, target_pos_v);

	LinearRegression lr(current_trainingset, current_response);

	arma::vec paras = lr.Parameters();
	double a = paras[1]; // a
	double b = paras[0]; // b

	// find the max positive error and the max negative error
	double max_pos_err = 0, max_neg_err = 0, maximum_error = 0, abs_error;
	double predicted_position, error;
	for (int i = 0; i < key_v.size(); i++) {
		predicted_position = a * key_v[i] + b;
		error = position_v[i] - predicted_position;
		abs_error = abs(error);
		if (abs_error > maximum_error) {
			maximum_error = abs_error;
		}
		/*if (error > max_pos_err) {
			max_pos_err = error;
		}
		else if (error < max_neg_err) {
			max_neg_err = error;
		}*/
	}
	double adjusted_b = b + (max_pos_err - abs(max_neg_err)) / 2;
	double maxerr = (max_pos_err - max_neg_err) / 2;


	auto t1 = chrono::steady_clock::now();
	cout << maxerr << " " << maximum_error << endl;
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
}

void TestTimeLinearSegment() {
	StageModelBottomUp stage_model_bottom_up(9, 1000, 100); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(trainingset, key_v);
	RowvecToVector(responses, position_v);

	auto t0 = chrono::steady_clock::now();

	double a, b;

	a = (position_v[position_v.size()-1] - position_v[0]) / (key_v[key_v.size() - 1] - key_v[0]); // monotonous
	b = position_v[0];

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
}

void TestStageModelBottomUpMaxLoss() {
	StageModelBottomUp stage_model_bottom_up(9, 1000, 100); // level, step, error

	mat dataset;

	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_SUM_2.csv", dataset); // this is for sum query

	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset); // the 1M tweet dataset
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv", dataset);

	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	cout << trainingset.size() << endl;
	//trainingset = trainingset.cols(0, 10000);
	//responses = responses.cols(0, 10000);

	auto t00 = chrono::steady_clock::now();
	//stage_model_bottom_up.TrainBottomLayerWithFastDetectForMaxLoss(trainingset, responses,1000, 2, 5);// do not need to dump parameter, step control, representative points
	//stage_model_bottom_up.TrainBottomLayerWithSegmentOnTrainMaxLoss(trainingset, responses);// do not need to dump parameter, step control, representative points
	stage_model_bottom_up.TrainBottomLayerWithSegmentOnTrainMaxLossOptimized(trainingset, responses); // faster, using LP
	//stage_model_bottom_up.TrainBottomLayerWithSegmentOnTrainApproximateMaxLossOptimized(trainingset, responses); // faster, using Approximate max loss
	auto t11 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() << " ns" << endl;

	stage_model_bottom_up.BuildNonLeafLayerWithBtree(); // btree

	vector<double> trainingset_v, responses_v;
	RowvecToVector(trainingset, trainingset_v);
	RowvecToVector(responses, responses_v);
 	//stage_model_bottom_up.SelfChecking(trainingset_v, responses_v);

	cout.precision(11);
	/*cout << "dataset_range[i].first" << " " << "dataset_range[i].second" << " " << "parameters[i].first" << " " << "parameters[i].second" << " " << "y2" << endl;
	for (int i = 0; i < stage_model_bottom_up.stage_model_parameter[8].size(); i++) {
		cout << stage_model_bottom_up.dataset_range[8][i].first << " " << stage_model_bottom_up.dataset_range[8][i].second << " " << stage_model_bottom_up.stage_model_parameter[8][i][0] << " " << stage_model_bottom_up.stage_model_parameter[8][i][1] << "  y1:" << stage_model_bottom_up.dataset_range[8][i].first * stage_model_bottom_up.stage_model_parameter[8][i][0] + stage_model_bottom_up.stage_model_parameter[8][i][1] << "  y2: " << stage_model_bottom_up.dataset_range[8][i].second * stage_model_bottom_up.stage_model_parameter[8][i][0] + stage_model_bottom_up.stage_model_parameter[8][i][1] << endl;
	}*/

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Query_1D.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Query_1D.csv", queryset);

	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);

	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	auto t0 = chrono::steady_clock::now();

	stage_model_bottom_up.PredictWithStxBtree(queryset_x_up_v, predicted_results_x_up);
	stage_model_bottom_up.PredictWithStxBtree(queryset_x_low_v, predicted_results_x_low);

	double t_abs = stage_model_bottom_up.threshold;
	int count_over = 0, count_negative = 0;
	for (int i = 0; i < predicted_results_x_up.size(); i++) {
		predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);

		// analysis estimated maximum relative error:
		double max_err_rel = (2 * t_abs) / (predicted_results[i] - 2 * t_abs);
		if (max_err_rel > 0.01) {
			count_over++;
		}
		else if (max_err_rel < 0) {
			count_negative++;
		}
		//cout << "maximum relative error: " << max_err_rel * 100 << "%" << endl;
	}

	auto t1 = chrono::steady_clock::now();

	cout << "over threshold relative error: " << count_over << "   negative relative error: " << count_negative << endl;

	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "bottom model count: " << stage_model_bottom_up.dataset_range[stage_model_bottom_up.level - 1].size() << endl;
	cout << "Total nodes in btree index: " << stage_model_bottom_up.bottom_layer_index.CountNodesPrimary() << endl;
	cout << "Total parameters in btree index: " << stage_model_bottom_up.bottom_layer_index.CountParametersPrimary() << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted1DimResults_LEARN.csv");
	for (int i = 0; i < predicted_results.size(); i++) {
		outfile << predicted_results[i] << endl;
	}
	outfile.close();

	//CalculateRealCountWithScan1D(queryset, real_results, "C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Sorted_Value.csv");
	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results); // need to adjust the error threshold by hand!!!!
}

void TestMaxLoss() {
	StageModelBottomUp stage_model_bottom_up(9, 1000, 100); // level, step, error

	mat dataset;
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_SUM_2.csv", dataset); // this is for sum query
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(trainingset, key_v);
	RowvecToVector(responses, position_v);

	double a, b, c;

	//stage_model_bottom_up.ApproximateMaxLossLinearRegressionMultipass(0, key_v.size()-1, key_v, position_v, trainingset, responses, 2, 1000, 10, a,b,c, 5);

	stage_model_bottom_up.MyCplexSolverForMaxLossOptimized(key_v, position_v, 11370, 11764, a, b, c);
	cout << "slope: " << a << "  bias: " << b << "  max error: " << c << endl;
}

void TestLoadingNumpySaved() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/MaxLoss_2000.csv", dataset);
	
	arma::mat submat = dataset.rows(0, 5);
	cout << submat.n_rows << endl;

	//arma::rowvec trainingset1 = dataset.row(0); // the first column
	//arma::rowvec trainingset2 = dataset.row(1); // the second column
	//
	//cout << trainingset1.at(0, 73) << endl;
	
	/*trainingset1.print();

	vector<double> keys;
	RowvecToVector(trainingset1, keys);
	for (int i = 0; i < keys.size(); i++) {
		cout << keys[i] << endl;
	}*/
}

void TestSegmentation() {

	int segmentations[] = { 0,9124, 152159, 254259, 582548, 1014537, 1145349, 1157569 };
	//int segmentations[] = { 0,1157569 };

	int start_pos, end_pos;
	int total_count = 0;
	for (int i = 0; i < 7; i++) {
		start_pos = segmentations[i];
		end_pos = segmentations[i+1];
		StageModelBottomUp stage_model_bottom_up(9, end_pos - start_pos, 100);

		mat dataset;
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
		arma::rowvec trainingset = dataset.row(0);
		arma::rowvec responses = dataset.row(dataset.n_rows - 1);

		trainingset = trainingset.cols(start_pos, end_pos);
		responses = responses.cols(start_pos, end_pos);
		stage_model_bottom_up.TrainBottomLayerWithFastDetect(trainingset, responses);
		total_count += stage_model_bottom_up.dataset_range[stage_model_bottom_up.level - 1].size();
	}

	cout << "total models: " << total_count;
}

void TestPrefixSumHistogram() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> keys;
	RowvecToVector(trainingset, keys);

	Histogram hist(keys);

	arma::mat splits_mat;
	splits_mat.load("C:/Users/Cloud/Desktop/LearnIndex/data/EntropyHistSplits5000.bin");
	vector<int> splits;
	for (int i = 0; i < splits_mat.n_rows; i++) {
		splits.push_back(splits_mat[i]);
	}
	hist.GenerateHistogramFromSplits(splits);

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results, real_results;

	hist.GeneratePrefixSUM();

	auto t0 = chrono::steady_clock::now();

	//hist.Query(queryset_x_low_v, queryset_x_up_v, predicted_results);
	hist.QueryWithPrefixSUM(queryset_x_low_v, queryset_x_up_v, predicted_results);

	auto t1 = chrono::steady_clock::now();

	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

void TestEntropyHistOptimized() {

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> keys;
	RowvecToVector(trainingset, keys);
	
	EntropyHist(500, keys);

	arma::mat splits_mat;
	splits_mat.load("C:/Users/Cloud/Desktop/LearnIndex/data/EntropyHistSplits500.bin");
	vector<int> splits;
	for (int i = 0; i < splits_mat.n_rows; i++) {
		splits.push_back(splits_mat[i]);
	}

	vector<vector<double>> buckets; // start_key, end_key, frequency
	vector<double> bucket;
	int frequency = 0;
	int start_index = 0;
	int end_index = 0;
	double start_key = 0, end_key = 0;

	stx::btree<double, int> stx_hist;

	for (int i = 0; i < splits.size(); i++) {

		start_index = splits[i];
		if (i + 1 < splits.size()) {
			end_index = splits[i + 1];
		}
		else {
			end_index = keys.size()-1;
		}

		frequency = 0;
		start_key = keys[start_index];
		end_key = keys[end_index];
		for (int j = start_index; j < end_index; j++) {
			frequency++;
		}

		bucket.clear();
		bucket.push_back(start_key);
		bucket.push_back(end_key);
		bucket.push_back(frequency);
		
		buckets.push_back(bucket);

		stx_hist.insert(pair<double, int>(start_key, i));
	}

	// query ============================
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	stx::btree<double, int>::iterator iter1, iter2, temp1, temp2;
	auto t0 = chrono::steady_clock::now();

	// get range count

	double start_key1, start_key2, end_key1, end_key2, low_frequency, up_frequency;
	//int start_index, end_index;

	for (int i = 0; i < queryset_x_up_v.size(); i++) {
		int query_count = 0;

		iter1 = stx_hist.lower_bound(queryset_x_low_v[i]);
		iter2 = stx_hist.lower_bound(queryset_x_up_v[i]);

		start_index = iter1->second;
		end_index = iter2->second;

		low_frequency = buckets[start_index][2];
		start_key1 = buckets[start_index][0];
		start_key2 = buckets[start_index][1];

		up_frequency = buckets[end_index][2];
		end_key1 = buckets[end_index][0];
		end_key2 = buckets[end_index][1];

		query_count += (start_key2 - queryset_x_low_v[i]) / (start_key2 - start_key1) * low_frequency;
		start_index++;

		while (start_index < end_index) {
			//cout << start_index << " " << end_index << " " << i << endl;
			query_count += buckets[start_index][2];
			start_index++;
		}

		query_count += (queryset_x_up_v[i] - end_key1) / (end_key2 - end_key1) * up_frequency;

		predicted_results.push_back(query_count);
	}

	auto t1 = chrono::steady_clock::now();

	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

void TestEntropyHist() {

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> keys;
	RowvecToVector(trainingset, keys);

	EntropyHist(5000, keys);

	arma::mat splits_mat;
	splits_mat.load("C:/Users/Cloud/Desktop/LearnIndex/data/EntropyHistSplits5000.bin");
	vector<int> splits;
	for (int i = 0; i < splits_mat.n_rows; i++) {
		splits.push_back(splits_mat[i]);
	}

	// generate statistic info according to splits
	
	vector<pair<double, int>> key_frequency;
	int frequency = 0;
	int start_index = 0;
	int end_index = 0;
	double start_key = 0;
	for (int i = 0; i < splits.size(); i++) {

		start_index = splits[i];
		if (i + 1 < splits.size()) {
			end_index = splits[i + 1];
		} else {
			end_index = keys.size();
		}

		frequency = 0;
		start_key = keys[start_index];
		for (int j = start_index; j < end_index; j++) {
			frequency++;
		}
		key_frequency.push_back(pair<double, int>(start_key, frequency));
	}
	// save the generated frequency result!

	stx::btree<double, int> stx_hist;

	for (int i = 0; i < key_frequency.size(); i++) {
		stx_hist.insert(pair<double, int>(key_frequency[i].first, key_frequency[i].second));
	}

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	stx::btree<double, int>::iterator iter1, iter2, temp1, temp2;
	auto t0 = chrono::steady_clock::now();

	// get range count

	double start_key1, start_key2, end_key1, end_key2, low_frequency, up_frequency;

	for (int i = 0; i < queryset_x_up_v.size(); i++) {
		int query_count = 0;

		iter1 = stx_hist.lower_bound(queryset_x_low_v[i]);
		iter2 = stx_hist.lower_bound(queryset_x_up_v[i]);

		low_frequency = iter1->second;
		start_key1 = iter1->first;
		temp1 = iter1;
		temp1++;
		start_key2 = temp1->first;

		up_frequency = iter2->second;
		end_key1 = iter2->first;
		temp2 = iter2;
		temp2++;
		end_key2 = temp2->first;

		while (iter1 != iter2) {
			query_count += iter1->second;
			iter1++;
		}

		//cout << "here: " << (iter1 == iter2) << endl;

		query_count -= (queryset_x_low_v[i] - start_key1) / (start_key2 - start_key1)*low_frequency;
		//query_count -= (end_key2 - queryset_x_up_v[i]) / (end_key2 - end_key1)*up_frequency;
		query_count += (queryset_x_up_v[i] - end_key1) / (end_key2 - end_key1) * up_frequency;

		predicted_results.push_back(query_count);
	}

	auto t1 = chrono::steady_clock::now();

	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	CalculateRealCountWithScan1D (queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

void TestDouglas() {

	Douglas douglas(10000);

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	douglas.BuildDouglas(trainingset, responses);

	cout.precision(11);
	cout << "dataset_range[i].first" << " " << "dataset_range[i].second" << " " << "parameters[i].first" << " " << "parameters[i].second" << " " << "y2" << endl;
	for (int i = 0; i < douglas.parameters.size(); i++) {
		cout << douglas.dataset_range[i].first << " " << douglas.dataset_range[i].second << " " << douglas.parameters[i].first << " " << douglas.parameters[i].second << " " <<(douglas.dataset_range[i].second - douglas.dataset_range[i].first)*douglas.parameters[i].first + douglas.parameters[i].second << endl;
	}

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	auto t0 = chrono::steady_clock::now();

	douglas.Predict(queryset_x_up_v, predicted_results_x_up);
	douglas.Predict(queryset_x_low_v, predicted_results_x_low);

	for (int i = 0; i < predicted_results_x_up.size(); i++) {
		predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);
	}

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "bottom model count: " << douglas.dataset_range.size() << endl;
	cout << "Total nodes in btree index: " << douglas.bottom_layer_index.CountNodesPrimary() << endl;
	cout << "Total parameters in btree index: " << douglas.bottom_layer_index.CountParametersPrimary() << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted1DimResults_LEARN.csv");
	for (int i = 0; i < predicted_results.size(); i++) {
		outfile << predicted_results[i] << endl;
	}
	outfile.close();

	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

void TestSampleling(double sample_percentage = 1.0) {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	vector<double> key_v;
	RowvecToVector(trainingset, key_v);

	// do sampling
	srand((unsigned)time(NULL));
	std::random_shuffle(key_v.begin(), key_v.end());
	int sample_total = key_v.size() * sample_percentage;
	vector<double> key_v_sampled;
	for (int i = 0; i < sample_total; i++) {
		key_v_sampled.push_back(key_v[i]);
	}

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	auto t0 = chrono::steady_clock::now();

	double count = 0;
	int predicted_count = 0;
	for (int i = 0; i < queryset_x_up_v.size(); i++) {
		count = 0;
		predicted_count = 0;
		for (int j = 0; j < key_v_sampled.size(); j++) {
			if (key_v_sampled[j] <= queryset_x_up_v[i] && key_v_sampled[j] >= queryset_x_low_v[i]) {
				count++;
			}
		}
		predicted_count = count / sample_percentage;
		predicted_results.push_back(predicted_count);
	}

	auto t1 = chrono::steady_clock::now();

	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

// equal width histogram, directly locatable.
void TestHist(int number_of_hist = 1000) {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	vector<double> key_v;
	RowvecToVector(trainingset, key_v);

	// find the border of dataset; I think it should be the domain instead of the real dataset's occupation
	double max_dataset = 180, min_dataset = -180;
	double range_size = abs(max_dataset - min_dataset);
	double hist_width = range_size / number_of_hist;

	// generate hist data
	int hist_index;
	vector<int> hist;
	for (int i = 0; i < number_of_hist; i++) {
		hist.push_back(0);
	}

	for (int i = 0; i < key_v.size(); i++) {
		hist_index = (key_v[i] - min_dataset) / hist_width;
		hist[hist_index]++;
	}

	// query set
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	auto t0 = chrono::steady_clock::now();

	int hist_index_first, hist_index_last;
	double predicted_count;
	for (int i = 0; i < queryset_x_up_v.size(); i++) {
		
		hist_index_last = (queryset_x_up_v[i] - min_dataset) / hist_width;
		hist_index_first = (queryset_x_low_v[i] - min_dataset) / hist_width;

		if (hist_index_first != hist_index_last) {
			predicted_count = 0;

			// first part
			predicted_count = (((hist_index_first + 1) * hist_width - (queryset_x_low_v[i] - min_dataset)) / hist_width) * hist[hist_index_first];

			// middle part
			for (int j = hist_index_first + 1; j < hist_index_last; j++) {
				predicted_count += hist[j];
			}

			// last part
			predicted_count += (((queryset_x_up_v[i] - min_dataset) - hist_index_last * hist_width) / hist_width) * hist[hist_index_last];
		}
		else {
			predicted_count = hist[hist_index_first] * (queryset_x_up_v[i] - queryset_x_low_v[i]) / hist_width;
		}

		predicted_results.push_back(predicted_count);
	}


	auto t1 = chrono::steady_clock::now();

	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

// ascending order
bool cmp_index(int & record1, int & record2) {
	return record1 < record2;
}

void TestSTXBtree(double sample_percentage = 1.0) {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	vector<double> key_v, position_v;
	RowvecToVector(trainingset, key_v);
	RowvecToVector(responses, position_v);

	// do sampling
	vector<int> sample_index, sample_index2;
	for (int i = 0; i < key_v.size(); i++) {
		sample_index.push_back(i);
	}
	int sample_total = key_v.size() * sample_percentage;
	srand((unsigned)time(NULL));
	// use random shuffle !!!
	std::random_shuffle(sample_index.begin(), sample_index.end());

	for (int i = 0; i < sample_total; i++) {
		sample_index2.push_back(sample_index[i]);
	}
	sort(sample_index2.begin(), sample_index2.end(), cmp_index);
	vector<double> key_v_sampled, position_v_sampled;
	for (int i = 0; i < sample_index2.size(); i++) {
		key_v_sampled.push_back(key_v[sample_index2[i]]);
		position_v_sampled.push_back(position_v[sample_index2[i]]);
	}

	stx::btree<double, int> stx_index;
	for (int i = 0; i < key_v_sampled.size(); i++) {
		stx_index.insert(pair<double, int>(key_v_sampled[i], position_v_sampled[i]));
	}

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	auto t0 = chrono::steady_clock::now();

	int position = 0;
	stx::btree<double, int>::iterator iter;

	for (int i = 0; i < queryset_x_up_v.size(); i++) {
		iter = stx_index.lower_bound(queryset_x_up_v[i]);
		predicted_results_x_up.push_back(iter->second);
	}

	for (int i = 0; i < queryset_x_low_v.size(); i++) {
		iter = stx_index.lower_bound(queryset_x_low_v[i]);
		predicted_results_x_low.push_back(iter->second);
	}

	for (int i = 0; i < predicted_results_x_up.size(); i++) {
		predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);
	}

	auto t1 = chrono::steady_clock::now();

	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "Total nodes in btree index: " << stx_index.CountNodesPrimary() << endl;
	cout << "Total parameters in btree index: " << stx_index.CountParametersPrimary() << endl;
	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

// double Tabs=100, double Trel = 0.01, string recordfilepath= "C:/Users/Cloud/Desktop/LearnedAggregateData/experiment_result.csv"
void TestAtree(double Tabs, double Trel, string recordfilepath) {
	
	ATree atree(Tabs, Trel); // error threshold

	mat dataset;

	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_SUM_2.csv", dataset); // this is for sum query

	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_1M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Sorted_Value.csv", dataset);
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	auto t00 = chrono::steady_clock::now();
	atree.TrainAtree(trainingset, responses);
	auto t11 = chrono::steady_clock::now();
	cout << "Atree experiment: " << endl;
	cout << "Total Construction Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t11 - t00).count() << " ns" << endl;

	cout.precision(11);
	/*cout << "dataset_range[i].first" << " " << "dataset_range[i].second" << " " << "parameters[i].first" << " " << "parameters[i].second" << " " << "y2" << endl;
	for (int i = 0; i < atree.atree_parameters.size(); i++) {
		cout << atree.dataset_range[i].first << " " << atree.dataset_range[i].second << " " << atree.atree_parameters[i][0] << " " << atree.atree_parameters[i][1] << " " << (atree.dataset_range[i].second - atree.dataset_range[i].first)* atree.atree_parameters[i][0] + atree.atree_parameters[i][1] << endl;
	}*/
	
	mat queryset;
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_1M_Query_1D.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Query_1D.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Query_1D.csv", queryset);
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Query_1D.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);
	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	vector<double> key_v;
	RowvecToVector(trainingset, key_v);

	atree.CountPrediction(queryset_x_low_v, queryset_x_up_v, predicted_results, key_v, recordfilepath);

	//auto t0 = chrono::steady_clock::now();

	//atree.Predict(queryset_x_up_v, predicted_results_x_up);
	//atree.Predict(queryset_x_low_v, predicted_results_x_low);
	////atree.PredictExact(queryset_x_up_v, predicted_results_x_up);
	////atree.PredictExact(queryset_x_low_v, predicted_results_x_low);

	//double t_abs = atree.error_threshold;
	//int count_over = 0, count_negative = 0;

	//for (int i = 0; i < predicted_results_x_up.size(); i++) {
	//	predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);

	//	// analysis estimated maximum relative error:
	//	double max_err_rel = (2 * t_abs) / (predicted_results[i] - 2 * t_abs);
	//	if (max_err_rel > 0.01) {
	//		count_over++;
	//	}
	//	else if (max_err_rel < 0) {
	//		count_negative++;
	//	}
	//	//cout << "maximum relative error: " << max_err_rel * 100 << "%" << endl;
	//}

	//auto t1 = chrono::steady_clock::now();
	//cout << "over threshold relative error: " << count_over << "   negative relative error: " << count_negative << endl;

	//cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	//cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "bottom model count: " << atree.dataset_range.size() << endl;
	cout << "Total nodes in btree index: " << atree.bottom_layer_index.CountNodesPrimary() << endl;
	cout << "Total parameters in btree index: " << atree.bottom_layer_index.CountParametersPrimary() << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted1DimResults_LEARN.csv");
	for (int i = 0; i < predicted_results.size(); i++) {
		outfile << predicted_results[i] << endl;  
	}
	outfile.close();

	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

void TestStageModelBottomUp() {
	StageModelBottomUp stage_model_bottom_up(9,1000,100); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	//stage_model_bottom_up.TrainBottomLayer(trainingset, responses);
	//stage_model_bottom_up.TrainAdaptiveBottomLayer(trainingset, responses);
	//stage_model_bottom_up.TrainBottomLayerWithFastDetect(trainingset, responses);
	
	stage_model_bottom_up.TrainBottomLayerWithFastDetectOptimized(trainingset, responses);
	//stage_model_bottom_up.LoadBottomModelFromFile("C:/Users/Cloud/Desktop/LearnIndex/data/MaxLoss_2000.csv"); // do not need to dump parameter

	//stage_model_bottom_up.DumpParameter();

	stage_model_bottom_up.BuildNonLeafLayerWithBtree(); // btree
	//stage_model_bottom_up.BuildNonLeafLayerWithLR(8); // multi layer, parameter: non-leaf layer threhold
	//stage_model_bottom_up.BuildNonLeafLayerWithHybrid(10); // hybrid
	//stage_model_bottom_up.BuildTopDownModelForNonLeafLayer(); // top down learned index
	//stage_model_bottom_up.BuildTopDownModelForNonLeafLayer2(trainingset); // top down learned index using original dataset and model index to replace pos
	//stage_model_bottom_up.BuildAtreeForNonLeafIndex(32); // for this method, the dump parameter should execute first! parameter: error threshold

	stage_model_bottom_up.DumpParameter();

	mat queryset; 
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Query_1D.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Query_1D.csv", queryset);
	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);
	
	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	auto t0 = chrono::steady_clock::now();

	//stage_model_bottom_up.PredictWithHybrid(queryset_x_up_v, predicted_results_x_up); // prediction using hybrid without exact search
	//stage_model_bottom_up.PredictWithHybrid(queryset_x_low_v, predicted_results_x_low);
	//stage_model_bottom_up.PredictWithHybridShift(queryset_x_up_v, predicted_results_x_up); // prediction using hybrid with exact search
	//stage_model_bottom_up.PredictWithHybridShift(queryset_x_low_v, predicted_results_x_low);
	//stage_model_bottom_up.PredictWithLR(queryset_x_up_v, predicted_results_x_up); // prediction using lr without exact search
	//stage_model_bottom_up.PredictWithLR(queryset_x_low_v, predicted_results_x_low);
	//stage_model_bottom_up.PredictWithLRBinaryShiftOptimized(queryset_x_up_v, predicted_results_x_up); // prediction using lr with exact search
	//stage_model_bottom_up.PredictWithLRBinaryShiftOptimized(queryset_x_low_v, predicted_results_x_low);
	//stage_model_bottom_up.PredictWithLRBinary(queryset_x_up_v, predicted_results_x_up); // used for bounded atree non-leaf index
	//stage_model_bottom_up.PredictWithLRBinary(queryset_x_low_v, predicted_results_x_low);
	stage_model_bottom_up.PredictWithStxBtree(queryset_x_up_v, predicted_results_x_up);
	stage_model_bottom_up.PredictWithStxBtree(queryset_x_low_v, predicted_results_x_low);

	for (int i = 0; i < predicted_results_x_up.size(); i++) {
		predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);
	}

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "bottom model count: " << stage_model_bottom_up.dataset_range[stage_model_bottom_up.level-1].size() << endl;
	//cout << "Total nodes in btree index: " << stage_model_bottom_up.btree_model_index.CountNodesPrimary() << endl;
	//cout << "Total parameters in btree index: " << stage_model_bottom_up.btree_model_index.CountParametersPrimary() << endl;
	cout << "Total nodes in btree index: " << stage_model_bottom_up.bottom_layer_index.CountNodesPrimary() << endl;
	cout << "Total parameters in btree index: " << stage_model_bottom_up.bottom_layer_index.CountParametersPrimary() << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted1DimResults_LEARN.csv");
	for (int i = 0; i < predicted_results.size(); i++) {
		outfile << predicted_results[i] << endl;
	}
	outfile.close();

	//CalculateRealCountWithScan1D(queryset, real_results, "C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv");
	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

void Count2DLearnedIndex2D_2(int query_region) {
	vector<pair<int, int>> arch;
	arch.push_back(pair<int, int>(1,1));
	arch.push_back(pair<int, int>(3, 3));
	arch.push_back(pair<int, int>(10, 10));  
	arch.push_back(pair<int, int>(100, 100));

	StageModel2D_2 stage_model_2d_2(arch);

	mat dataset;
	mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimTrainingSet1000_1000.csv", dataset);
	arma::mat trainingset = dataset.rows(0, 1); // x and y
	arma::rowvec responses = dataset.row(2); // count
	stage_model_2d_2.Train(trainingset, responses);

	mat queryset;
	switch (query_region) {
	case 1:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection100m.csv", queryset);
		break;
	case 2:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection200m.csv", queryset);
		break;
	case 3:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection500m.csv", queryset);
		break;
	case 4:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection1km.csv", queryset);
		break;
	case 5:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection2km.csv", queryset);
		break;
	case 6:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection5km.csv", queryset);
		break;
	case 7:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection10km.csv", queryset);
		break;
	case 8:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection20km.csv", queryset);
		break;
	case 9:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection50km.csv", queryset);
		break;
	case 10:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection100km.csv", queryset);
		break;
	default:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimQuery2.csv", queryset);
		break;
	}
	mat queryset_x_low = queryset.row(0);
	mat queryset_x_up = queryset.row(1);
	mat queryset_y_low = queryset.row(2);
	mat queryset_y_up = queryset.row(3);
	vector<double> query_x_low_v, query_x_up_v, query_y_low_v, query_y_up_v;
	RowvecToVector(queryset_x_low, query_x_low_v);
	RowvecToVector(queryset_x_up, query_x_up_v);
	RowvecToVector(queryset_y_low, query_y_low_v);
	RowvecToVector(queryset_y_up, query_y_up_v);

	auto t0 = chrono::steady_clock::now();

	//double count, count_full, count_min, count_half_1, count_half_2;
	vector<double> count_upper_right, count_lower_left, count_upper_left, count_lower_right;
	stage_model_2d_2.PredictVector(query_x_up_v, query_y_up_v, count_upper_right);
	stage_model_2d_2.PredictVector(query_x_low_v, query_y_low_v, count_lower_left);
	stage_model_2d_2.PredictVector(query_x_low_v, query_y_up_v, count_upper_left);
	stage_model_2d_2.PredictVector(query_x_up_v, query_y_low_v, count_lower_right);

	vector<int> predicted_results;
	double count = 0;
	for (int i = 0; i < count_upper_right.size(); i++) {
		count = count_upper_right[i] - count_upper_left[i] - count_lower_right[i] + count_lower_left[i];
		predicted_results.push_back(count);
	}

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "finish stage model prediction.." << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_LEARN.csv");
	for (int i = 0; i < predicted_results.size(); i++) {
		outfile << predicted_results[i] << endl;
	}
	outfile.close();

	// measure accuracy
	vector<int> real_results;
	CalculateRealCountWithScan2D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

void Count2DLearnedIndex2D(int query_region) {
	vector<int> arch;
	arch.push_back(1);
	arch.push_back(10);
	arch.push_back(100);
	arch.push_back(1000);

	StageModel2D stage_model_2d(arch);

	mat dataset;
	mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimTrainingSet1000_1000.csv", dataset);
	arma::mat trainingset = dataset.rows(0, 1); // x and y
	arma::rowvec responses = dataset.row(2); // count
	stage_model_2d.Train(trainingset, responses);

	mat queryset;
	switch (query_region) {
	case 1:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection100m.csv", queryset);
		break;
	case 2:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection200m.csv", queryset);
		break;
	case 3:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection500m.csv", queryset);
		break;
	case 4:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection1km.csv", queryset);
		break;
	case 5:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection2km.csv", queryset);
		break;
	case 6:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection5km.csv", queryset);
		break;
	case 7:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection10km.csv", queryset);
		break;
	case 8:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection20km.csv", queryset);
		break;
	case 9:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection50km.csv", queryset);
		break;
	case 10:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection100km.csv", queryset);
		break;
	default:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimQuery2.csv", queryset);
		break;
	}
	mat queryset_x_low = queryset.row(0);
	mat queryset_x_up = queryset.row(1);
	mat queryset_y_low = queryset.row(2);
	mat queryset_y_up = queryset.row(3);
	vector<double> query_x_low_v, query_x_up_v, query_y_low_v, query_y_up_v;
	RowvecToVector(queryset_x_low, query_x_low_v);
	RowvecToVector(queryset_x_up, query_x_up_v);
	RowvecToVector(queryset_y_low, query_y_low_v);
	RowvecToVector(queryset_y_up, query_y_up_v);

	auto t0 = chrono::steady_clock::now();

	//double count, count_full, count_min, count_half_1, count_half_2;
	vector<double> count_upper_right, count_lower_left, count_upper_left, count_lower_right;
	stage_model_2d.PredictVector(query_x_up_v, query_y_up_v, count_upper_right);
	stage_model_2d.PredictVector(query_x_low_v, query_y_low_v, count_lower_left);
	stage_model_2d.PredictVector(query_x_low_v, query_y_up_v, count_upper_left);
	stage_model_2d.PredictVector(query_x_up_v, query_y_low_v, count_lower_right);

	vector<int> predicted_results;
	double count = 0;
	for (int i = 0; i < count_upper_right.size(); i++) {
		count = count_upper_right[i] - count_upper_left[i] - count_lower_right[i] + count_lower_left[i];
		predicted_results.push_back(count);
	}

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "finish stage model prediction.." << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_LEARN.csv");
	for (int i = 0; i < predicted_results.size(); i++) {
		outfile << predicted_results[i] << endl;
	}
	outfile.close();

	// measure accuracy
	vector<int> real_results;
	CalculateRealCountWithScan2D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results);
}

void Count2DLearnedIndex(int query_region, unsigned max_bits, unsigned using_bits = 0) {

	if (using_bits == 0) {
		using_bits = max_bits;
	}

	vector<int> arch;
	arch.push_back(1);
	arch.push_back(10);
	arch.push_back(100);
	arch.push_back(1000);

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_22.csv", dataset);
	arma::rowvec trainingset = dataset.row(4); // hilbert value
	arma::rowvec responses = dataset.row(5); // order
	StageModel stage_model(trainingset, responses, arch);
	stage_model.DumpParameters();

	mat queryset;
	switch (query_region) {
	case 1:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection100m.csv", queryset);
		break;
	case 2:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection200m.csv", queryset);
		break;
	case 3:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection500m.csv", queryset);
		break;
	case 4:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection1km.csv", queryset);
		break;
	case 5:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection2km.csv", queryset);
		break;
	case 6:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection5km.csv", queryset);
		break;
	case 7:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection10km.csv", queryset);
		break;
	case 8:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection20km.csv", queryset);
		break;
	case 9:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection50km.csv", queryset);
		break;
	case 10:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/RangeQueryCollection100km.csv", queryset);
		break;
	default:
		mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimQuery2.csv", queryset);
		break;
	}

	mat queryset_x_low = queryset.row(0);
	mat queryset_x_up = queryset.row(1);
	mat queryset_y_low = queryset.row(2);
	mat queryset_y_up = queryset.row(3);
	vector<double> query_x_low_v, query_x_up_v, query_y_low_v, query_y_up_v;
	StageModel::RowvecToVector(queryset_x_low, query_x_low_v);
	StageModel::RowvecToVector(queryset_x_up, query_x_up_v);
	StageModel::RowvecToVector(queryset_y_low, query_y_low_v);
	StageModel::RowvecToVector(queryset_y_up, query_y_up_v);

	int FULL_BITS = 22; // that's for our dataset
	int differ_bits = FULL_BITS - max_bits;
	int lower_x, upper_x, lower_y, upper_y;
	vector<interval> intervals;
	int count;
	vector<int> counts;
	long intervals_count = 0;
	vector<double> lower_hilbert, upper_hilbert;
	vector<double> results_low, results_up;

	// query
	auto t0 = chrono::steady_clock::now();
	for (int i = 0; i < query_x_low_v.size(); i++) {
		// generate Hilbert value
		lower_x = (int)((2 * query_x_low_v[i] + 180) * 10000) >> differ_bits;
		upper_x = (int)((2 * query_x_up_v[i] + 180) * 10000) >> differ_bits;
		lower_y = (int)((query_y_low_v[i] + 180) * 10000) >> differ_bits;
		upper_y = (int)((query_y_up_v[i] + 180) * 10000) >> differ_bits;
		intervals.clear();
		GetIntervals(max_bits, using_bits, lower_x, upper_x, lower_y, upper_y, intervals);

		// extract intervals lower and upper hilbert value
		lower_hilbert.clear();
		upper_hilbert.clear();
		for (int j = 0; j < intervals.size(); j++) {
			lower_hilbert.push_back(intervals[j].lower_hilbert_value);
			upper_hilbert.push_back(intervals[j].upper_hilbert_value);
		}

		stage_model.PredictVector(lower_hilbert, results_low);
		stage_model.PredictVector(upper_hilbert, results_up);

		// calculate the count
		count = 0;
		double temp;
		for (int j = 0; j < results_low.size(); j++) {
			temp = results_up[j] - results_low[j];
			if(temp > 0)
				count += temp;
		}
		
		// save results
		counts.push_back(count);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "finish stage model prediction.." << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_LEARN.csv");
	for (int i = 0; i < counts.size(); i++) {
		outfile << counts[i] << endl;
	}
	outfile.close();

	// measure accuracy
	vector<int> real_results;
	CalculateRealCountWithScan2D(dataset, queryset, real_results);
	double relative_error;
	double accu = 0;
	double accu_absolute = 0;
	int total_size = counts.size();
	for (int i = 0; i < counts.size(); i++) {
		if (real_results[i] == 0) {
			total_size--;
			continue;
		}
		relative_error = abs(double(counts[i] - real_results[i]) / real_results[i]);
		accu += relative_error;
		accu_absolute += abs(counts[i] - real_results[i]);
		//cout << "relative error " << i << ": " << relative_error << endl;
	}
	double avg_rel_err = accu / total_size;
	cout << "average relative error: " << avg_rel_err << endl;
	cout << "average absolute error: " << accu_absolute / total_size << endl;
}

void SumShiftLearnedIndex() {
	vector<int> arch;
	arch.push_back(1);
	arch.push_back(10);
	arch.push_back(100);
	arch.push_back(1000);

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimSUM_SHIFT2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0); // x
	arma::rowvec responses = dataset.row(4); // sum_shift
	arma::rowvec responses_order = dataset.row(5); // order
	//double total_sum = responses[responses.n_cols - 1];
	//cout << "total_sum" << total_sum << endl;
	StageModel stage_model(trainingset, responses, arch, 1); // need to set TOTAL_SIZE to total_sum
	StageModel stage_model_order(trainingset, responses_order, arch, 0); // need to set TOTAL_SIZE to total_size
	stage_model.DumpParameters();
	stage_model_order.DumpParameters();

	mat queryset;
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimSUM2.csv", queryset);
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);

	arma::rowvec predictions_low, predictions_up;
	vector<double> queryset_low_v, queryset_up_v, results_low, results_up, results_low_order, results_up_order;
	vector<double> da;
	vector<int> re;
	StageModel::RowvecToVector(dataset, da);
	StageModel::RowvecToVector(queryset_low, queryset_low_v);
	StageModel::RowvecToVector(queryset_up, queryset_up_v);

	auto t0 = chrono::steady_clock::now();
	stage_model.PredictVector(queryset_low_v, results_low);
	stage_model.PredictVector(queryset_up_v, results_up);
	stage_model_order.PredictVector(queryset_low_v, results_low_order);
	stage_model_order.PredictVector(queryset_up_v, results_up_order);
	for (int i = 0; i < results_up.size(); i++) {
		results_up[i] = results_up[i] - results_low[i] - 180*(results_up_order[i]- results_low_order[i]);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "finish stage model prediction.." << endl;

	//// save the predicted result
	//ofstream outfile;
	//outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSUM_SHIFT_DATASET_PREDICATED.csv");
	//for (int i = 0; i < results_low.size(); i++) {
	//	outfile << queryset_low_v[i] << "," << results_low[i] << endl;
	//}
	//outfile.close();

	mat real_results;
	bool loaded3 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSUM.csv", real_results);
	arma::rowvec real = real_results.row(0);
	vector<double> real_v;
	StageModel::RowvecToVector(real, real_v);
	stage_model.MeasureAccuracyWithVector(results_up, real_v);
}

void SumLearnedIndex() {
	vector<int> arch;
	arch.push_back(1);
	arch.push_back(10);
	arch.push_back(100);
	arch.push_back(1000);

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimSUM2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0); // x
	arma::rowvec responses = dataset.row(2); // sum
	//double total_sum = responses[responses.n_cols - 1];
	//cout << "total_sum" << total_sum << endl;
	StageModel stage_model(trainingset, responses, arch, 1); // need to set TOTAL_SIZE to total_sum
	stage_model.DumpParameters();

	mat queryset;
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimSUM2.csv", queryset);
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);

	arma::rowvec predictions_low, predictions_up;
	vector<double> queryset_low_v, queryset_up_v, results_low, results_up;
	vector<double> da;
	vector<int> re;
	StageModel::RowvecToVector(dataset, da);
	StageModel::RowvecToVector(queryset_low, queryset_low_v);
	StageModel::RowvecToVector(queryset_up, queryset_up_v);

	auto t0 = chrono::steady_clock::now();
	stage_model.PredictVector(queryset_low_v, results_low);
	stage_model.PredictVector(queryset_up_v, results_up);
	for (int i = 0; i < results_up.size(); i++) {
		results_up[i] -= results_low[i];
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "finish stage model prediction.." << endl;

	//// save the predicted result
	//ofstream outfile;
	//outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSUM_DATASET_PREDICATED.csv");
	//for (int i = 0; i < results_low.size(); i++) {
	//	outfile << queryset_low_v[i] << "," << results_low[i] << endl;
	//}
	//outfile.close();
	
	mat real_results;
	bool loaded3 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSUM.csv", real_results);
	arma::rowvec real = real_results.row(0);
	vector<double> real_v;
	StageModel::RowvecToVector(real, real_v);
	stage_model.MeasureAccuracyWithVector(results_up, real_v);
}

// a relative naive method, do not reconstruct the structure
void SumStxBtree() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimSUM2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0); // x
	arma::rowvec responses = dataset.row(2); // sum
	vector<double> train_v, response_v;
	StageModel::RowvecToVector(trainingset, train_v);
	StageModel::RowvecToVector(responses, response_v);

	stx::btree<double, double> btree;
	for (int i = 0; i < train_v.size(); i++) {
		btree.insert(pair<double, double>(train_v[i], response_v[i]));
	}

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);
	vector<double> query_low_v, query_up_v;
	StageModel::RowvecToVector(queryset_low, query_low_v);
	StageModel::RowvecToVector(queryset_up, query_up_v);

	stx::btree<double, double>::iterator iter_low, iter_up;
	double sum = 0;
	vector<double> results;
	auto t0 = chrono::steady_clock::now(); // am I doing the correct thing???????????????
	for (int i = 0; i < query_low_v.size(); i++) {
		iter_low = btree.lower_bound(query_low_v[i]); // include the key
		iter_up = btree.lower_bound(query_up_v[i]); // include the key
		//cout << iter_low->first << " " << iter_up->first << endl;
		if (iter_low!= btree.begin()) {
			iter_low--;
		}
		//cout << iter_low->first << endl;
		sum = iter_up->second - iter_low->second;
		results.push_back(sum);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;

	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSUM.csv");
	for (int i = 0; i < results.size(); i++) {
		outfile << results[i] << endl;
	}
	outfile.close();
}

void CountLearnedIndex() {
	//TestBtreeAggregate("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv");

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	dataset.shed_rows(1, 2);

	vector<int> arch;
	arch.push_back(1);
	//arch.push_back(10);
	arch.push_back(100);
	//arch.push_back(500);
	//arch.push_back(1890);
	StageModel stage_model(dataset, responses, arch, 0, 5000); // last: error_threshold

	mat queryset;
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", queryset);
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	//bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);

	arma::rowvec predictions_low, predictions_up;
	vector<double> queryset_low_v, queryset_up_v, results_low, results_up;

	stage_model.DumpParameters();
	//stage_model.DumpParametersWithNN();

	vector<double> da;
	vector<int> re;
	StageModel::RowvecToVector(dataset, da);
	//stage_model.RecordBucket(da, re, "C:/Users/Cloud/Desktop/LearnIndex/data/BucketRecord.csv");

	StageModel::RowvecToVector(queryset_low, queryset_low_v);
	StageModel::RowvecToVector(queryset_up, queryset_up_v);

	//stage_model.TrainBucketAssigner("C:/Users/Cloud/Desktop/LearnIndex/data/BucketRecord.csv");

	auto t0 = chrono::steady_clock::now();

	//stage_model.PredictVectorWithBucketMap(queryset_low_v, results_low);
	//stage_model.PredictVectorWithBucketMap(queryset_up_v, results_up);
	//stage_model.PredictVectorWithBucketAssigner(queryset_low_v, results_low);
	//stage_model.PredictVectorWithBucketAssigner(queryset_up_v, results_up);

	//stage_model.PredictVector(queryset_low_v, results_low);
	//stage_model.PredictVector(queryset_up_v, results_up);

	stage_model.PredictVectorWithErrorThreshold(queryset_low_v, results_low);
	stage_model.PredictVectorWithErrorThreshold(queryset_up_v, results_up);

	//stage_model.PredictVectorWithNN(queryset_low_v, results_low);
	//stage_model.PredictVectorWithNN(queryset_up_v, results_up);

	for (int i = 0; i < results_up.size(); i++) {
		results_up[i] -= results_low[i];
	}

	/*StageModel::PredictNaiveSingleLR2(dataset, responses, queryset_low, predictions_low);
	StageModel::PredictNaiveSingleLR2(dataset, responses, queryset_up, predictions_up);*/

	//StageModel::PredictNaiveSingleLR(dataset, responses, queryset_low, predictions_low);
	//StageModel::PredictNaiveSingleLR(dataset, responses, queryset_up, predictions_up);

	//stage_model.InitQuerySet(queryset_low);
	//stage_model.Predict(queryset_low, predictions_low);
	//stage_model.InitQuerySet(queryset_up);   
	//stage_model.Predict(queryset_up, predictions_up);
	//predictions_up -= predictions_low;

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "replaced model count: " << stage_model.replaced_btree.size() << endl;

	//// save the predicted result
	//ofstream outfile;
	//outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsCOUNT_DATASET_PREDICATED.csv");
	//for (int i = 0; i < results_low.size(); i++) {
	//	outfile << queryset_low_v[i] << "," << results_low[i] << endl;
	//}
	//outfile.close();

	StageModel::MeasureAccuracyWithVector(results_up);
	//StageModel::VectorToRowvec(predictions_up, results_up);
	//tage_model.MeasureAccuracy(predictions_up);
}