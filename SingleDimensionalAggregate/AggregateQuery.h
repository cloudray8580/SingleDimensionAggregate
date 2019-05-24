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

using namespace std;

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

void TestX4() {
	ROLLearnedIndex_quartic learnedindex(9, 1000, 100); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x4.csv", dataset);

	arma::mat trainingset = dataset.rows(0, 3);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(dataset.row(3), key_v);
	RowvecToVector(responses, position_v);

	double a, b, c, d, e, current_absolute_accuracy;

	learnedindex.ApproximateMaxLossLinearRegression(0, key_v.size() - 1, key_v, position_v, trainingset, responses, 2, 100, 0.1, a, b, c, d, e, current_absolute_accuracy);
}

void TestX3_Complete() {
	ROLLearnedIndex_cubic learnedindex(9, 1000, 100000); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x3.csv", dataset);

	arma::mat trainingset = dataset.rows(0, 2);
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

void TestX3() {
	ROLLearnedIndex_cubic learnedindex(9, 1000, 100); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x3.csv", dataset);

	arma::mat trainingset = dataset.rows(0, 2);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(dataset.row(2), key_v);
	RowvecToVector(responses, position_v);

	double a, b, c, d, current_absolute_accuracy;

	learnedindex.ApproximateMaxLossLinearRegression(0, key_v.size()-1, key_v, position_v, trainingset, responses, 2, 100, 0.1, a, b, c, d, current_absolute_accuracy);
}

void TestX2() {
	ROLLearnedIndex_quadratic learnedindex(9, 1000, 100); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x2.csv", dataset);

	arma::mat trainingset = dataset.rows(0, 1);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(dataset.row(1), key_v);
	RowvecToVector(responses, position_v);

	double a, b, c, current_absolute_accuracy;

	learnedindex.ApproximateMaxLossLinearRegression(0, key_v.size() - 1, key_v, position_v, trainingset, responses, 2, 100, 0.1, a, b, c, current_absolute_accuracy);
}

void TestROLQuadratic() {
	ROLLearnedIndex_quadratic learnedindex_quadratic(9, 1000, 100000); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/SortedSingleDimPOIs2_x2.csv", dataset);

	arma::mat trainingset = dataset.rows(0,1);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	learnedindex_quadratic.TrainBottomLayerWithFastDetectForMaxLoss(trainingset, responses, 1000000, 0.5, 0.1);
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
	StageModelBottomUp stage_model_bottom_up(9, 1000, 100000); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_30M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv", dataset);

	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);


	stage_model_bottom_up.TrainBottomLayerWithFastDetectForMaxLoss(trainingset, responses,1000, 2, 0.1);// do not need to dump parameter
	stage_model_bottom_up.BuildNonLeafLayerWithBtree(); // btree


	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);

	arma::rowvec query_x_low = queryset.row(0);
	arma::rowvec query_x_up = queryset.row(1);

	vector<double> queryset_x_up_v, queryset_x_low_v;
	RowvecToVector(query_x_up, queryset_x_up_v);
	RowvecToVector(query_x_low, queryset_x_low_v);

	vector<int> predicted_results_x_up, predicted_results_x_low, predicted_results, real_results;

	auto t0 = chrono::steady_clock::now();

	stage_model_bottom_up.PredictWithStxBtree(queryset_x_up_v, predicted_results_x_up);
	stage_model_bottom_up.PredictWithStxBtree(queryset_x_low_v, predicted_results_x_low);

	for (int i = 0; i < predicted_results_x_up.size(); i++) {
		predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);
	}

	auto t1 = chrono::steady_clock::now();
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

	//CalculateRealCountWithScan1D(queryset, real_results, "C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv");
	CalculateRealCountWithScan1D(queryset, real_results);
	MeasureAccuracy(predicted_results, real_results); // need to adjust the error threshold by hand!!!!
}

void TestMaxLoss() {
	StageModelBottomUp stage_model_bottom_up(9, 1000, 100); // level, step, error

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	vector<double> key_v, position_v;
	RowvecToVector(trainingset, key_v);
	RowvecToVector(responses, position_v);

	double a, b, c;

	stage_model_bottom_up.ApproximateMaxLossLinearRegression(0, key_v.size()-1, key_v, position_v, trainingset, responses, 2, 1000, 0.085, a,b,c);
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

	Douglas douglas(900);

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	douglas.BuildDouglas(trainingset, responses);

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

void TestAtree() {
	ATree atree(100); // error threshold

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_10M_Sorted_Value.csv", dataset);
	//bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnedAggregateData/MapData_100M_Sorted_Value.csv", dataset);
	arma::rowvec trainingset = dataset.row(0);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);

	atree.TrainAtree(trainingset, responses);
	
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

	//atree.Predict(queryset_x_up_v, predicted_results_x_up);
	//atree.Predict(queryset_x_low_v, predicted_results_x_low);
	atree.PredictExact(queryset_x_up_v, predicted_results_x_up);
	atree.PredictExact(queryset_x_low_v, predicted_results_x_low);

	for (int i = 0; i < predicted_results_x_up.size(); i++) {
		predicted_results.push_back(predicted_results_x_up[i] - predicted_results_x_low[i]);
	}

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
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