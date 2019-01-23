#pragma once
#include "LearnIndexAggregate.h"
#include "STXBtreeAggregate.h"
#include "StageModel.h"
#include "Hilbert.h"

void CountStx2D(unsigned max_bits, unsigned using_bits = 0) {
	mat dataset;
	if (max_bits == 22) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_22.csv", dataset);
	}
	else if (max_bits == 16) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_16.csv", dataset);
		//dataset.load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_16.csv", csv_ascii);
	}
	else if (max_bits == 14) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2.csv", dataset);
	}
	else if (max_bits == 13) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_13.csv", dataset);
	}
	else if (max_bits == 12) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_12.csv", dataset);
	}
	else if (max_bits == 10) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_10.csv", dataset);
	}

	if (using_bits == 0) {
		using_bits = max_bits;
	}

	arma::rowvec trainingset = dataset.row(4); // hilbert value
	arma::rowvec responses = dataset.row(5); // order

	vector<double> train_v, response_v;
	StageModel::RowvecToVector(trainingset, train_v);
	StageModel::RowvecToVector(responses, response_v);

	stx::btree<double, double> btree;
	for (int i = 0; i < train_v.size(); i++) {
		btree.insert(pair<double, double>(train_v[i], response_v[i]));
	}

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimQuery2.csv", queryset);
	mat queryset_x_low = queryset.row(0);
	mat queryset_x_up = queryset.row(1);
	mat queryset_y_low = queryset.row(2);
	mat queryset_y_up = queryset.row(3);
	vector<double> query_x_low_v, query_x_up_v, query_y_low_v, query_y_up_v;
	StageModel::RowvecToVector(queryset_x_low, query_x_low_v);
	StageModel::RowvecToVector(queryset_x_up, query_x_up_v);
	StageModel::RowvecToVector(queryset_y_low, query_y_low_v);
	StageModel::RowvecToVector(queryset_y_up, query_y_up_v);

	// calculate the begin and end of hilbert value
	int FULL_BITS = 22; // that's for our dataset
	int differ_bits = FULL_BITS - max_bits;
	int lower_x, upper_x, lower_y, upper_y;
	vector<interval> intervals;
	stx::btree<double, double>::iterator iter_low, iter_up;
	int count;
	vector<int> counts;
	long intervals_count = 0;
	auto t0 = chrono::steady_clock::now();
	vector<long> intervals_count_v;
	int abnormal = 0;

	for (int i = 0; i < query_x_low_v.size(); i++) {
		lower_x = (int)((2 * query_x_low_v[i] + 180) * 10000) >> differ_bits;
		upper_x = (int)((2 * query_x_up_v[i] + 180) * 10000) >> differ_bits;
		lower_y = (int)((query_y_low_v[i] + 180) * 10000) >> differ_bits;
		upper_y = (int)((query_y_up_v[i] + 180) * 10000) >> differ_bits;
		intervals.clear();
		GetIntervals(max_bits, using_bits, lower_x, upper_x, lower_y, upper_y, intervals);
		intervals_count_v.push_back(intervals.size());
		count = 0;
		int upper=0, lower=0;
		int MAX = responses.n_cols;
		for (int j = 0; j < intervals.size(); j++) { // why the 10th(i=9) is negative?
			iter_low = btree.lower_bound(intervals[j].lower_hilbert_value);
			iter_up = btree.upper_bound(intervals[j].upper_hilbert_value);
			//count += iter_up->second - iter_low->second + 1;
			//cout << j << " " <<iter_up->second << " " << iter_low->second << " " << iter_up->second - iter_low->second << endl;
			if (iter_up == iter_low) { // there are not records within this range
				continue; // ignore this value, ie., count should be 0
			}
			else {
				if (iter_up == btree.end()) {
					upper = MAX - 1;
				}
				else if (iter_up == btree.begin()) {
					upper = 0;
				}
				else {
					upper = iter_up->second;
				}

				if (iter_low == btree.end()) {
					lower = MAX - 1;
				} 
				else if (iter_low == btree.begin()) {
					lower = 0;
				}
				else {
					lower = iter_low->second;
				}
			}
			count += upper - lower;

			if (intervals[j].lower_hilbert_value > intervals[j].upper_hilbert_value) {
				cout << "greater than!!! " << i << " " << j << " " << lower_x << " " << upper_x << " " << lower_y << " " << upper_y << endl;
				abnormal++;
			}
			/*if (count < 0) {
				cout << "here " << upper << " " << lower << endl;
				cout << lower_x << " " << upper_x << " " << lower_y << " " << upper_y << endl;
				cout << j << " " << iter_up->second << " " << iter_low->second << " " << iter_up->second - iter_low->second << endl;
				if (iter_up == btree.end())
					cout << "yes!" << endl;
			}*/
		}
		intervals_count += intervals.size();
		counts.push_back(count);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "average number of intervals: " << intervals_count / (queryset.size() / queryset.n_rows) << endl;
	cout << "number of nodes: " << btree.CountNodesPrimary() << endl;

	cout << "abnormal: " << abnormal << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults.csv");
	for (int i = 0; i < counts.size(); i++) {
		outfile << counts[i] << endl;
	}
	outfile.close();

	//// record intervals count to file
	//ofstream outfile2;
	//outfile2.open("C:/Users/Cloud/Desktop/LearnIndex/data/intervals_10.csv");
	//for (int i = 0; i < counts.size(); i++) {
	//	outfile2 << intervals_count_v[i] << endl;
	//}
	//outfile2.close();

	// compare the correctness
	vector<int> real_results;
	CalculateRealCountWithScan2D(dataset, queryset, real_results);
	// todo: change to read real results from file
	/*mat resultset;
	bool loaded3 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_REAL.csv", dataset);
	StageModel::RowvecToVector(resultset, real_results);*/

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

void CountStx2DSmallRegion(int query_region, unsigned max_bits, unsigned using_bits = 0) {
	mat dataset;
	if (max_bits == 22) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_22.csv", dataset);
	}
	else if (max_bits == 16) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_16.csv", dataset);
	}
	else if (max_bits == 14) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2.csv", dataset);
	}
	else if (max_bits == 13) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_13.csv", dataset);
	}
	else if (max_bits == 12) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_12.csv", dataset);
	}
	else if (max_bits == 10) {
		bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2_10.csv", dataset);
	}

	if (using_bits == 0) {
		using_bits = max_bits;
	}

	arma::rowvec trainingset = dataset.row(4); // hilbert value
	arma::rowvec responses = dataset.row(5); // order

	vector<double> train_v, response_v;
	StageModel::RowvecToVector(trainingset, train_v);
	StageModel::RowvecToVector(responses, response_v);

	stx::btree<double, double> btree;
	for (int i = 0; i < train_v.size(); i++) {
		btree.insert(pair<double, double>(train_v[i], response_v[i]));
	}

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

	// calculate the begin and end of hilbert value
	int FULL_BITS = 22; // that's for our dataset
	int differ_bits = FULL_BITS - max_bits;
	int lower_x, upper_x, lower_y, upper_y;
	vector<interval> intervals;
	stx::btree<double, double>::iterator iter_low, iter_up;
	int count;
	vector<int> counts;
	long intervals_count = 0;
	auto t0 = chrono::steady_clock::now();
	vector<long> intervals_count_v;
	int abnormal = 0;

	for (int i = 0; i < query_x_low_v.size(); i++) {
		lower_x = (int)((2 * query_x_low_v[i] + 180) * 10000) >> differ_bits;
		upper_x = (int)((2 * query_x_up_v[i] + 180) * 10000) >> differ_bits;
		lower_y = (int)((query_y_low_v[i] + 180) * 10000) >> differ_bits;
		upper_y = (int)((query_y_up_v[i] + 180) * 10000) >> differ_bits;
		intervals.clear();
		GetIntervals(max_bits, using_bits, lower_x, upper_x, lower_y, upper_y, intervals);
		//GetIntervals(2, 0, 1, 2, 1, 2, intervals);
		intervals_count_v.push_back(intervals.size());
		count = 0;
		int upper = 0, lower = 0;
		int MAX = responses.n_cols;
		for (int j = 0; j < intervals.size(); j++) { // why the 10th(i=9) is negative?
			iter_low = btree.lower_bound(intervals[j].lower_hilbert_value);
			iter_up = btree.upper_bound(intervals[j].upper_hilbert_value);
			//count += iter_up->second - iter_low->second + 1;
			//cout << j << " " <<iter_up->second << " " << iter_low->second << " " << iter_up->second - iter_low->second << endl;
			if (iter_up == iter_low) { // there are not records within this range
				continue; // ignore this value, ie., count should be 0
			}
			else {
				if (iter_up == btree.end()) {
					upper = MAX - 1;
				}
				else if (iter_up == btree.begin()) {
					upper = 0;
				}
				else {
					upper = iter_up->second;
				}

				if (iter_low == btree.end()) {
					lower = MAX - 1;
				}
				else if (iter_low == btree.begin()) {
					lower = 0;
				}
				else {
					lower = iter_low->second;
				}
			}
			count += upper - lower;

			if (intervals[j].lower_hilbert_value > intervals[j].upper_hilbert_value) {
				cout << "greater than!!! " << i << " " << j << " " << lower_x << " " << upper_x << " " << lower_y << " " << upper_y << endl;
				abnormal++;
			}
			/*if (count < 0) {
				cout << "here " << upper << " " << lower << endl;
				cout << lower_x << " " << upper_x << " " << lower_y << " " << upper_y << endl;
				cout << j << " " << iter_up->second << " " << iter_low->second << " " << iter_up->second - iter_low->second << endl;
				if (iter_up == btree.end())
					cout << "yes!" << endl;
			}*/
		}
		intervals_count += intervals.size();
		counts.push_back(count);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "average number of intervals: " << intervals_count / (queryset.size() / queryset.n_rows) << endl;
	cout << "number of nodes: " << btree.CountNodesPrimary() << endl;

	cout << "abnormal: " << abnormal << endl;

	// save results to file
	ofstream outfile;
	outfile.open("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults.csv");
	for (int i = 0; i < counts.size(); i++) {
		outfile << counts[i] << endl;
	}
	outfile.close();

	//// record intervals count to file
	//ofstream outfile2;
	//outfile2.open("C:/Users/Cloud/Desktop/LearnIndex/data/intervals_10.csv");
	//for (int i = 0; i < counts.size(); i++) {
	//	outfile2 << intervals_count_v[i] << endl;
	//}
	//outfile2.close();

	// compare the correctness
	vector<int> real_results;
	CalculateRealCountWithScan2D(dataset, queryset, real_results);
	// todo: change to read real results from file
	/*mat resultset;
	bool loaded3 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimResults_REAL.csv", dataset);
	StageModel::RowvecToVector(resultset, real_results);*/

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