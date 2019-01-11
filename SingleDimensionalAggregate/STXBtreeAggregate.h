#pragma once
#include "pch.h"
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <map>
#include <iomanip>
#include <stx/btree.h> 
#include <chrono>
#include "mlpack/core.hpp"

using namespace std;

typedef vector<pair<double, vector<double>>> MyDataset;

void LoadDataset(string filename, MyDataset & dataset) {
	ifstream infile;
	infile.open(filename); // latitude, longitude, order
	
	vector<double> onerow;

	string latitude_s;
	string longitude_s;
	string order_s;

	dataset.clear();
	while (getline(infile, latitude_s, ',')) {
		onerow.clear();

		onerow.push_back(stod(latitude_s));

		getline(infile, longitude_s, ',');
		onerow.push_back(stod(longitude_s));

		getline(infile, order_s, '\n'); //last one
		onerow.push_back(stol(order_s));

		dataset.push_back(pair<double, vector<double>>(stod(latitude_s), onerow));
	}
}

void LoadQueryset(string filename, vector<pair<double, double>> & queryset) {
	ifstream infile;
	infile.open(filename); // lower spot and upper spot on latitude
	string lower_s;
	string upper_s;

	queryset.clear();
	while (getline(infile, lower_s, ',')) {
		getline(infile, upper_s, '\n'); //last one
		queryset.push_back(pair<double, double>(stod(lower_s), stod(upper_s)));
	}
}

inline void CalculateCount(double lowerbound, double upperbound, stx::btree<double, vector<double>> &btree, int &count) {
	//auto t0 = chrono::steady_clock::now();
	
	stx::btree<double, vector<double>>::iterator iter_low, iter_up;
	iter_low = btree.lower_bound(lowerbound); // include the key
	iter_up = btree.upper_bound(upperbound); // not include the key
	if (iter_low != btree.end() && iter_up != btree.end()) {
		count = iter_up->second[2] - iter_low->second[2];
	/*	cout << lowerbound << " " << upperbound << endl;
		cout << iter_low->first << " " << iter_up->first << endl;
		cout << iter_low->second[2] << " " << iter_up->second[2] << endl;
		cout << count << endl;
		for (; iter_low != btree.end(); iter_low++) {
			cout << iter_low->first << " " << iter_low->second[2] << endl;
		}*/
	}
	else if (iter_low != btree.end()) {
		count = btree.size() - iter_low->second[2];
	}
	else {
		count = 0;
	}

	//auto t1 = chrono::steady_clock::now();
	//cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
}

void SaveResutToFile(string filename, vector<double> &results) {
	ofstream outfile;
	outfile.open(filename);
	for (int i = 0; i < results.size(); i++) {
		outfile << results[i] << "\n";
	}
	outfile.close();
}

void TestBtreeAggregate(string filename_query = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", string filename_data="C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", string filename_result= "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResults.csv") {
	// read dataset and !!!handle the order!!! 
	MyDataset dataset;
	LoadDataset(filename_data, dataset);
	cout << "reading dataset finished.." << endl;

	// load dataset into stx-tree
	stx::btree<double, vector<double>> btree;
	//btree.bulk_load(dataset.begin(), dataset.end()); // this will not maintain the order
	for (int i = 0; i < dataset.size(); i++) {
		btree.insert(dataset[i]);
	}
	
	cout << "bulk loading stx-btree finished.." << endl;

	// read and load query set
	vector<pair<double, double>> queryset;
	LoadQueryset(filename_query ,queryset);

	// execute and measure aggregate query
	auto t0 = chrono::steady_clock::now();
	int count;
	vector<double> results;
	for (int i = 0; i < queryset.size(); i++) {
		CalculateCount(queryset[i].first, queryset[i].second, btree, count);
		results.push_back(count);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count()/queryset.size() << " ns" << endl;

	// execute again to save the correct result
	/*vector<double> results;
	for (int i = 0; i < queryset.size(); i++) {
		CalculateCount(queryset[i].first, queryset[i].second, btree, count);
		results.push_back(count);
	}*/
	SaveResutToFile(filename_result, results);

	/*ofstream outfile;
	outfile.open(filename_result);
	btree.dump(outfile);*/
	cout << "number of nodes: " << btree.CountNodesPrimary() << endl;
	//cout << "size of the Btree: " << sizeof(btree) << endl; //56
}

// ascending order
bool cmp_dataset(pair<double, vector<double>> & record1, pair<double, vector<double>> & record2) {
	return record1.first < record2.first;
}

// using sampling of dataset
void TestBtreeAggregateVariant1(double sample_percentage=0.1, string filename_query = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", string filename_data = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", string filename_result = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResultsSample.csv"){
	// read dataset
	MyDataset dataset;
	LoadDataset(filename_data, dataset);
	cout << "reading dataset finished.." << endl;

	// do sampling
	int sample_total = dataset.size() * sample_percentage;
	MyDataset dataset_sample;
	srand((unsigned)time(NULL));
	// use random shuffle !!!
	std::random_shuffle(dataset.begin(), dataset.end());
	for (int i = 0; i < sample_total; i++) {
		dataset_sample.push_back(dataset[i]);
	}
	sort(dataset_sample.begin(), dataset_sample.end(), cmp_dataset);

	// load dataset into stx-tree
	stx::btree<double, vector<double>> btree;
	//btree.bulk_load(dataset_sample.begin(), dataset_sample.end()); // this will not maintain the order
	for (int i = 0; i < dataset_sample.size(); i++) {
		dataset_sample[i].second[2] = i;
		btree.insert(dataset_sample[i]);
	}
	cout << "bulk loading stx-btree finished.." << endl;
	cout << "stx-btree size: " << btree.size() << endl;

	// read and load query set
	vector<pair<double, double>> queryset;
	LoadQueryset(filename_query, queryset);

	// execute and measure aggregate query
	auto t0 = chrono::steady_clock::now();
	int count;
	vector<double> results;
	for (int i = 0; i < queryset.size(); i++) {
		CalculateCount(queryset[i].first, queryset[i].second, btree, count);
		count /= sample_percentage;
		results.push_back(count); // test the time spent on this
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / queryset.size() << " ns" << endl;
	
	SaveResutToFile(filename_result, results);

	arma::mat real_result, predicted_result;
	bool loaded3 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimResults.csv", real_result);
	bool loaded4 = mlpack::data::Load(filename_result, predicted_result);
	arma::rowvec real_range = real_result.row(0);
	arma::rowvec predicted_range = predicted_result.row(0);
	arma::rowvec relative_error = abs(predicted_range - real_range);
	relative_error /= real_range;
	double total_error = arma::accu(relative_error);
	double average_relative_error = total_error / relative_error.size();
	cout << "average error: " << average_relative_error << endl;

	cout << "number of nodes: " << btree.CountNodesPrimary() << endl;
	//// return average_relative_error;
}

void SingleLookUpTest(string filename_data = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv") {
	MyDataset dataset, resultset;
	LoadDataset(filename_data, dataset);
	cout << "reading dataset finished.." << endl;
	stx::btree<double, vector<double>> btree;
	btree.bulk_load(dataset.begin(), dataset.end());
	cout << "bulk loading stx-btree finished.." << endl;

	stx::btree<double, vector<double>>::iterator iter;
	vector<double> results;
	auto t0 = chrono::steady_clock::now();
	for (int i = 0; i < dataset.size()-1; i++) {
		//iter = btree.lower_bound(dataset[i].first);
		//iter = btree.upper_bound(dataset[i].first);
		/*iter = btree.find(dataset[i].first);
		if (iter == btree.end()) {
			continue;
		}*/
		//resultset.push_back(*iter);

		int count;
		stx::btree<double, vector<double>>::iterator iter_low, iter_up;
		iter_low = btree.lower_bound(dataset[i].first); // include the key
		iter_up = btree.upper_bound(dataset[i+1].first); // not include the key
		if (iter_low != btree.end() && iter_up != btree.end()) {
			count = iter_up->second[2] - iter_low->second[2];
		}
		else if (iter_low != btree.end()) {
			count = btree.size() - iter_low->second[2];
		}
		else {
			count = 0;  
		}
		results.push_back(count);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / dataset.size() << " ns" << endl;
}

void SingleLookUpTest2(string filename_query = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", string filename_data = "C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv") {
	MyDataset dataset, resultset;
	LoadDataset(filename_data, dataset);
	cout << "reading dataset finished.." << endl;
	stx::btree<double, vector<double>> btree;
	btree.bulk_load(dataset.begin(), dataset.end());
	cout << "bulk loading stx-btree finished.." << endl;

	stx::btree<double, vector<double>>::iterator iter;
	vector<pair<double, double>> queryset;
	LoadQueryset(filename_query, queryset);
	vector<double> results;
	auto t0 = chrono::steady_clock::now();
	for (int i = 0; i < queryset.size(); i++) {
		//iter = btree.lower_bound(queryset[i].first);
		//iter = btree.upper_bound(queryset[i].first);
		iter = btree.find(queryset[i].first);
		//if (iter == btree.end()) {
		//	continue;
		//}
		//resultset.push_back(*iter);
		//
		//int count;
		//stx::btree<double, vector<double>>::iterator iter_low, iter_up;
		//iter_low = btree.lower_bound(queryset[i].first); // include the key
		//iter_up = btree.upper_bound(queryset[i].second); // not include the key
		//if (iter_low != btree.end() && iter_up != btree.end()) {
		//	count = iter_up->second[2] - iter_low->second[2];
		//}
		//else if (iter_low != btree.end()) {
		//	count = btree.size() - iter_low->second[2];
		//}
		//else {
		//	count = 0;
		//}
		//results.push_back(count);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / queryset.size() << " ns" << endl;
}