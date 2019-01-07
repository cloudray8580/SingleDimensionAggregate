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
	// read dataset
	MyDataset dataset;
	LoadDataset(filename_data, dataset);
	cout << "reading dataset finished.." << endl;

	// load dataset into stx-tree
	stx::btree<double, vector<double>> btree;
	btree.bulk_load(dataset.begin(), dataset.end());
	cout << "bulk loading stx-btree finished.." << endl;

	// read and load query set
	vector<pair<double, double>> queryset;
	LoadQueryset(filename_query ,queryset);

	// execute and measure aggregate query
	auto t0 = chrono::steady_clock::now();
	int count;
	for (int i = 0; i < queryset.size(); i++) {
		CalculateCount(queryset[i].first, queryset[i].second, btree, count);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count()/queryset.size() << " ns" << endl;

	// execute again to save the correct result
	vector<double> results;
	for (int i = 0; i < queryset.size(); i++) {
		CalculateCount(queryset[i].first, queryset[i].second, btree, count);
		results.push_back(count);
	}
	SaveResutToFile(filename_result, results);

	/*ofstream outfile;
	outfile.open(filename_result);
	btree.dump(outfile);*/
	cout << "number of nodes: " << btree.CountNodesPrimary() << endl;
	//cout << "size of the Btree: " << sizeof(btree) << endl; //56
}

