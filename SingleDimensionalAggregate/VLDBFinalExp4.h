#pragma once

#include "AggregateQuery.h"
#include "MethodTesting.h"
#include "Utils.h"

// Experiment 4: Parameters - Eabs
// Query: COUNT, MAX, Count(2D)
// - COUNT: S2, FITingTree, RMI, Polyfit; Dataset: TWEET
// - MAX: aMaxTree, Polyfit; Dataset: HKI
// - COUNT(2D): S2, aR-tree, Polyfit; Dataset: OSM

// this could be actually the same as Exp 1

// Exp4-1 COUNT: S2, FITingTree, RMI, Polyfit; Dataset: TWEET
void VLDB_Final_Experiment_4_COUNT() {

	vector<double> keys, values, query_low, query_up;
	LoadTweetDataset(keys, values);
	LoadTweetQuerySet(query_low, query_up);

	double Eabs = 100;
	vector<double> Eabs_collection = { 50, 100, 200, 500, 1000 };

	QueryResult QS;
	vector<QueryResult> QSS;

	std::ofstream run_result;
	run_result.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RunResults/Exp4_COUNT.csv", std::ios::app);

	// S2 sampling
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestS2Sampling1D(keys, query_low, query_up, 0.9, 0.01, Eabs); // probability= 0.9, Trel = 0.01, double Tabs = 100
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << endl;
	}
	run_result << endl;


	// FITingTree
	QSS.clear();
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestFITingTree(keys, values, query_low, query_up, 0.01, Eabs);
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << endl;
	}
	run_result << endl;


	// RMI, set the dataset inside the method!!!
	QSS.clear();
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestRMI(keys, values, query_low, query_up, 0.01, Eabs); // the detailed settings are inside this method, along with the dataset it used !
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << endl;
	}
	run_result << endl;


	// Polyfit 
	QSS.clear();
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestPolyfit(keys, values, query_low, query_up, 0.01, Eabs, 1); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << endl;
	}
	run_result << endl;

	run_result.close();
}

// Exp4-2 MAX: aMaxTree, Polyfit; Dataset: HKI
void VLDB_Final_Experiment_4_MAX() {

	vector<double> keys, values, query_low, query_up;
	LoadHKIDataset(keys, values);
	LoadHKIQuerySet(query_low, query_up);

	double Eabs = 100;
	vector<double> Eabs_collection = { 50, 100, 200, 500, 1000 };
	//vector<double> Eabs_collection = { 1000,2000 };

	QueryResult QS;
	vector<QueryResult> QSS;

	std::ofstream run_result;
	run_result.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RunResults/Exp4_MAX.csv", std::ios::app);

	// Polyfit
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestPolyfit_MAX(keys, values, query_low, query_up, 0.01, Eabs, 1); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].hit_count << "," << QSS[i].model_amount << endl;
	}
	run_result << endl;

	// the result of aggregate max tree should be a single line
}


// Exp4-3 MAX: COUNT(2D): S2, aR-tree, Polyfit; Dataset: OSM
void VLDB_Final_Experiment_4_COUNT2D() {

	vector<double> keys1, keys2, query_low1, query_low2, query_up1, query_up2;
	LoadOSMDataset(keys1, keys2);
	LoadOSMQuerySet(query_low1, query_low2, query_up1, query_up2);

	double Eabs = 100;
	vector<double> Eabs_collection = { 500, 1000, 2000 };

	QueryResult QS;
	vector<QueryResult> QSS;

	std::ofstream run_result;
	run_result.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RunResults/Exp4_COUNT2D.csv", std::ios::app);

	// S2 sampling
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestS2Sampling2D(keys1, keys2, query_low1, query_low2, query_up1, query_up2, 0.9, 0.01, Eabs); // probability= 0.9, Trel = 0.01, double Tabs = 100
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << endl;
	}
	run_result << endl;

	QSS.clear();
	// Polyfit, use its own accumulation dataset, set in its own method
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestPolyfit_COUNT2D(query_low1, query_low2, query_up1, query_up2, 0.01, Eabs);
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << endl;
	}
	run_result << endl;

	// the result of aggregate max tree should be a single line
}