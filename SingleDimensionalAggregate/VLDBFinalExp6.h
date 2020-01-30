#pragma once

#include "AggregateQuery.h"
#include "MethodTesting.h"
#include "Utils.h"

// Experiment 4: QueryTime - Eabs - Degree
// Query: COUNT, MAX, Count(2D)
// - COUNT: Polyfit; Dataset: TWEET
// - MAX: Polyfit; Dataset: HKI
// - COUNT(2D): Polyfit; Dataset: OSM

// Exp6-1 COUNT: S2, FITingTree, RMI, Polyfit; Dataset: TWEET
void VLDB_Final_Experiment_6_COUNT() {

	vector<double> keys, values, query_low, query_up;
	LoadTweetDataset(keys, values);
	LoadTweetQuerySet(query_low, query_up);

	double Eabs = 100;
	vector<double> Eabs_collection = { 50, 100, 200, 500, 1000 };

	QueryResult QS;
	vector<QueryResult> QSS;

	std::ofstream run_result;
	run_result.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RunResults/Exp6_COUNT.csv", std::ios::app);

	// Polyfit - 1
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

	// Polyfit - 2
	QSS.clear();
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestPolyfit(keys, values, query_low, query_up, 0.01, Eabs, 2); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << endl;
	}
	run_result << endl;

	// Polyfit - 3
	QSS.clear();
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestPolyfit(keys, values, query_low, query_up, 0.01, Eabs, 3); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << endl;
	}
	run_result << endl;

	run_result.close();
}


// Exp6-2 MAX: aMaxTree, Polyfit; Dataset: HKI
void VLDB_Final_Experiment_6_MAX() {

	vector<double> keys, values, query_low, query_up;
	LoadHKIDataset(keys, values);
	LoadHKIQuerySet(query_low, query_up);

	double Eabs = 100;
	vector<double> Eabs_collection = { 50, 100, 200, 500, 1000 };
	//vector<double> Eabs_collection = { 1000,2000 };

	QueryResult QS;
	vector<QueryResult> QSS;

	std::ofstream run_result;
	run_result.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RunResults/Exp6_MAX.csv", std::ios::app);

	// Polyfit - 1
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestPolyfit_MAX(keys, values, query_low, query_up, 0.01, Eabs, 1); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << " " << QSS[i].hit_count << "," << QSS[i].model_amount << endl;
	}
	run_result << endl;

	// Polyfit - 2
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestPolyfit_MAX(keys, values, query_low, query_up, 0.01, Eabs, 2); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << " " << QSS[i].hit_count << "," << QSS[i].model_amount << endl;
	}
	run_result << endl;

	// Polyfit - 3
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestPolyfit_MAX(keys, values, query_low, query_up, 0.01, Eabs, 3); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << " " << QSS[i].hit_count << "," << QSS[i].model_amount << endl;
	}
	run_result << endl;
}

// Exp6-3 MAX: COUNT(2D): Polyfit; Dataset: OSM
//void VLDB_Final_Experiment_1_COUNT2D() {

//	// the result of aggregate max tree should be a single line
//}