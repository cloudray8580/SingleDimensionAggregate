#pragma once

#include "AggregateQuery.h"
#include "MethodTesting.h"
#include "Utils.h"

// Exp1-1 COUNT: S2, FITingTree, RMI, Polyfit; Dataset: TWEET
void VLDB_Final_Experiment_POI_COUNT() {

	vector<double> keys, values, query_low, query_up;
	LoadTWEET20MDataset(keys, values);
	LoadTWEET20MQuerySet(query_low, query_up);

	double Eabs = 100;
	//vector<double> Eabs_collection = { 1000 };
	vector<double> Eabs_collection = { 20, 50, 100, 200, 500, 1000 };

	QueryResult QS;
	vector<QueryResult> QSS;

	std::ofstream run_result;
	run_result.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RunResults/Exp0_POI.csv", std::ios::app);

	string RealResultPath = "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/POI_COUNT.csv";

	//// S2 sampling
	//for (int i = 0; i < Eabs_collection.size(); i++) {
	//	Eabs = Eabs_collection[i];
	//	QS = TestS2Sampling1D(keys, query_low, query_up, 0.9, 0.01, Eabs, RealResultPath); // probability= 0.9, Trel = 0.01, double Tabs = 100
	//	QSS.push_back(QS);
	//}
	////store it in file
	//for (int i = 0; i < QSS.size(); i++) {
	//	run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << endl;
	//}
	//run_result << endl;


	// FITingTree
	QSS.clear();
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestFITingTree(keys, values, query_low, query_up, 0.01, Eabs, false, RealResultPath);
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].total_paras << endl;
	}
	run_result << endl;


	// RMI, set the dataset inside the method!!!
	QSS.clear();
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestRMI(keys, values, query_low, query_up, 0.01, Eabs, false, RealResultPath); // the detailed settings are inside this method, along with the dataset it used !
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].hit_count << "," << QSS[i].total_paras << endl;
	}
	run_result << endl;

	////Polyfit-1
	//QSS.clear();
	//for (int i = 0; i < Eabs_collection.size(); i++) {
	//	Eabs = Eabs_collection[i];
	//	QS = TestPolyfit(keys, values, query_low, query_up, 0.01, Eabs, 1, true, RealResultPath); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
	//	QSS.push_back(QS);
	//}
	////store it in file
	//for (int i = 0; i < QSS.size(); i++) {
	//	run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].total_paras << "," << QSS[i].hit_count << "," << QSS[i].model_amount << endl;
	//}
	//run_result << endl;

	// Polyfit-2
	QSS.clear();
	for (int i = 0; i < Eabs_collection.size(); i++) {
		Eabs = Eabs_collection[i];
		QS = TestPolyfit(keys, values, query_low, query_up, 0.01, Eabs, 2, false, RealResultPath); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].total_paras << "," << QSS[i].hit_count << "," << QSS[i].model_amount << endl;
	}
	run_result << endl;

	//// Polyfit-3
	//QSS.clear();
	//for (int i = 0; i < Eabs_collection.size(); i++) {
	//	Eabs = Eabs_collection[i];
	//	QS = TestPolyfit(keys, values, query_low, query_up, 0.01, Eabs, 3, true, RealResultPath); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
	//	QSS.push_back(QS);
	//}
	////store it in file
	//for (int i = 0; i < QSS.size(); i++) {
	//	run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].total_paras << "," << QSS[i].hit_count << "," << QSS[i].model_amount << endl;
	//}
	//run_result << endl;

	run_result.close();
}