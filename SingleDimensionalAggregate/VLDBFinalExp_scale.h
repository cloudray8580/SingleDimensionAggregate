#pragma once

#include "AggregateQuery.h"
#include "MethodTesting.h"
#include "Utils.h"

// Exp1-1 COUNT: S2, FITingTree, RMI, Polyfit; Dataset: OSM
void VLDB_Final_Experiment_SCALE_COUNT() {

	vector<double> keys, values, query_low, query_up;
	double Eabs = 100;

	QueryResult QS;
	vector<QueryResult> QSS;

	std::ofstream run_result;
	run_result.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RunResults/Exp_SCALE.csv", std::ios::app);

	// S2 sampling
	for (int i = 1; i <= 4; i++) {
		LoadOSM_Dataset_Queryset_1D_SUBSET(keys, values, query_low, query_up, i);
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
	for (int i = 1; i <= 4; i++){
		LoadOSM_Dataset_Queryset_1D_SUBSET(keys, values, query_low, query_up, i);
		QS = TestFITingTree(keys, values, query_low, query_up, 0.01, Eabs);
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].total_paras << endl;
	}
	run_result << endl;


	// RMI, set the dataset inside the method!!!
	QSS.clear();
	for (int i = 1; i <= 4; i++) {
		LoadOSM_Dataset_Queryset_1D_SUBSET(keys, values, query_low, query_up, i);
		QS = TestRMI(keys, values, query_low, query_up, 0.01, Eabs); // the detailed settings are inside this method, along with the dataset it used !
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].hit_count << "," << QSS[i].total_paras << endl;
	}
	run_result << endl;

	//Polyfit-1
	QSS.clear();
	for (int i = 1; i <= 4; i++) {
		LoadOSM_Dataset_Queryset_1D_SUBSET(keys, values, query_low, query_up, i);
		QS = TestPolyfit(keys, values, query_low, query_up, 0.01, Eabs, 1); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].total_paras << "," << QSS[i].hit_count << "," << QSS[i].model_amount << endl;
	}
	run_result << endl;

	// Polyfit-2
	QSS.clear();
	for (int i = 1; i <= 4; i++) {
		LoadOSM_Dataset_Queryset_1D_SUBSET(keys, values, query_low, query_up, i);
		QS = TestPolyfit(keys, values, query_low, query_up, 0.01, Eabs, 2); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].total_paras << "," << QSS[i].hit_count << "," << QSS[i].model_amount << endl;
	}
	run_result << endl;

	// Polyfit-3
	QSS.clear();
	for (int i = 1; i <= 4; i++) {
		LoadOSM_Dataset_Queryset_1D_SUBSET(keys, values, query_low, query_up, i);
		QS = TestPolyfit(keys, values, query_low, query_up, 0.01, Eabs, 3); // probability= 0.9, Trel = 0.01, double Tabs = 100, int highest_term
		QSS.push_back(QS);
	}
	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].total_paras << "," << QSS[i].hit_count << "," << QSS[i].model_amount << endl;
	}
	run_result << endl;

	run_result.close();
}