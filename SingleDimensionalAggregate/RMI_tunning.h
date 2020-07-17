#pragma once

#include "AggregateQuery.h"
#include "MethodTesting.h"
#include "Utils.h"

void RMI_Tuning() {

	vector<double> keys, values, query_low, query_up;
	LoadTweetDataset(keys, values);
	LoadTweetQuerySet(query_low, query_up);
	vector<QueryResult> QSS;

	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	dataset.shed_rows(1, 2);

	vector<int> bottom_model_amount = { 100, 200, 300, 400, 500, 1000, 1500 };
	int models;
	for (int i = 0; i < bottom_model_amount.size(); i++) {

		double result;
		vector<int> predicted_results;

		models = bottom_model_amount[i];

		vector<int> arch;
		arch.push_back(1);
		arch.push_back(10);
		arch.push_back(100);
		arch.push_back(models);

		StageModel stage_model(dataset, responses, arch, 100, 0.01);
		stage_model.DumpParameters();

		QueryResult query_result = stage_model.CountPrediction(query_low, query_up, predicted_results, keys, false);
		QSS.push_back(query_result);
	}
	

	std::ofstream run_result;
	run_result.open("C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RunResults/RMI_Tuning.csv", std::ios::app);

	//store it in file
	for (int i = 0; i < QSS.size(); i++) {
		run_result << QSS[i].average_query_time << "," << QSS[i].total_query_time << "," << QSS[i].measured_absolute_error << "," << QSS[i].measured_relative_error << "," << QSS[i].hit_count << "," << QSS[i].total_paras << endl;
	}
	run_result << endl;
}