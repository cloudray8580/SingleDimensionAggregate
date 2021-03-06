#pragma once

#include"AggregateQuery.h"
#include "Utils.h"
#include <string>
//struct QueryResult {
//	//std::chrono::duration<double> average_query_time;
//	//std::chrono::duration<double> total_query_time;
//	unsigned long long average_query_time;
//	unsigned long long total_query_time;
//	double measured_absolute_error;
//	double measured_relative_error;
//};

QueryResult TestS2Sampling1D(vector<double> &keys, vector<double> queryset_L, vector<double> queryset_U, double p = 0.9, double Trel = 0.01, double Tabs = 100, string RealResultPath = "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/TWEET_1D.csv") {

	double result;
	vector<int> predicted_results;

	auto t0 = chrono::steady_clock::now();

	for (int i = 0; i < queryset_L.size(); i++) {
		result = SequentialSampling(keys, queryset_L[i], queryset_U[i], 0.9, Trel, Tabs);
		predicted_results.push_back(int(result));
	}

	auto t1 = chrono::steady_clock::now();

	auto average_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / queryset_L.size();
	auto total_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();

	double MEabs, MErel;

	// check correctness
	MeasureAccuracy(predicted_results, RealResultPath, MEabs, MErel);

	//ErrorGuaranteedSampling();

	QueryResult query_result;
	query_result.average_query_time = average_time;
	query_result.total_query_time = total_time;
	query_result.measured_absolute_error = MEabs;
	query_result.measured_relative_error = MErel;

	return query_result;
}


QueryResult TestFITingTree(vector<double> &keys, vector<double> &values, vector<double> queryset_L, vector<double> queryset_U, double Trel = 0.01, double Tabs = 100, bool DoRefinement = true, string RealResultPath = "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/TWEET_1D.csv") {
	
	double result;
	vector<int> predicted_results;
	ATree atree(Tabs, Trel);
	atree.TrainAtree(keys, values);
	QueryResult query_result = atree.CountPrediction2(queryset_L, queryset_U, predicted_results, keys, DoRefinement, RealResultPath);
	return query_result;
}

QueryResult TestRMI(vector<double> &keys, vector<double> &values, vector<double> queryset_L, vector<double> queryset_U, double Trel = 0.01, double Tabs = 100, bool DoRefinement = true, string RealResultPath = "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/TWEET_1D.csv") {
	
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	dataset.shed_rows(1, 2);

	double result;
	vector<int> predicted_results;

	vector<int> arch;
	arch.push_back(1);
	arch.push_back(10);
	arch.push_back(100);
	arch.push_back(1000);

	StageModel stage_model(dataset, responses, arch, Tabs, Trel);
	stage_model.DumpParameters();
	QueryResult query_result = stage_model.CountPrediction(queryset_L, queryset_U, predicted_results, keys, DoRefinement, RealResultPath);

	return query_result;
}

QueryResult TestPolyfit(vector<double> &keys, vector<double> &values, vector<double> queryset_L, vector<double> queryset_U, double Trel = 0.01, double Tabs = 100, int highest_term = 1, bool DoRefinement = true, string RealResultPath = "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/TWEET_1D.csv") {
	
	double result;
	vector<int> predicted_results;
	
	ReverseMaxlossOptimal RMLO(Tabs, Trel, highest_term);
	RMLO.SegmentOnTrainMaxLossModel(keys, values);
	RMLO.BuildNonLeafLayerWithBtree();
	QueryResult query_result = RMLO.CountPrediction2(queryset_L, queryset_U, predicted_results, keys, DoRefinement, RealResultPath);
	return query_result;
}

QueryResult TestPolyfit_MAX(vector<double> &keys, vector<double> &values, vector<double> queryset_L, vector<double> queryset_U, double Trel = 0.01, double Tabs = 100, int highest_term = 1, bool DoRefinement = true) {

	double result;
	vector<double> predicted_results;
	ReverseMaxlossOptimal RMLO(Tabs, Trel, highest_term);
	RMLO.SegmentOnTrainMaxLossModel(keys, values);
	RMLO.BuildNonLeafLayerWithBtree();
	RMLO.PrepareMaxAggregateTree(keys, values);
	RMLO.PrepareExactAggregateMaxTree(keys, values);
	QueryResult query_result = RMLO.MaxPrediction(queryset_L, queryset_U, predicted_results, DoRefinement);
	//RMLO.ExportDatasetRangeAndModels();
	return query_result;
}

QueryResult TestS2Sampling2D(vector<double> &keys1, vector<double> &keys2, vector<double> &queryset_L1, vector<double> &queryset_L2, vector<double> &queryset_U1, vector<double> &queryset_U2, double p = 0.9, double Trel = 0.01, double Tabs = 100) {

	double result;
	vector<int> predicted_results;

	auto t0 = chrono::steady_clock::now();

	for (int i = 0; i < queryset_L1.size(); i++) {
		result = SequentialSampling2D(keys1, keys2, queryset_L1[i], queryset_L2[i], queryset_U1[i], queryset_U2[i], p, Trel, Tabs);
		predicted_results.push_back(int(result));
	}

	auto t1 = chrono::steady_clock::now();

	auto average_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / queryset_L1.size();
	auto total_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();

	double MEabs, MErel;

	// check correctness
	MeasureAccuracy(predicted_results, "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/OSM_2D.csv", MEabs, MErel);

	//ErrorGuaranteedSampling();

	QueryResult query_result;
	query_result.average_query_time = average_time;
	query_result.total_query_time = total_time;
	query_result.measured_absolute_error = MEabs;
	query_result.measured_relative_error = MErel;

	return query_result;
}

// call when the Abs error is different
QueryResult TestPolyfit_COUNT2D(vector<double> &key1, vector<double> &key2, vector<double> &queryset_L1, vector<double> &queryset_L2, vector<double> &queryset_U1, vector<double> &queryset_U2, double Trel = 0.01, double Tabs = 100, bool DoRefinement = true) {

	double result;
	vector<double> predicted_results;
	
	Maxloss2D_QuadDivide model2d(Tabs, Trel, -180.0, 180.0, -90.0, 90.0);
	model2d.GenerateKeysAndAccuFromFile("C:/Users/Cloud/iCloudDrive/LearnedAggregate/Sampled2D_100M_1000_1000_ADJUSTED.csv");
	model2d.TrainModel();

	int AbsErr = int(Tabs);
	string AbsErrStr = to_string(AbsErr);
	string filename = "C:/Users/Cloud/Desktop/LearnedAggregateData/2D_LP_models_100M_1000_1000_" + AbsErrStr + ".csv";

	// try to save models to file
	model2d.WriteTrainedModelsToFile(filename);

	// try to read models from file
	//model2d.ReadTrainedModelsFromFile(filename);
	//model2d.LoadRtree();

	model2d.PrepareExactAggregateRtree(key1, key2);
	QueryResult query_result = model2d.CountPrediction2(queryset_L1, queryset_L2, queryset_U1, queryset_U2, predicted_results, DoRefinement);

	return query_result;
}

// call when the abs error is fixed to 10000
QueryResult TestPolyfit_COUNT2D_FIXABS(vector<double> &key1, vector<double> &key2, vector<double> &queryset_L1, vector<double> &queryset_L2, vector<double> &queryset_U1, vector<double> &queryset_U2, double Trel = 0.01, double Tabs = 100, bool DoRefinement = true) {

	double result;
	vector<double> predicted_results;
	
	Maxloss2D_QuadDivide model2d(Tabs, Trel, -180.0, 180.0, -90.0, 90.0);
	//model2d.GenerateKeysAndAccuFromFile("C:/Users/Cloud/iCloudDrive/LearnedAggregate/Sampled2D_100M_1000_1000_ADJUSTED.csv");
	//model2d.TrainModel();

	int AbsErr = int(Tabs);
	string AbsErrStr = to_string(AbsErr);
	string filename = "C:/Users/Cloud/Desktop/LearnedAggregateData/2D_LP_models_100M_1000_1000_" + AbsErrStr + ".csv";

	model2d.ReadTrainedModelsFromFile(filename);
	model2d.LoadRtree();
	model2d.PrepareExactAggregateRtree(key1, key2);
	QueryResult query_result = model2d.CountPrediction2(queryset_L1, queryset_L2, queryset_U1, queryset_U2, predicted_results, DoRefinement);

	return query_result;
}