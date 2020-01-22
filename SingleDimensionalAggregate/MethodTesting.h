#pragma once

#include"AggregateQuery.h"
#include "Utils.h"
//struct QueryResult {
//	//std::chrono::duration<double> average_query_time;
//	//std::chrono::duration<double> total_query_time;
//	unsigned long long average_query_time;
//	unsigned long long total_query_time;
//	double measured_absolute_error;
//	double measured_relative_error;
//};

QueryResult TestS2Sampling1D(vector<double> &keys, vector<double> queryset_L, vector<double> queryset_U, double p = 0.9, double Trel = 0.01, double Tabs = 100) {

	double result;
	vector<int> predicted_results;

	auto t0 = chrono::steady_clock::now();

	for (int i = 0; i < queryset_L.size(); i++) {
		result = SequentialSampling(keys, queryset_L[i], queryset_U[i]);
		predicted_results.push_back(int(result));
	}

	auto t1 = chrono::steady_clock::now();

	auto average_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / queryset_L.size();
	auto total_time = chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count();

	double MEabs, MErel;

	// check correctness
	MeasureAccuracy(predicted_results, "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/TWEET_1D.csv", MEabs, MErel);

	//ErrorGuaranteedSampling();

	QueryResult query_result;
	query_result.average_query_time = average_time;
	query_result.total_query_time = total_time;
	query_result.measured_absolute_error = MEabs;
	query_result.measured_relative_error = MErel;

	return query_result;
}


QueryResult TestFITingTree(vector<double> &keys, vector<double> &values, vector<double> queryset_L, vector<double> queryset_U, double Trel = 0.01, double Tabs = 100) {
	
	double result;
	vector<int> predicted_results;
	ATree atree(Tabs, Trel);
	atree.TrainAtree(keys, values);
	QueryResult query_result = atree.CountPrediction2(queryset_L, queryset_U, predicted_results, keys);
	return query_result;
}

QueryResult TestRMI(vector<double> &keys, vector<double> &values, vector<double> queryset_L, vector<double> queryset_U, double Trel = 0.01, double Tabs = 100) {
	
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

	StageModel stage_model(dataset, responses, arch, 0, 5000);
	stage_model.DumpParameters();
	QueryResult query_result = stage_model.CountPrediction(queryset_L, queryset_U, predicted_results, keys);

	return query_result;
}

QueryResult TestPolyfit(vector<double> &keys, vector<double> &values, vector<double> queryset_L, vector<double> queryset_U, double Trel = 0.01, double Tabs = 100, int highest_term = 1) {
	
	double result;
	vector<int> predicted_results;
	
	ReverseMaxlossOptimal RMLO(Tabs, Trel, highest_term);
	RMLO.SegmentOnTrainMaxLossModel(keys, values);
	RMLO.BuildNonLeafLayerWithBtree();
	QueryResult query_result = RMLO.CountPrediction2(queryset_L, queryset_U, predicted_results, keys);
	return query_result;
}

QueryResult TestPolyfit_MAX(vector<double> &keys, vector<double> &values, vector<double> queryset_L, vector<double> queryset_U, double Trel = 0.01, double Tabs = 100, int highest_term = 1) {

	double result;
	vector<double> predicted_results;
	ReverseMaxlossOptimal RMLO(Tabs, Trel, highest_term);
	RMLO.SegmentOnTrainMaxLossModel(keys, values);
	RMLO.BuildNonLeafLayerWithBtree();
	RMLO.PrepareMaxAggregateTree();
	QueryResult query_result = RMLO.MaxPredictionWithoutRefinement(queryset_L, queryset_U, predicted_results, keys);
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
	MeasureAccuracy(predicted_results, "C:/Users/Cloud/iCloudDrive/LearnedAggregate/VLDB_Final_Experiments/RealQueryResults/COUNT_2D.csv", MEabs, MErel);

	//ErrorGuaranteedSampling();

	QueryResult query_result;
	query_result.average_query_time = average_time;
	query_result.total_query_time = total_time;
	query_result.measured_absolute_error = MEabs;
	query_result.measured_relative_error = MErel;

	return query_result;
}

QueryResult TestPolyfit_COUNT2D(vector<double> &queryset_L1, vector<double> &queryset_L2, vector<double> &queryset_U1, vector<double> &queryset_U2, double Trel = 0.01, double Tabs = 100) {

	double result;
	vector<double> predicted_results;
	
	Maxloss2D_QuadDivide model2d(Tabs, Trel, -180.0, 180.0, -90.0, 90.0);
	//model2d.GenerateKeysAndAccuFromFile("C:/Users/Cloud/Desktop/LearnedAggregateData/Sampled2D_100M_1000_1000.csv");
	//model2d.TrainModel();
	//cout << "Bottom model size: " << model2d.model_rtree.size() << endl;
	//cout << "Bottom model size: " << model2d.temp_models.size() << endl;

	// try to save models to file
	//model2d.WriteTrainedModelsToFile("C:/Users/Cloud/Desktop/LearnedAggregateData/2D_LP_models_100M_1000_1000.csv");
	// try to read models from file
	model2d.ReadTrainedModelsFromFile("C:/Users/Cloud/Desktop/LearnedAggregateData/2D_LP_models_100M_1000_1000.csv");
	model2d.LoadRtree();

	QueryResult query_result = model2d.CountPrediction2(queryset_L1, queryset_L2, queryset_U1, queryset_U2, predicted_results);

	return query_result;
}