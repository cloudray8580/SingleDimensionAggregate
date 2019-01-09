// SingleDimensionalAggregate.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include "pch.h"
#include <iostream>
#include <vector>
#include "STXBtreeAggregate.h"
#include "LearnIndexAggregate.h"
#include "StageModel.h"
#include "TestLibs.h"

void main()
{
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimPOIs2.csv", dataset);
	arma::rowvec responses = dataset.row(dataset.n_rows - 1);
	dataset.shed_rows(1, 2);

	vector<int> arch;
	arch.push_back(1);
	arch.push_back(10);
	arch.push_back(100);
	arch.push_back(1000);
	StageModel stage_model(dataset, responses, arch);
	cout << "finish stage model initilization.." << endl;

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);

	arma::rowvec predictions_low, predictions_up;
	vector<double> queryset_low_v, queryset_up_v, results_low, results_up;
	StageModel::RowvecToVector(queryset_low, queryset_low_v);
	StageModel::RowvecToVector(queryset_up, queryset_up_v);
	stage_model.DumpParameters();

	auto t0 = chrono::steady_clock::now();
	stage_model.PredictVector(queryset_low_v, results_low);
	stage_model.PredictVector(queryset_up_v, results_up);

	/*StageModel::PredictNaiveSingleLR2(dataset, responses, queryset_low, predictions_low);
	StageModel::PredictNaiveSingleLR2(dataset, responses, queryset_up, predictions_up);*/

	//StageModel::PredictNaiveSingleLR(dataset, responses, queryset_low, predictions_low);
	//StageModel::PredictNaiveSingleLR(dataset, responses, queryset_up, predictions_up);
	
	//stage_model.InitQuerySet(queryset_low);
	//stage_model.Predict(queryset_low, predictions_low);
	//stage_model.InitQuerySet(queryset_up);   
	//stage_model.Predict(queryset_up, predictions_up);
	//predictions_up -= predictions_low;

	for (int i = 0; i < results_up.size(); i++) {
		results_up[i] -= results_low[i];
	}

	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "finish stage model prediction.." << endl;

	StageModel::VectorToRowvec(predictions_up, results_up);
	stage_model.MeasureAccuracy(predictions_up);

	//===================================================

	//TestPara();

	/*SingleLookUpTest();
	SingleLookUpTest();
	SingleLookUpTest();

	SingleLookUpTest2();
	SingleLookUpTest2();
	SingleLookUpTest2();*/
	
	// to eliminate the effect of cache
	//TestBtreeAggregateVariant1(0.01);
	//TestBtreeAggregateVariant1(0.01);
	//TestBtreeAggregateVariant1(0.01);

	//TestJoin();
	//TestSort();
	//TestCondition();

	/*TestBtreeAggregate();
	TestBtreeAggregate();
	TestBtreeAggregate();*/
	//TestLearnedMethodAggregate();

	/*TestBtreeAggregate("C:/Users/Cloud/Desktop/LearnIndex/data/SingleDimQueryTest.csv");
	TestLearnedMethodAggregate("C:/Users/Cloud/Desktop/LearnIndex/data/SingleDimQueryTest.csv");*/
}