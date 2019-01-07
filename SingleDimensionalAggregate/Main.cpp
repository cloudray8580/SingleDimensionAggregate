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
	arch.push_back(1);
	arch.push_back(1);
	StageModel stage_model(dataset, responses, arch);
	cout << "finish stage model initilization.." << endl;

	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQuery2.csv", queryset);
	mat queryset_low = queryset.row(0);
	mat queryset_up = queryset.row(1);

	auto t0 = chrono::steady_clock::now();
	arma::rowvec predictions_low, predictions_up;
	stage_model.InitQuerySet(queryset_low);
	stage_model.Predict(queryset_low, predictions_low);
	stage_model.InitQuerySet(queryset_up);
	stage_model.Predict(queryset_up, predictions_up);
	auto t1 = chrono::steady_clock::now();

	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;
	cout << "finish stage model prediction.." << endl;

	stage_model.MeasureAccuracy(predictions_up-predictions_low);

	//TestJoin();
	//TestSort();
	//TestCondition();

	//TestBtreeAggregate();
	//TestLearnedMethodAggregate();

	/*TestBtreeAggregate("C:/Users/Cloud/Desktop/LearnIndex/data/SingleDimQueryTest.csv");
	TestLearnedMethodAggregate("C:/Users/Cloud/Desktop/LearnIndex/data/SingleDimQueryTest.csv");*/
}