// SingleDimensionalAggregate.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include "pch.h"
#include <iostream>
#include <vector>
#include "STXBtreeAggregate.h"
#include "LearnIndexAggregate.h"
#include "StageModel.h"
#include "TestLibs.h"
#include "Utils.h"
#include "AggregateQuery.h"

void main()
{
	SumStxBtree();
	SumLearnedIndex();

	//===================================================

	//GenerateRandomQuerySet("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQueryRandom.csv");
	//TestStxMap();
	//TestPara();

	//SingleLookUpTest();
	//SingleLookUpTest();
	//SingleLookUpTest();

	//SingleLookUpTest2("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQueryRandom.csv");
	//SingleLookUpTest2("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQueryRandom.csv");
	//SingleLookUpTest2("C:/Users/Cloud/Desktop/LearnIndex/data/SortedSingleDimQueryRandom.csv");
	
	// to eliminate the effect of cache
	//TestBtreeAggregateVariant1(0.01);
	//TestBtreeAggregateVariant1(0.01);
	//TestBtreeAggregateVariant1(0.01);

	//TestJoin();
	//TestSort();
	//TestCondition();

	//TestBtreeAggregate();
	//TestBtreeAggregate();
	//TestBtreeAggregate();
	//TestLearnedMethodAggregate();

	//TestBtreeAggregate("C:/Users/Cloud/Desktop/LearnIndex/data/SingleDimQueryTest.csv");
	//TestLearnedMethodAggregate("C:/Users/Cloud/Desktop/LearnIndex/data/SingleDimQueryTest.csv");
}