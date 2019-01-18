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
#include "Stx2D.h"
#include "BoostRtree.h"

void main()
{
	//CompareResults();
	//TestApproximationWithDataset();

	//Count2DLearnedIndex(22,14);
	//RtreeCount();

	// if meet some problem during switch the bits, try to turn off the optimization, run and then turn on the optimization
	
	CountStx2D(22,11);
	//CountStx2D(16);
	//CountStx2D(14);
	//CountStx2D(13);
	//CountStx2D(12);
	//CountStx2D(10);
	
	//SumShiftLearnedIndex();
	//SumStxBtree();
	//SumLearnedIndex();
	//CountLearnedIndex();

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