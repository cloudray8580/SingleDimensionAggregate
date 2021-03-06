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
#include "StageModel2D.h"
#include "TestNN.h"
#include "VLDBFinalExp1.h"
#include "VLDBFinalExp2.h"
#include "VLDBFinalExp6.h"
#include "VLDBFinalExp_scale.h"
#include "VLDBFinalExp_Easb.h"
#include "VLDBFinalExp_TWEET20M.h"
#include "RMI_tunning.h"

//#include <ilcplex/ilocplex.h>

void main()
{
	FangTopic();
	//RMI_Tuning();

	//TestLPHighestTerm();

	//VLDB_Final_Experiment_2_MAX();
	//VLDB_Final_Experiment_6_MAX();

	//VLDB_Final_Experiment_2_COUNT(); // for EXP6 Problem2

	//FindExactResultByScan_POI(); 
	//VLDB_Final_Experiment_POI_COUNT(); // change the real result

	//VLDB_Final_Experiment_0_COUNT();
	//VLDB_Final_Experiment_0_MAX();
	//VLDB_Final_Experiment_0_COUNT2D();

	//VLDB_Final_Experiment_SCALE_COUNT();
	
	//Approximate2D();

	//VLDB_Final_Experiment_6_COUNT();

	//Generate2DSampling(1000, 1000);
	//SpecificScan(); 
	//Approximate1DMax();
	//FindExactResultByScan2();

	//VLDB_Final_Experiment_1_COUNT();
	//VLDB_Final_Experiment_1_MAX();
	//VLDB_Final_Experiment_1_COUNT2D();
	//VLDB_Final_Experiment_2_COUNT();
	//VLDB_Final_Experiment_2_MAX();
	//VLDB_Final_Experiment_2_COUNT2D();

	//FindExactResultByScan();

	//VLDB_Final_Experiment_1_MAX();

	//FindExactResultByScan();

	//VLDB_Final_Experiment_1_COUNT();

	//ErrorGuaranteedSampling();

	//Test2DLeafVisited();

	//FinancialDataset1Model();

	//Test2DSingleQueryMax();

	//Test2DMinMax();

	//GenHist();

	//HistSegmentationWithPolyfit();

	//TestAtree(100, 0.01);

	//TestRMLO(1, 100, 0.01);

	//CompareMinMaxFinancial();

	//CompareMinMax();

	//TestMaxAggregate();

	//TestMultipleTimes();

	//PolyfitExperimentRelativeErrorAndQueryTime();

	//TestApproximateAggregateRTree();

	//experimentVerifyTrel();

	//TestRMLOApproximation();

	//TestSimpleRTree();

	//TestMaxloss2D();

	//TestLPHighestTerm();

	//TestRMLOHighestTerm();

	//TestRMLO(3);

	//TestBinarySearch();

	//TestEqualDepthSegmentation();

	//TestEqualWidthSegmentation();

	//TestDouglas();

	//TestAtree();

	//TestROLQuadratic();

	//TestStageModelBottomUpMaxLoss();

	//MeasureTimeWithoutPow();

	//TestCplex();

	//measuretime();

	// TestX(1000000, 1001000);

	//TestX1();

	//TestX2();

	//TestX4();

	//TestX4_Complete();

	//TestX3_Complete();

	//TestX3();

	//TestROLQuadratic(); //!!!

	//TestQuadraticTime();

	//TestConvexHullMethod();

	//TestFNN4();

	//TestEqualWidthSampling();

	//TestTimeLinearSegment(); 

	//TestStageModelBottomUpMaxLoss(); // !!!

	//TestMaxLoss(); // test the approximation algorithm !

	//TestLoadingNumpySaved();

	//TestSegmentation();

	//TestPrefixSumHistogram();

	//TestEntropyHistOptimized(); // actually not good

	//TestEntropyHist();

	//TestFNN3();

	//TestDouglas();

	//TestFNN2();

	//TestHist(200);

	//TestSampleling(0.01);

	//TestSTXBtree(0.0003);

	//CountLearnedIndex();

	//TestAtree(); // !!

	//TestStageModelBottomUp();

	//Count2DLearnedIndex2D_2(10); // using x, y blocks

	//Count2DLearnedIndex2D(10); // using count only in diagonal

	//Count2DLearnedIndex(10, 22, 14);

	//CountStx2DSmallRegion(10, 22, 15);
	//RtreeCount(10);
	//CountStx2DSmallRegion(10, 22, 14);

	//CompareResults();
	//TestApproximationWithDataset();

	//RtreeCount();

	// if meet some problem during switch the bits, try to turn off the optimization, run and then turn on the optimization
	
	//CountStx2D(22,14);
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