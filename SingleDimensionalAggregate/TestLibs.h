#pragma once
#include "LearnIndexAggregate.h"

void TestCondition() {
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SingleDimQueryTest.csv", queryset);

	arma::rowvec first_row = queryset.row(0);
	arma::rowvec second_row = queryset.row(1);
	//mat queryset_low = queryset.row(0);

	uvec indices = find(first_row < 0);
	first_row.elem(find(first_row < 0)).zeros();
	second_row(indices).print();
	first_row.print();

	first_row.elem(find(first_row < 0)).zeros();

	first_row.for_each([](mat::elem_type& val) { val = int(val); });
	first_row.print();
}

void TestSort() {
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SingleDimQueryTest2.csv", queryset);
	
	//queryset.print();

	// sort all
	//queryset = sort(queryset,"ascend",1);
	//queryset.print();

	cout << "===================" << endl;
	// relative sort
	uvec index = sort_index(queryset.row(0), "ascend");
	queryset.cols(index).print();
}

void TestJoin() {
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SingleDimQueryTest2.csv", queryset);
	arma::rowvec first_row = queryset.row(0);
	arma::rowvec second_row = queryset.row(1);

	//mat temp = join_rows(first_row, second_row);
	mat temp = join_cols(first_row, second_row);
	temp.print();
}

void TestPara() {
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/SingleDimQueryTest3.csv", dataset);
	rowvec data = dataset.row(0);
	rowvec label = dataset.row(1);
	LinearRegression lr(data, label);
	arma::vec paras = lr.Parameters();
	paras.print();
	cout << paras[0] << " " << paras[1] << endl;

	rowvec result;
	lr.Predict(data, result);
	result.print();

	double a = paras[1], b = paras[0];
	cout << a << " " << b << endl;
}