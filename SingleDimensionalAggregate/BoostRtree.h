#pragma once
#include <boost/geometry.hpp> // annotate this for ambiguous rank, or adjust the initilization list with normal initilization way
#include <boost/geometry/index/rtree.hpp>
#include <boost/geometry/geometries/point_xy.hpp>
#include <boost/geometry/geometries/box.hpp>
#include <boost/function_output_iterator.hpp>

#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <stx/btree.h>
#include "LearnIndexAggregate.h"
#include "StageModel.h"
#include "Utils.h"
#include <chrono>

namespace bg = boost::geometry;
namespace bgi = boost::geometry::index;
namespace bgid = boost::geometry::index::detail;

using namespace std;

//typedef boost::geometry::model::point<,2,>

typedef bg::model::d2::point_xy<double> MyPoint; // need to include point_xy.hpp
typedef bg::model::box<MyPoint> Box;

void RtreeCount() {
	// load dataset
	mat dataset;
	bool loaded = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/HilbertSortedPOIs2.csv", dataset);
	arma::rowvec x = dataset.row(0); // x
	arma::rowvec y = dataset.row(1); // y
	vector<double> x_v, y_v;
	StageModel::RowvecToVector(x, x_v);
	StageModel::RowvecToVector(y, y_v);

	// construct Rtree
	bgi::rtree<MyPoint, bgi::linear<16>> rtree;
	for (int i = 0; i < x_v.size(); i++) {
		MyPoint point(x_v[i], y_v[i]);
		rtree.insert(point);
	}
	cout << "finsih inserting data to Rtree" << endl;

	// load queryset
	mat queryset;
	bool loaded2 = mlpack::data::Load("C:/Users/Cloud/Desktop/LearnIndex/data/Sorted2DimQuery2.csv", queryset);
	mat queryset_x_low = queryset.row(0);
	mat queryset_x_up = queryset.row(1);
	mat queryset_y_low = queryset.row(2);
	mat queryset_y_up = queryset.row(3);
	vector<double> query_x_low_v, query_x_up_v, query_y_low_v, query_y_up_v;
	StageModel::RowvecToVector(queryset_x_low, query_x_low_v);
	StageModel::RowvecToVector(queryset_x_up, query_x_up_v);
	StageModel::RowvecToVector(queryset_y_low, query_y_low_v);
	StageModel::RowvecToVector(queryset_y_up, query_y_up_v);

	// query
	std::vector<MyPoint> returned_values;
	vector<int> counts;
	int count;
	auto t0 = chrono::steady_clock::now();

	size_t cardinality = 0; // number of matches in set
	auto count_only = boost::make_function_output_iterator([&cardinality](bgi::rtree<MyPoint, bgi::linear<16>>::value_type const&) { ++cardinality; });

	for (int i = 0; i < query_x_low_v.size(); i++) {
		
		Box query_region(MyPoint(query_x_low_v[i], query_y_low_v[i]), MyPoint(query_x_up_v[i], query_y_up_v[i]));
		//returned_values.clear();
		//rtree.query(bgi::intersects(query_region), std::back_inserter(returned_values));
		//counts.push_back(returned_values.size());
		
		//count = rtree.count(bgi::intersects(query_region)); // not works
		//rtree.count(MyPoint(query_x_low_v[i], query_y_low_v[i])); // works
		//counts.push_back(count);

		cardinality = 0;
		rtree.query(bgi::intersects(query_region), count_only);
		counts.push_back(cardinality);
	}
	auto t1 = chrono::steady_clock::now();
	cout << "Total Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() << " ns" << endl;
	cout << "Average Time in chrono: " << chrono::duration_cast<chrono::nanoseconds>(t1 - t0).count() / (queryset.size() / queryset.n_rows) << " ns" << endl;

	// compare the correctness
	vector<int> real_results;
	CalculateRealCountWithScan2D(dataset, queryset, real_results);

	double relative_error;
	double accu = 0;
	double accu_absolute = 0;
	int total_size = counts.size();
	for (int i = 0; i < counts.size(); i++) {
		if (real_results[i] == 0) {
			total_size--;
			continue;
		}
		relative_error = abs(double(counts[i] - real_results[i]) / real_results[i]);
		accu += relative_error;
		accu_absolute += abs(counts[i] - real_results[i]);
		//cout << "relative error " << i << ": " << relative_error << endl;
	}
	double avg_rel_err = accu / total_size;
	cout << "average relative error: " << avg_rel_err << endl;
	cout << "average absolute error: " << accu_absolute / total_size << endl;
}