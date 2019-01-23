#pragma once

#include "mlpack/core.hpp"
#include <mlpack/methods/linear_regression/linear_regression.hpp>
#include <vector>

using namespace std;
using namespace mlpack::regression;

class StageModelBottomUp {
public:
	StageModelBottomUp(int level = 4, int blocks = 1000, double threshold=0.01) {
		this->level = level;
		this->initial_blocks = blocks;
		this->threshold = threshold;
	}

	void Train(const arma::mat& dataset, const arma::rowvec& labels) {
		double current_error = 0;
		int continue_block = 1;
		int TOTAL_SIZE = dataset.n_cols;
		int step = TOTAL_SIZE / initial_blocks;
		int dataset_begin = 0, dataset_end = 0 + step;

		// for each continous block
		arma::mat current_trainingset = dataset.cols(dataset_begin, dataset_end);
		arma::rowvec current_response = labels.cols(dataset_begin, dataset_end);

		double current_accuracy;
		bool first_flag = true;

		// for the entire dataset
		while (true) {

			// for an continous_block
			do {
				LinearRegression lr_poineer(current_trainingset, current_response);
				// Test the lr's accuracy
				current_accuracy = 0; // measure the accuracy

				// if the accuracy is below threshold, continous merge with the next block
				if (current_accuracy < threshold) {
					dataset_end += step; // judge border!!!
				}
				else if (first_flag) {
					// if the first_accuracy is already too large, start a new block, except it is the first block, if it's the first block, divide the region into half?
				}
				else {
					// else if the second_accuracy is also too large
					dataset_end -= step; // backward
					// save the best range, model
					break;
				}

			} while (true);

			// reset the start and end for the next block
			if (true) {// judge border
				dataset_begin = dataset_end + 1;
				dataset_end = dataset_begin + step;
			}
		}

	}

	int level;
	int initial_blocks;
	double threshold;
};