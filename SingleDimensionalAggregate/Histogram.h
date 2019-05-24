#pragma once

#include<vector>

using namespace std;

class Histogram {
public:

	Histogram(vector<double> &keys) {
		this->keys = keys;
	}

	// assume the first split is 0
	void GenerateHistogramFromSplits(vector<int> &splits) {
		int start_index = 0;
		int end_index = 0;
		double start_key = 0, end_key = 0;
		int frequency = 0;
		vector<double> bucket;

		for (int i = 0; i < splits.size(); i++) {

			start_index = splits[i];
			if (i + 1 < splits.size()) {
				end_index = splits[i + 1];
			}
			else {
				end_index = keys.size() - 1;
			}

			frequency = 0;
			start_key = keys[start_index];
			end_key = keys[end_index];
			for (int j = start_index; j < end_index; j++) { // [start_index, end_index)
				frequency++;
			}

			bucket.clear();
			bucket.push_back(start_key);
			bucket.push_back(end_key);
			bucket.push_back(frequency);

			buckets.push_back(bucket);
		}
	}

	// the buckets should already be constructed
	int inline BinarySearch(double search_key, int start_index, int end_index) {
		if (start_index >= end_index) {
			return start_index;
		}

		int middle_index = (start_index + end_index) / 2;
		if (search_key >= buckets[middle_index][0] && search_key < buckets[middle_index][1]) {
			return middle_index;
		}
		else if (search_key < buckets[middle_index][0]) {
			return BinarySearch(search_key, start_index, middle_index);
		}
		else if (search_key >= buckets[middle_index][1]) {
			return BinarySearch(search_key, middle_index+1, end_index);
		}
	}

	int inline BinarySearch2(double search_key, int start_index, int end_index) {
		while (start_index < end_index) {
			int middle_index = (start_index + end_index) / 2;
			if (search_key >= buckets[middle_index][0] && search_key < buckets[middle_index][1]) {
				return middle_index;
			}
			else if (search_key < buckets[middle_index][0]) {
				end_index = middle_index;
			}
			else if (search_key >= buckets[middle_index][1]) {
				start_index = middle_index + 1;
			}
		}
	}

	void Query(vector<double> &query_low, vector<double> &query_up, vector<int> &predicted_results) {
		
		predicted_results.clear();
		double start_key1, start_key2, end_key1, end_key2, low_frequency, up_frequency;
		int start_index, end_index;
		for (int i = 0; i < query_low.size(); i++) {
			int query_count = 0;

			start_index = BinarySearch2(query_low[i], 0, buckets.size() - 1);
			end_index = BinarySearch2(query_up[i], 0, buckets.size() - 1);

			low_frequency = buckets[start_index][2];
			start_key1 = buckets[start_index][0];
			start_key2 = buckets[start_index][1];

			up_frequency = buckets[end_index][2];
			end_key1 = buckets[end_index][0];
			end_key2 = buckets[end_index][1];

			query_count += (start_key2 - query_low[i]) / (start_key2 - start_key1) * low_frequency;
			start_index++;

			while (start_index < end_index) {
				//cout << start_index << " " << end_index << " " << i << endl;
				query_count += buckets[start_index][2];
				start_index++;
			}

			query_count += (query_up[i] - end_key1) / (end_key2 - end_key1) * up_frequency;
			predicted_results.push_back(query_count);
		}
	}

	// the buckets should already be constructed
	void GeneratePrefixSUM() {

		double sum = 0; 
		vector<double> prefixsum_bucket;
		for (int i = 0; i < buckets.size(); i++) {
			prefixsum_bucket.clear();
			prefixsum_bucket.push_back(buckets[i][0]);
			prefixsum_bucket.push_back(buckets[i][1]);
			prefixsum_bucket.push_back(buckets[i][2]);
			
			sum += buckets[i][2];
			prefixsum_bucket.push_back(sum);
			prefixsum_buckets.push_back(prefixsum_bucket);
		}
	}

	void QueryWithPrefixSUM(vector<double> &query_low, vector<double> &query_up, vector<int> &predicted_results) {
		
		predicted_results.clear();
		double start_key1, start_key2, end_key1, end_key2, low_frequency, up_frequency, low_sum, up_sum;
		int start_index, end_index;
		for (int i = 0; i < query_low.size(); i++) {
			int query_count = 0;

			start_index = BinarySearch2(query_low[i], 0, buckets.size() - 1);
			end_index = BinarySearch2(query_up[i], 0, buckets.size() - 1);

			low_sum = prefixsum_buckets[start_index][3];
			low_frequency = prefixsum_buckets[start_index][2];
			start_key1 = prefixsum_buckets[start_index][0];
			start_key2 = prefixsum_buckets[start_index][1];

			up_sum = prefixsum_buckets[end_index][3];
			up_frequency = prefixsum_buckets[end_index][2];
			end_key1 = prefixsum_buckets[end_index][0];
			end_key2 = prefixsum_buckets[end_index][1];
			
			query_count = up_sum - low_sum;

			query_count += (start_key2 - query_low[i]) / (start_key2 - start_key1) * low_frequency; // include the first part

			query_count -= (end_key2 - query_up[i]) / (end_key2 - end_key1) * up_frequency; // exclude the last part

			predicted_results.push_back(query_count);
		}
	}

	vector<double> keys;
	vector<vector<double>> buckets; // start, end, frequency
	vector<vector<double>> prefixsum_buckets; // start, end, frequency, sum
};