#pragma once

using namespace std;

// Haar Wavlet Synopsis
class Wavlet {

public:

	Wavlet(double t_abs = 100, double t_rel = 0.01) {
		this->t_abs = t_abs;
		this->t_rel = t_rel;
	}


	vector<double> 
	double t_abs;
	double t_rel;
}