#include <cmath>

#include <Eigen/Core>

#pragma once
class Math
{
public:
	static double Sigmoid(double);
	static double SigmoidPrime(double);
	static double SigmoidPrimeUn(double);
	static Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> SigmoidPrime(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>&);
};

