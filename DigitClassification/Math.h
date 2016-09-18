#include <cmath>

#include <Eigen/Core>

#pragma once
class Math
{
public:
	static double Sigmoid(double);
	static Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Sigmoid(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>&);
};

