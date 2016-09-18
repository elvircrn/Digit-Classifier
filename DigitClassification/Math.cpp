#include "stdafx.h"

#include <Eigen/Core>

#include "Math.h"

double Math::Sigmoid(double x)
{
	return 1.0 / (1.0 + std::exp(-x));
}

Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& Math::Sigmoid(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> &m)
{
	auto x = m;

	for (int i = 0; i < m.rows(); i++)
		for (int j = 0; j < m.cols(); j++)
			x(i, j) = Math::Sigmoid(m(i, j));

	return x;
}
