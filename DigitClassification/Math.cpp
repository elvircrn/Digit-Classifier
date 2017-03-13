#include "stdafx.h"

#include <Eigen/Core>

#include "Math.h"

double Math::Sigmoid(double x)
{
	return 1.0 / (1.0 + std::exp(-x));
}

double Math::SigmoidPrime(double x)
{
	double sigmoid = Sigmoid(x);
	return sigmoid * (1 - sigmoid);
}

double Math::SigmoidPrimeUn(double x)
{
	double sigmoid = Sigmoid(x);
	return sigmoid * (1 - sigmoid);
}
