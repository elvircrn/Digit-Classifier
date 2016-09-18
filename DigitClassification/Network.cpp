#include "stdafx.h"

#include <Eigen/Core>

#include "Math.h"
#include "Network.h"

using namespace Eigen;

Network::Network() { }

Network::Network(const std::vector<int>& _layerSizes)
{
	layerSizes = _layerSizes;

	biases.resize(1, GetNumLayers() - 1);
	weights.resize(1, GetNumLayers() - 1);

	for (int i = 1; i < GetNumLayers(); i++)
	{
		weights(i - 1) = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Random(layerSizes[i], layerSizes[i - 1]);
		biases(i - 1) = Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(layerSizes[i], 1);
	}
}

Network::~Network()
{

}

int Network::GetNumLayers()
{
	return layerSizes.size();
}

auto Network::FeedForward(Network::DMatrix &a)
{
	for (int i = 0; i < GetNumLayers(); i++)
		a = Math::Sigmoid(weights(i) * a + biases (i));

	return a;
}

void Network::SGD(const DataSet &dataSet, int epochs, double learningRate)
{
	for (int t = 0; t < epochs; t++)
	{
		
	}
}
