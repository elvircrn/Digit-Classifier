#include "stdafx.h"

#include <Eigen/Core>

#include "Math.h"
#include "Network.h"

using namespace Eigen;

Network::Network() { }

Network::Network(const std::vector<int>& _layerSizes)
{
	layerSizes = _layerSizes;

	biases.resize(1, NumLayers() - 1);
	weights.resize(1, NumLayers() - 1);

	for (int i = 1; i < NumLayers(); i++)
	{
		weights(i - 1) = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>::Random(layerSizes[i], layerSizes[i - 1]);
		biases(i - 1) = Eigen::Matrix<double, Eigen::Dynamic, 1>::Random(layerSizes[i], 1);
	}
}

Network::~Network()
{

}

int Network::NumLayers()
{
	return layerSizes.size();
}

auto Network::FeedForward(Network::DMatrix &a)
{
	for (int i = 0; i < NumLayers(); i++)
		a = Math::Sigmoid(weights(i) * a + biases (i));

	return a;
}

void Network::SGD(DataSet &dataSet, int epochs, double learningRate)
{
	int batchSize = dataSet.ImageCount() / epochs;
	for (int t = 0; t < dataSet.ImageCount(); t += batchSize)
	{
		dataSet.Shuffle();
		UpdateMiniBatch(dataSet, t, batchSize, learningRate);
	}
}

void Network::UpdateMiniBatch(const DataSet &batch, int batchStart, int batchSize, double learningRate)
{
	auto nabla_w = weights;
	auto nabla_b = biases;

	for (int i = 0; i < NumLayers() - 1; i++)
		nabla_w(i).setZero();
	nabla_b.setZero();

	for (int i = batchStart; i < batchStart + batchSize; i++)
	{
		std::pair<Eigen::Matrix<Eigen::Matrix<double, Eigen::Dynamic, 1>, 1, Eigen::Dynamic>,
			Eigen::Matrix<Network::DMatrix, Eigen::Dynamic, Eigen::Dynamic>> 
			back = Backprop(batch[i].first, *batch[i].second);

		nabla_b += back.first;
		nabla_w += back.second;
	}

	for (int i = 0; i < NumLayers() - 1; i++)
	{
		weights(i) -= (learningRate / batch.ImageCount()) *	nabla_w(i);
		biases(i)  -= (learningRate / batch.ImageCount()) *	nabla_b(i);
	}
}


std::pair<Eigen::Matrix<Eigen::Matrix<double, Eigen::Dynamic, 1>, 1, Eigen::Dynamic>,
	Eigen::Matrix<Network::DMatrix, Eigen::Dynamic, Eigen::Dynamic>>
	Network::Backprop(unsigned char* input, unsigned char output)
{
	auto nabla_b = Eigen::Matrix<Eigen::Matrix<double, Eigen::Dynamic, 1>, 1, Eigen::Dynamic>();
	auto nabla_w = Eigen::Matrix<Network::DMatrix, Eigen::Dynamic, Eigen::Dynamic>();
	return std::make_pair(biases, weights);
}


