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
			back = Backprop(batch, batch[i].first, *batch[i].second);

		nabla_b += back.first;
		nabla_w += back.second;
	}

	for (int i = 0; i < NumLayers() - 1; i++)
	{
		weights(i) -= (learningRate / batch.ImageCount()) *	nabla_w(i);
		biases(i)  -= (learningRate / batch.ImageCount()) *	nabla_b(i);
	}
}

Network::DVectorV CostDerivative(Network::DVectorV networkOut,
								 Network::DVectorV expectedOut)
{																
	return networkOut - expectedOut;
}

std::pair<Eigen::Matrix<Network::DVectorV, Eigen::Dynamic, 1>, Network::DTensor>
	Network::Backprop(const DataSet &batch, unsigned char* input, unsigned char output)
{
	auto nablaB = Eigen::Matrix<Network::DVectorV, 1, Eigen::Dynamic>();
	auto nablaW = std::vector<Network::DVectorV>(NumLayers());

	Network::DVectorV activation = Network::DVectorV();
	Network::DVectorV z = Network::DVectorV();

	std::vector<Network::DVectorV> activations;
	std::vector<Network::DVectorV> zs;

	activation.resize(batch.DataSize());

	for (int i = 0; i < batch.DataSize(); i++)
		activation(i) = input[i];

	activations.push_back(activation);

	for (int i = 1; i < NumLayers(); i++)
	{
		activations.push_back(weights(i) * activations[i - 1] + biases(i));
	}

	auto delta = CostDerivative(activations.back(), batch.ToVector(output)) *
		Math::SigmoidPrime(activations [NumLayers() - 1]);

	/* feedforward */

	for (int i = NumLayers() - 1; i > -1; i--)
	{
		int asd = 2;
	}
	
	return std::make_pair(biases, weights);
}
