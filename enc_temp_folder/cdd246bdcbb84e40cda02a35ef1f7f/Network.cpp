#include "stdafx.h"

#include <Eigen/Core>

#include "Math.h"
#include "Network.h"

using namespace Eigen;

Network::Network() { }

Network::Network(const std::vector<int>& _layerSizes) : layerSizes(_layerSizes), biases(NumLayers()), weights(NumLayers())
{
	for (int i = 1; i < NumLayers(); i++)
	{
		weights[i - 1] = Network::DMatrix::Random(layerSizes[i], layerSizes[i - 1]);
		biases[i - 1]  = Network::DVectorV::Random(layerSizes[i], 1);
	}
	int asd = 2;
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
	for (int i = 0; i < NumLayers() - 1; i++)
		a = Math::Sigmoid(weights[i] * a + biases[i]);
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
	std::vector<Network::DMatrix>  nabla_w = weights;
	std::vector<Network::DVectorV> nabla_b = biases;

	for (int i = 0; i < NumLayers() - 1; i++)
	{
		nabla_w[i].setZero();
		nabla_b[i].setZero();
	}

	for (int i = batchStart; i < batchStart + batchSize; i++)
	{
		auto back = Backprop(batch, batch[i].first, *batch[i].second);

		for (int j = 0; j < NumLayers() - 1; j++)
		{
			nabla_b[j] += back.first[j];
			nabla_w[j] += back.second[j];
		}
	}

	for (int i = 0; i < NumLayers() - 1; i++)
	{
		if (weights[i].rows() != nabla_w[i].rows() || weights[i].cols() != nabla_w[i].cols())
		{
			int x = nabla_w[i].rows();
			int y = nabla_w[i].cols();
			int ses = 2;
		}
		weights[i] -= (learningRate / batch.ImageCount()) *	nabla_w[i];
		biases[i]  -= (learningRate / batch.ImageCount()) *	nabla_b[i];
	}
	int asd = 2;
}

Network::DVectorV CostDerivative(Network::DVectorV networkOut,
	Network::DVectorV expectedOut)
{
	return networkOut - expectedOut;
}

std::pair<std::vector<Network::DVectorV>, std::vector<Network::DMatrix>>
Network::Backprop(const DataSet &batch, unsigned char* input, unsigned char output)
{
	std::vector<DMatrix> nablaW = weights;
	std::vector<DVectorV> nablaB = biases;

	for (int i = 0; i < NumLayers() - 1; i++)
	{
		nablaW[i].setZero();
		nablaB[i].setZero();
	}

	Network::DVectorV activation = Network::DVectorV(layerSizes[0], 1);
	Network::DVectorV z = Network::DVectorV();

	std::vector<Network::DVectorV> activations;
	std::vector<Network::DVectorV> zs;

	/* Feedforward */
	//activation.resize(batch.PixelCount());

	for (int i = 0; i < batch.DataSize(); i++)
		activation(i, 0) = input[i];

	activations.push_back(activation);

	for (int i = 0; i < NumLayers() - 1; i++)
	{
		zs.push_back((weights[i] * activation) + biases[i]);
		activation.resize(zs.back().rows(), zs.back().cols());
		activation = Math::Sigmoid(zs.back());
		activations.push_back(weights[i] * activations[i] + biases[i]);
	}

	Network::DVectorV cd = CostDerivative(activations.back(), batch.ToVector(output));
	Network::DVectorV ac = Math::SigmoidPrime(zs.back());

	/* Backpropagation */
	Network::DVectorV delta = cd.cwiseProduct(ac);

	nablaB[nablaW.size() - 2] = delta;
	nablaW[nablaW.size() - 2] = delta * activations[activations.size() - 2].transpose();

	for (int i = NumLayers() - 2; i > 0; i--)
	{
		DVectorV stepBack = weights[i].transpose() * delta;
		DVectorV sprime = Math::SigmoidPrime(zs[i - 1]);

		delta = stepBack.cwiseProduct(sprime);

		nablaB[i] = delta;
		nablaW[i] = delta * activations[i + 1].transpose();
	}


	return std::make_pair(nablaB, nablaW);
}
