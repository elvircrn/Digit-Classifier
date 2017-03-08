#include "stdafx.h"

#include <Eigen/Core>

#include "Math.h"
#include "Network.h"

using namespace Eigen;

Network::Network() { }

Network::Network(const std::vector<int>& _layerSizes) : 
	layerSizes(_layerSizes), biases(NumLayers() - 1), weights(NumLayers() - 1),
	nablaW(NumLayers() - 1), nablaB(NumLayers() - 1),
	batchNablaW(NumLayers() - 1), batchNablaB(NumLayers() - 1)
{
	for (int i = 1; i < NumLayers(); i++)
	{
		weights[i - 1] = Network::DMatrix::Random(layerSizes[i], layerSizes[i - 1]);
		biases[i - 1]  = Network::DVectorV::Random(layerSizes[i], 1);
	}
}

Network::~Network()
{

}

int Network::NumLayers() const
{
	return layerSizes.size();
}

Network::DVectorV Network::FeedForward(const Network::DMatrix &_a) const
{
	Network::DMatrix a = _a;
	for (int i = 0; i < NumLayers() - 1; i++)
		a = Math::Sigmoid(weights[i] * a + biases[i]);
	return (Network::DVectorV)a;
}

void Network::SGD(DataSet &dataSet, int epochs, double learningRate)
{
	int batchSize = dataSet.ImageCount() / epochs;
	for (int t = 0; t < DataSet::TRAINING_COUNT; t += batchSize)
	{
		dataSet.Shuffle(0, DataSet::TRAINING_COUNT);
		UpdateMiniBatch(dataSet, t, batchSize, learningRate);
	}
}

void Network::UpdateMiniBatch(const DataSet &batch, 
								int batchStart, 
								int batchSize, 
								double learningRate)
{
	for (int i = 0; i < NumLayers() - 1; i++)
	{
		batchNablaW[i].setZero();
		batchNablaB[i].setZero();
	}

	for (int i = batchStart; i < batchStart + batchSize; i++)
	{
		Backprop(batch, batch[i].first, *batch[i].second);

		for (int j = 0; j < NumLayers() - 1; j++)
		{
			batchNablaW[j] += nablaW[j];
			batchNablaB[j] += nablaB[j];
		}
	}

	for (int i = 0; i < NumLayers() - 1; i++)
	{
		weights[i] -= (learningRate / batch.ImageCount()) *	batchNablaW[i];
		biases[i]  -= (learningRate / batch.ImageCount()) *	batchNablaB[i];
	}
}

Network::DVectorV CostDerivative(Network::DVectorV networkOut,
	Network::DVectorV expectedOut)
{
	return networkOut - expectedOut;
}

void Network::Backprop(const DataSet &batch, 
						std::vector<unsigned char>::const_iterator input, 
						unsigned char output)
{
	Network::DVectorV activation = Network::DVectorV(layerSizes[0], 1);

	std::vector<Network::DMatrix> activations;
	std::vector<Network::DVectorV> zs;

	/* Feedforward */
	for (int i = 0; i < batch.PixelCount(); i++)
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

	nablaB.back() = delta;
	nablaW.back() = delta * activations[activations.size() - 2].transpose();

	for (int i = NumLayers() - 3; i > -1; i--)
	{
		DVectorV stepBack = weights[i + 1].transpose() * delta;
		DVectorV sprime = Math::SigmoidPrime(zs[i]);

		delta = stepBack.cwiseProduct(sprime);

		nablaB[i] = delta;
		nablaW[i] = delta * activations[i].transpose();
	}
}
