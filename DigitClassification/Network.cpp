#include "stdafx.h"

#include <Eigen/Core>

#include "Math.h"
#include "Network.h"
#include "Tester.h"

using namespace Eigen;

Network::Network() { }

Network::Network(const std::vector<int>& _layerSizes) : 
	layerSizes(_layerSizes), biases(NumLayers() - 1), weights(NumLayers() - 1),
	nablaW(NumLayers() - 1), nablaB(NumLayers() - 1),
	batchNablaW(NumLayers() - 1), batchNablaB(NumLayers() - 1)
{
	double eInit = std::sqrt(6) / std::sqrt(layerSizes[0] + layerSizes.back());
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(-eInit, eInit);

	for (int i = 1; i < NumLayers(); i++)
	{
		weights[i - 1] = Network::DMatrix(layerSizes[i], layerSizes[i - 1]);
		biases[i - 1]  = Network::DVectorV(layerSizes[i], 1);

		for (int r = 0; r < weights[i - 1].rows(); r++)
			for (int c = 0; c < weights[i - 1].cols(); c++)
				weights[i - 1](r, c) = dist(mt);

		for (int r = 0; r < biases[i - 1].rows(); r++)
			biases[i - 1](r) = dist(mt);
		
		batchNablaW[i - 1] = weights[i - 1];
		batchNablaB[i - 1] = biases[i - 1];

		nablaW[i - 1] = weights[i - 1];
		nablaB[i - 1] = biases[i - 1];
	}
	biases[0].setZero();
	nablaB[0].setZero();
}

Network::Network(const std::vector<int>& _layerSizes, Network::FnCost _func, double _lambda) :
	layerSizes(_layerSizes), biases(NumLayers() - 1), weights(NumLayers() - 1),
	nablaW(NumLayers() - 1), nablaB(NumLayers() - 1),
	batchNablaW(NumLayers() - 1), batchNablaB(NumLayers() - 1),
	lambda(_lambda)
{
	double eInit = std::sqrt(6) / std::sqrt(layerSizes[0] + layerSizes.back());
	std::random_device rd;
	std::mt19937 mt(rd());
	std::uniform_real_distribution<double> dist(-eInit, eInit);

	for (int i = 1; i < NumLayers(); i++)
	{
		weights[i - 1] = Network::DMatrix(layerSizes[i], layerSizes[i - 1]);
		biases[i - 1] = Network::DVectorV(layerSizes[i], 1);

		for (int r = 0; r < weights[i - 1].rows(); r++)
			for (int c = 0; c < weights[i - 1].cols(); c++)
				weights[i - 1](r, c) = dist(mt);

		for (int r = 0; r < biases[i - 1].rows(); r++)
			biases[i - 1](r) = dist(mt);

		batchNablaW[i - 1] = weights[i - 1];
		batchNablaB[i - 1] = biases[i - 1];

		nablaW[i - 1] = weights[i - 1];
		nablaB[i - 1] = biases[i - 1];
	}
	biases[0].setZero();
	nablaB[0].setZero();
	CostDerivative = _func;
}

Network::~Network()
{

}

int Network::NumLayers() const
{
	return layerSizes.size();
}

Network::DVectorV Network::FeedForward(const Network::DVectorV &_a) const
{
	Network::DVectorV a = _a;
	for (int i = 0; i < NumLayers() - 1; i++)
	{
		DVectorV v = weights[i] * a + biases[i];
		a = v.unaryExpr(&Math::Sigmoid);
	}
	return a;
}

void Network::PrintMaxLayers() const
{
	for (int i = 0; i < NumLayers() - 1; i++)
		std::cout << "layer " << i + 1 << " max weights: " << weights[i].maxCoeff()
			      << " max biases: " << biases[i].maxCoeff() << '\n';
}

void Network::SGD(DataSet &dataSet, int epochs, int batchSize, double learningRate)
{
	while (epochs--)
	{
		std::cout << "Epochs left: " << epochs << '\n';
		dataSet.Shuffle(0, dataSet.TrainingCount());
		for (int t = 0; t < dataSet.TrainingCount(); t += batchSize)
			UpdateMiniBatch(dataSet, t, batchSize, learningRate);
		Tester::Analyze(dataSet, *this);
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
		Backprop(batch, i, batch._labels[i]);

		for (int j = 0; j < NumLayers() - 1; j++)
		{
			batchNablaW[j] += nablaW[j];
			batchNablaB[j] += nablaB[j];
		}
	}

	for (int i = 0; i < NumLayers() - 1; i++)
	{
		weights[i] -= (learningRate / batchSize) * batchNablaW[i];
		biases[i]  -= (learningRate / batchSize) * batchNablaB[i];
	}
}

double Cost(const Network::DVectorV &networkOut,
	const Network::DVectorV &expectedOut)
{
	return 0;
}

void Network::Backprop(const DataSet &batch, 
						int inputIndex, 
						unsigned char output)
{
	std::vector<Network::DVectorV> activations(NumLayers());
	std::vector<Network::DVectorV> zs;

	activations[0] = batch.GetInputVector(inputIndex);

	for (int i = 0; i < NumLayers() - 1; i++)
	{
		nablaB[i].setZero();
		nablaW[i].setZero();
		zs.push_back((weights[i] * activations[i]) + biases[i]);
		activations[i + 1] = zs.back().unaryExpr(&Math::Sigmoid);
	}

	/* Backpropagation */
	Network::DVectorV cd = CostDerivative(activations.back(), batch.ToVector(output), *this);

	Network::DVectorV delta = cd.cwiseProduct(zs.back().unaryExpr(&Math::SigmoidPrimeUn));

	nablaB.back() = delta;
	nablaW.back() = delta * activations[activations.size() - 2].transpose();

	Network::DMatrix wtd;

	for (int i = NumLayers() - 3; i > -1; i--)
	{
		wtd = weights[i + 1].transpose() * delta;
		delta = wtd.cwiseProduct(zs[i].unaryExpr(&Math::SigmoidPrimeUn));

		if (i)
			nablaB[i] = delta;

		nablaW[i] = delta * activations[i].transpose();

		if (IsRegularized())
			nablaW[i] += (lambda / NumLayers()) * weights[i];
	}
}

double Network::GetSum() const
{
	return 0.0;
}

bool Network::IsRegularized() const
{
	return lambda < std::numeric_limits<double>::epsilon();
}
