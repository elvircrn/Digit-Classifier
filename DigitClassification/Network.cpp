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
		weights[i - 1] = Network::DMatrix::Random(layerSizes[i], layerSizes[i - 1]);
		biases[i - 1]  = Network::DVectorV::Random(layerSizes[i], 1);

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
	std::ofstream xout("out.txt");
	for (int i = 0; i < 200; i++)
		xout << a(i) << ' ';
	xout << "---------\n";
	std::getchar();
	for (int i = 0; i < NumLayers() - 1; i++)
	{
		DVectorV v = weights[i] * a + biases[i];
		a = v.unaryExpr(&Math::Sigmoid).eval();
	}
	return a;
}

void Network::SGD(DataSet &dataSet, int epochs, int batchSize, double learningRate)
{
	while (epochs--)
	{
		std::cout << "Epochs left: " << epochs << '\n';
		dataSet.Shuffle(0, DataSet::TRAINING_COUNT);
		for (int t = 0; t < DataSet::TRAINING_COUNT; t += batchSize)
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

	std::vector<int> counter(10);

	for (int i = batchStart; i < batchStart + batchSize; i++)
	{
		Backprop(batch, i, batch._labels[i]);
		counter[batch._labels[i]]++;

		for (int j = 0; j < NumLayers() - 1; j++)
		{
			batchNablaW[j] += nablaW[j];
			batchNablaB[j] += nablaB[j];
		}
	}

	//std::cout << "Batch set:\n";
	//for (int i = 0; i < 10; i++)
		//std::cout << i << " -> " << counter[i] << '\n';
	//std::getchar();

	for (int i = 0; i < NumLayers() - 1; i++)
	{
		weights[i] -= (learningRate / batchSize) * batchNablaW[i];
		biases[i] -= (learningRate / batchSize) * batchNablaB[i];
	}
}

Network::DVectorV CostDerivative(const Network::DVectorV &networkOut,
	const Network::DVectorV &expectedOut)
{
	Network::DVectorV ret = networkOut - expectedOut;
	//Network::DVectorV ret = networkOut;
	//for (int i = 0; i < networkOut.rows(); i++)
		//ret(i) = (networkOut(i) - expectedOut(i)) / ((1 - networkOut(i)) * networkOut(i));
	return ret;
}

double Cost(const Network::DVectorV &networkOut,
	const Network::DVectorV &expectedOut)
{
	double ret(0);
	//for (int i = 0; i < networkOut.rows(); i++)
		//ret += ((networkOut(i) - expectedOut(i)).transpose() * (networkOut(i) - expectedOut(i)));
	return ret;
}

void Network::Backprop(const DataSet &batch, 
						int inputIndex, 
						unsigned char output)
{
	std::vector<Network::DMatrix> activations(NumLayers());
	std::vector<Network::DVectorV> zs;

	activations[0] = batch.GetInputVector(inputIndex);

	for (int i = 0; i < NumLayers() - 1; i++)
	{
		zs.push_back((weights[i] * activations[i]) + biases[i]);
		activations[i + 1] = zs.back().unaryExpr(&Math::Sigmoid);
	}

	Network::DVectorV cd = CostDerivative(activations.back(), batch.ToVector(output));

	/* Backpropagation */
	Network::DVectorV delta = cd.cwiseProduct(zs.back().unaryExpr(&Math::SigmoidPrimeUn));

	nablaB.back() = delta;
	nablaW.back() = (delta * activations[activations.size() - 2].transpose()).eval();

	// i = 0
	for (int i = NumLayers() - 3; i > -1; i--)
	{
		auto wtd = weights[i + 1].transpose() * delta;
		delta = wtd.cwiseProduct(zs[i].unaryExpr(&Math::SigmoidPrimeUn));
		nablaB[i] = delta;
		nablaW[i] = delta * activations[i].transpose();
	}

}
