#include <Eigen/Core>

#include "DataSet.h"
#include <tuple>

#pragma once
class Network
{
private:

protected:
	int _numLayers;
	
public:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> DMatrix;

	std::vector<int> layerSizes;
	Eigen::Matrix<Eigen::Matrix<double, Eigen::Dynamic, 1>, 1, Eigen::Dynamic> biases;
	Eigen::Matrix<Network::DMatrix, Eigen::Dynamic, Eigen::Dynamic> weights;

	Network();
	Network(const std::vector<int> &);
	~Network();

	int GetNumLayers();
	auto FeedForward(Network::DMatrix &a);
	void SGD(DataSet&, int, double);
	void UpdateMiniBatch(DataSet::Data, int, double);
};

