#include <Eigen/Core>

#include "DataSet.h"
#include <tuple>
#include <functional>

#pragma once
class Network
{
private:

protected:
	int _numLayers;
	
public:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> DMatrix;
	typedef Eigen::Matrix<double, Eigen::Dynamic, 1> DVectorV;
	typedef Eigen::Matrix<DMatrix, Eigen::Dynamic, Eigen::Dynamic> DTensor;

	std::vector<int> layerSizes;
	Eigen::Matrix<Eigen::Matrix<double, Eigen::Dynamic, 1>, 1, Eigen::Dynamic> biases;
	Eigen::Matrix<Network::DMatrix, Eigen::Dynamic, Eigen::Dynamic> weights;

	Network();
	Network(const std::vector<int> &);
	~Network();

	int NumLayers();
	auto FeedForward(Network::DMatrix &a);
	void SGD(DataSet&, int, double);
	void UpdateMiniBatch(const DataSet&, int, int, double);
	std::pair<Eigen::Matrix<Network::DVectorV, Eigen::Dynamic, 1>, DTensor> Backprop(const DataSet&, unsigned char*, unsigned char);
};

