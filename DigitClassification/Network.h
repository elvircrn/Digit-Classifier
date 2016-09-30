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

	int NumLayers();
	auto FeedForward(Network::DMatrix &a);
	void SGD(DataSet&, int, double);
	void UpdateMiniBatch(const DataSet&, int, int, double);
	std::pair<Eigen::Matrix<Eigen::Matrix<double, Eigen::Dynamic, 1>, 1, Eigen::Dynamic>,
		Eigen::Matrix<Network::DMatrix, Eigen::Dynamic, Eigen::Dynamic>> Backprop(unsigned char*, unsigned char);


};

