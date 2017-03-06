#include <Eigen/Core>

#include "DataSet.h"
#include <tuple>
#include <functional>
#include <Eigen/Core>

#pragma once
class Network
{
private:

protected:
	int _numLayers;
	
public:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>	    DVectorV;
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>		DMatrix;
	typedef std::vector<DMatrix>										DTensor;

	std::vector<int>         layerSizes;
	std::vector<DVectorV>    biases;
	std::vector<DMatrix>	 weights;

	Network();
	Network(const std::vector<int> &);
	~Network();

	int NumLayers();
	auto FeedForward(Network::DMatrix &a);
	void SGD(DataSet&, int, double);
	void UpdateMiniBatch(const DataSet&, int, int, double);
	std::pair<std::vector<Network::DVectorV>, std::vector<Network::DMatrix>> Backprop(const DataSet&, unsigned char*, unsigned char);
};

