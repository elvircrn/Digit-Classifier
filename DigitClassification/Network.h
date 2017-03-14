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
	typedef Eigen::MatrixXd                                      		DMatrix;
	typedef std::vector<DMatrix>										DTensor;

	std::vector<int>         layerSizes;
	std::vector<DVectorV>    biases;
	std::vector<DMatrix>	 weights;

	Network();
	Network(const std::vector<int> &);
	~Network();

	int NumLayers() const;
	Network::DVectorV FeedForward(const Network::DVectorV &a) const;
	void PrintMaxLayers() const;
	void SGD(DataSet&, int, int, double);
	void UpdateMiniBatch(const DataSet&, int, int, double);
	void Backprop(const DataSet &batch,
				 int inputIndex,
				 unsigned char output);
protected:
	std::vector<DMatrix> nablaW;
	std::vector<DVectorV> nablaB;
	std::vector<DMatrix> batchNablaW;
	std::vector<DVectorV> batchNablaB;
};

