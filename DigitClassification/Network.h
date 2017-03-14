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
	double lambda;
	
public:
	typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>	    DVectorV;
	typedef Eigen::MatrixXd                                      		DMatrix;
	typedef std::vector<DMatrix>										DTensor;
	typedef std::function<Network::DVectorV(const Network::DVectorV &networkOut,
		const Network::DVectorV & expectedOut, const Network &net)>     FnCost;

	std::vector<int>         layerSizes;
	std::vector<DVectorV>    biases;
	std::vector<DMatrix>	 weights;
	
	Network::FnCost CostDerivative;

	Network();
	Network(const std::vector<int> &);
	Network(const std::vector<int> &, FnCost, double lambda = 0);
	~Network();

	int NumLayers() const;
	Network::DVectorV FeedForward(const Network::DVectorV &a) const;
	void SGD(DataSet&, int, int, double);
	void PrintMaxLayers() const;
	void UpdateMiniBatch(const DataSet&, int, int, double);
	void Backprop(const DataSet &batch,
				 int inputIndex,
				 unsigned char output);
	double GetSum() const;
	bool IsRegularized() const;

protected:
	std::vector<DMatrix>     nablaW;
	std::vector<DVectorV>	 nablaB;
	std::vector<DMatrix>     batchNablaW;
	std::vector<DVectorV>    batchNablaB;
};

