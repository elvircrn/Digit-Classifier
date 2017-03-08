#include "stdafx.h"
#include <Eigen/Core>

#include "DataSet.h"
#include "Network.h"
#include "Tester.h"

using namespace Eigen;

int main()
{
	std::ios_base::sync_with_stdio(false);
	DataSet dataSet;

	dataSet.Load(DataSet::TRAINING_IMAGES, DataSet::TRAINING_LABELS);

	Network net = Network({ 784, 30, 10 });

	net.SGD(dataSet, 30, 1.0);

	Tester::Analyze(dataSet, net);

	std::cout << "\nFinished...\n";
	std::getchar();
    return 0;
}

