#include "stdafx.h"
#include <Eigen/Core>

#include "DataSet.h"
#include "Network.h"
#include "Tester.h"
#include "DataSetParser.h"

using namespace Eigen;

int main()
{
	std::ios_base::sync_with_stdio(false);

	DataSet dataSet = DataSetParser().ReadFERSet("Data/fer2013.csv");
	std::cout << "Finished parsing";

	Network net = Network({ dataSet.PixelCount(), 50, 60, 40, dataSet.CategoryCount() }, 
		[](const Network::DVectorV &a, 
			const Network::DVectorV &b, 
			const Network &net) -> Network::DVectorV 
		{ 
			return a - b; 
		}, 0.1);
	net.SGD(dataSet, 60, 10, 3.0);
	Tester::Analyze(dataSet, net);

	std::cout << "\nFinished...\n";
	std::getchar();

	return 0;
}



