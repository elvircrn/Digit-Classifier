#include "stdafx.h"

#include <Eigen/Core>

#include "DataSet.h"
#include "Network.h"

using namespace Eigen;

int main()
{
	std::ios_base::sync_with_stdio(false);
	DataSet dataSet(600);

	dataSet.Load(DataSet::TRAINING_IMAGES, DataSet::TRAINING_LABELS, 600);

	std::cout << "Press any key to continue...\n";
	std::getchar();
    return 0;
}
