#include "stdafx.h"
#include "Tester.h"


Tester::Tester()
{
}


Tester::~Tester()
{
}

int Tester::GetConclusion(const Network::DVectorV &v)
{
	double x = v(0);
	int ind = 0;
	for (int i = 1; i < v.rows(); i++)
	{
		std::cout << v(i) << '\n';
		if (x < v(i))
		{
			x = v(i);
			ind = i;
		}
	}
	std::cout << "\n\n";
	std::getchar();
	return ind;
}

void Tester::Analyze(const DataSet &testSet, const Network &network)
{
	int passed = 0;
	for (int i = DataSet::TRAINING_COUNT; i < DataSet::TOTAL_IMAGES_COUNT; i++)
	{
		int conclusion = Tester::GetConclusion(network.FeedForward(testSet.ToVector(testSet[i].first)));
		//std::cout << conclusion << ' ' << testSet.GetLabel(i)<< '\n';
		passed += (conclusion == testSet.GetLabel(i));
	}
	std::cout << "Accuracy: " << (passed / (double)DataSet::VALIDATION_COUNT) * 100.0 << "%\n";
}
