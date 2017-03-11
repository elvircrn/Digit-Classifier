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
	//std::cout << "conclusion:\n" << v << "\n\n";
	int ind = 0;
	for (int i = 1; i < v.rows(); i++)
	{
		if (x < v(i))
		{
			x = v(i);
			ind = i;
		}
	}
	return ind;
}

void Tester::Analyze(const DataSet &testSet, const Network &network)
{
	int passed = 0;
	for (int i = DataSet::TRAINING_COUNT; i < DataSet::TOTAL_IMAGES_COUNT; i++)
	{
		//std::cout << "input set\n" << testSet.GetInputVector(i) << "\n\n";
		int conclusion = Tester::GetConclusion(network.FeedForward(testSet.GetInputVector(i)));
		passed += (conclusion == testSet.GetLabel(i));
		//std::cout << "(" << conclusion << ", " << testSet.GetLabel(i) << ")\n";
	}
	std::cout << "Accuracy: " << (passed / (double)DataSet::VALIDATION_COUNT) * 100.0 << "%\n";
}
