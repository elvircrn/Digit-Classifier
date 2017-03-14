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
	std::vector<int> counter(10);
	for (int i = DataSet::TRAINING_COUNT; i < DataSet::TOTAL_IMAGES_COUNT; i++)
	{
		//std::cout << "input set\n" << testSet.GetInputVector(i) << "\n\n";
		Network::DVectorV output = network.FeedForward(testSet.GetInputVector(i));
		int conclusion = Tester::GetConclusion(output);
		passed += (conclusion == testSet.GetLabel(i));
		counter[conclusion]++;
		//std::cout << "(" << conclusion << ", " << testSet.GetLabel(i) << ")\n";
	}
	for (int i = 0; i < 10; i++)
		std::cout << i << " -> " << counter[i] << '\n';
	std::cout << "Accuracy: " << (passed / (double)DataSet::VALIDATION_COUNT) * 100.0 << "%\n";
}
