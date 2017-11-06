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
	std::vector<int> counter(testSet.CategoryCount());
	for (int i = testSet.TrainingCount(); i < testSet.ImageCount(); i++)
	{
		//std::cout << "input set\n" << testSet.GetInputVector(i) << "\n\n";
		Network::DVectorV output = network.FeedForward(testSet.GetInputVector(i));
		int conclusion = Tester::GetConclusion(output);
		passed += (conclusion == testSet.GetLabel(i));
		counter[conclusion]++;
		//std::cout << "(" << conclusion << ", " << testSet.GetLabel(i) << ")\n";
	}
	for (int i = 0; i < testSet.CategoryCount(); i++)
		std::cout << i << " -> " << counter[i] << '\n';
	std::cout << "Accuracy: " << (passed / (double)testSet.ValidationCount()) * 100.0 << "%\n";
}
