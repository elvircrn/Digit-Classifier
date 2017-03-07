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
	return (ind == 0) ? 10 : ind;
}

void Tester::Analyze(const DataSet &testSet, const Network &network)
{
	int passed = 0;
	for (int i = 500; i < 600; i++)
	{
		int conclusion = Tester::GetConclusion(network.FeedForward((testSet.ToVector(testSet[i].second))));
		passed += (conclusion == *testSet[i].second);
	}
	std::cout << "Accuracy: " << passed << "%\n";
}
