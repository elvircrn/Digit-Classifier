#pragma once

#include "Network.h"
#include "DataSet.h"

class Tester
{
private:
	Tester();
	~Tester();

	static int GetConclusion(const Network::DVectorV &v);

public:
	static void Analyze(const DataSet &testSet, const Network &network);
};

