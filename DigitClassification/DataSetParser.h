#include "stdafx.h"

#include <iostream>
#include <string>
#include <fstream>

#include "DataSet.h"

using std::string;

class DataSetParser
{
public:
	DataSet ReadFERSet(const string &filepath) const;
};