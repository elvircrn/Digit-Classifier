#include "stdafx.h"
#include "DataSetParser.h"

using std::vector;

DataSet DataSetParser::ReadFERSet(const string & filepath) const
{
	std::ifstream file(filepath);
	DataSet dataSet;

	if (!file.is_open())
		return dataSet;

	dataSet.SetImageHeight(48);
	dataSet.SetImageWidth(48);

	int dataSize;
	std::string buff, usage;
	vector<unsigned char> data, labels;
	data.reserve(1e5);
	labels.reserve(4 * 1e5);

	// Skip header
	std::getline(file, buff);

	unsigned char c, emotion;
	int pixel;

	vector<unsigned char> features(48 * 48);
	while (!file.eof())
	{
		emotion = file.get() - '0';

		// Skip comma after label
		file.get();

		for (int j = 0; j < 48 * 48; j++)
		{
			file >> pixel;
			features[j] = pixel;
		}

		file.get();
		std::getline(file, usage);

		for (auto& pixel : features)
			data.emplace_back(pixel);
		labels.emplace_back(emotion);
	}

	assert(data[0] == 70 && data[1] == 80);

	dataSet.SetImageCount(labels.size());
	dataSet.TrainingCount(0.8 * labels.size());
	dataSet.ValidationCount(labels.size() - 0.8 * labels.size());
	dataSet.SetData(data);
	dataSet.SetLabels(labels);
	dataSet.CategoryCount(7);
	
	assert(dataSet[0].first[0] == 70 && dataSet[0].first[1] == 80);

	return dataSet;
}
