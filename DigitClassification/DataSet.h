#pragma once

#include "stdafx.h"
#include <Eigen/Core>

using std::vector;

class DataSet
{
private:
	static inline int ReverseInt(int);

protected:
	int _dataSize, _imgWidth, _imgHeight, _imgCount;
	int _trainingCount, _validationCount;
	int _categoryCount;


	void Init();
	void LoadImages(std::string, int);
	void LoadLabels(std::string);
	void Swap(int, int);

public:
	std::vector<unsigned char> _data, _labels;

	typedef std::pair<std::vector<unsigned char>::const_iterator, 
					  std::vector<unsigned char>::const_iterator> Data;

	static const std::string TRAINING_LABELS;
	static const std::string TRAINING_IMAGES;

	DataSet();
	DataSet(int);

	#pragma region Getters And Setters
	void SetImageWidth(int);
	void SetImageHeight(int);
	void SetImageCount(int);
	int ImageCount() const;
	int ImageHeight() const;
	int ImageWidth() const;
	int DataSize() const;
	int PixelCount() const;
	int GetLabel(int) const;
	int TrainingCount() const { return _trainingCount; }
	int ValidationCount() const { return _validationCount; }
	void TrainingCount(int __trainingCount) { _trainingCount = __trainingCount; }
	void ValidationCount(int __validationCount) { _validationCount = __validationCount; }

	void SetData(const vector<unsigned char> &__data) { _data = __data; }
	void SetLabels(const vector<unsigned char> &__labels) { _labels = __labels; }

	int CategoryCount() const { return _categoryCount; }
	void CategoryCount(int __categoryCount) { _categoryCount = __categoryCount; }
	#pragma endregion

	void Load(std::string, std::string, int = 60000);
	unsigned char GetPixel(int, int) const;
	void Shuffle(int, int);
	Eigen::Matrix<double, Eigen::Dynamic, 1> ToVector(unsigned char) const;
	std::vector<DataSet> Split(const std::vector<int> &parts) const;
	Eigen::Matrix<double, Eigen::Dynamic, 1> GetInputVector(int) const;

	DataSet::Data operator[] (const int index) const;

	friend class DataSetParser;
};
