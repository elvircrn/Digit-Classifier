#pragma once

#include "stdafx.h"
#include <Eigen/Core>


class DataSet
{
private:
	static inline int ReverseInt(int);

protected:
	int _dataSize, _imgWidth, _imgHeight, _imgCount;


	void Init();
	void LoadImages(std::string, int);
	void LoadLabels(std::string);
	void Swap(int, int);

public:
	std::vector<unsigned char> _data, _labels;

	static constexpr int TRAINING_COUNT		= 50000;
	static constexpr int VALIDATION_COUNT   = 10000;
	static constexpr int TOTAL_IMAGES_COUNT = 60000;
	
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
	#pragma endregion

	void Load(std::string, std::string, int = TOTAL_IMAGES_COUNT);
	unsigned char GetPixel(int, int) const;
	void Shuffle(int, int);
	Eigen::Matrix<double, Eigen::Dynamic, 1> ToVector(unsigned char) const;
	std::vector<DataSet> Split(const std::vector<int> &parts) const;
	Eigen::Matrix<double, Eigen::Dynamic, 1> GetInputVector(int) const;

	DataSet::Data operator[] (const int index) const;
};
