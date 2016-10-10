#pragma once

#include "stdafx.h"
#include <Eigen/Core>

class DataSet
{
private:
	void _dispose();
	static inline int ReverseInt(int);

protected:
	int _dataSize, _imgWidth, _imgHeight, _imgCount;
	unsigned char *_data;
	unsigned char *_labels;

	void Init();
	void LoadImages(std::string, int);
	void LoadLabels(std::string);

public:
	typedef std::pair<unsigned char*, unsigned char*> Data;

	static const std::string TRAINING_LABELS;
	static const std::string TRAINING_IMAGES;

	DataSet();
	DataSet(int);
	~DataSet();

#pragma region Getters And Setters
	void SetImageWidth(int);
	void SetImageHeight(int);
	void SetImageCount(int);
	int ImageCount() const;
	int ImageHeight() const;
	int ImageWidth() const;
	int DataSize() const;
#pragma endregion

	void Load(std::string, std::string, int);
	unsigned char GetPixel(int, int, int) const;
	void Shuffle();
	Eigen::Matrix<double, Eigen::Dynamic, 1> ToVector(unsigned char) const;

	DataSet::Data operator[] (const int index) const;
};
