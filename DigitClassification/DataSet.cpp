#include "stdafx.h"

#include "DataSet.h"

const std::string DataSet::TRAINING_IMAGES = "train-images.idx3-ubyte";
const std::string DataSet::TRAINING_LABELS = "train-labels.idx1-ubyte";

void DataSet::_dispose()
{
	if (_data != nullptr)
		delete[] _data;

	_data = nullptr;

	if (_labels != nullptr)
		delete[] _labels;

	_labels = nullptr;
}

#pragma region Data Read

int DataSet::ReverseInt(int i)
{
	unsigned char ch1, ch2, ch3, ch4;
	ch1 = i & 255;
	ch2 = (i >> 8) & 255;
	ch3 = (i >> 16) & 255;
	ch4 = (i >> 24) & 255;
	return((int)ch1 << 24) + ((int)ch2 << 16) + ((int)ch3 << 8) + ch4;
}

void DataSet::LoadImages(std::string imagesLocation, int numberOfImages)
{
	std::ifstream file(imagesLocation, std::ios::binary);
	file.open(imagesLocation, std::ios::binary);
	if (file.is_open())
	{
		file.clear();
		file.seekg(0, std::ios::beg);

		int magic = 0, actualCount = 0;

		file.read((char*)&magic, sizeof(magic));

		file.read((char*)&actualCount, sizeof(actualCount));
		actualCount = ReverseInt(actualCount);

		SetImageCount(std::min(actualCount, numberOfImages));

		file.read((char*)&_imgHeight, sizeof(_imgHeight));
		_imgHeight = ReverseInt(_imgHeight);

		file.read((char*)&_imgWidth, sizeof(_imgWidth));
		_imgWidth = ReverseInt(_imgWidth);

		SetImageHeight(_imgHeight);
		SetImageWidth(_imgWidth);

		_data = new unsigned char[_imgCount * _imgHeight * _imgWidth];

		for (int i = 0; i < GetImageCount(); i++)
		{
			for (int r = 0; r < _imgHeight; r++)
			{
				for (int c = 0; c < _imgWidth; c++)
				{
					unsigned char temp = 0;
					file.read((char*)&temp, sizeof(temp));
					_data[i * (GetImageHeight() * GetImageWidth()) + r * GetImageWidth() + c] = (unsigned char)temp;
				}
			}
		}
	}
	else
	{
		throw std::exception(("Unable to open file " + imagesLocation + ".").c_str());
	}
}

void DataSet::LoadLabels(std::string labelsLocation)
{
	std::ifstream file(labelsLocation, std::ios::binary);
	file.open(labelsLocation, std::ios::binary);

	if (file.is_open())
	{
		file.clear();
		file.seekg(0, std::ios::beg);

		int magic, actualCount;

		file.read((char*)&magic, sizeof(magic));

		file.read((char*)&actualCount, sizeof(actualCount));

		_labels = new unsigned char[GetImageCount()];

		unsigned char temp;
		for (int i = 0; i < GetImageCount(); i++)
		{
			file.read((char*)&temp, sizeof(temp));
			_data[i] = (unsigned char)temp;
		}
	}
	else
	{
		throw std::exception(("Unable to open file " + labelsLocation + ".").c_str());
	}
}

#pragma endregion

void DataSet::Init()
{
	_data = nullptr;
	_labels = nullptr;
}

DataSet::DataSet()
{
	Init();
}

DataSet::DataSet(int __dataSize)
{
	Init();
	_dataSize = __dataSize;
}

DataSet::~DataSet()
{
	_dispose();
}

void DataSet::SetImageWidth(int imgWidth)
{
	_imgWidth = imgWidth;
}

void DataSet::SetImageHeight(int imgHeight)
{
	_imgHeight = imgHeight;
}

void DataSet::SetImageCount(int imgCount)
{
	_imgCount = imgCount;
}

int DataSet::GetImageCount()
{
	return _imgCount;
}

int DataSet::GetImageHeight()
{
	return _imgHeight;
}

int DataSet::GetImageWidth()
{
	return _imgWidth;
}

void DataSet::Load(std::string imagesLocation, std::string labelsLocation, int numberOfImages)
{
	_dispose();
	LoadImages(imagesLocation, numberOfImages);
	LoadLabels(labelsLocation);
}

unsigned char DataSet::GetPixel(int img, int h, int w)
{
	return _data[img * (GetImageHeight() * GetImageWidth()) + h * GetImageWidth() + w];
}
