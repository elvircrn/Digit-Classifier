#include "stdafx.h"
#include <random>

#include <Eigen/Core>

#include "DataSet.h"

const std::string DataSet::TRAINING_IMAGES = "train-images.idx3-ubyte";
const std::string DataSet::TRAINING_LABELS = "train-labels.idx1-ubyte";

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

		_data = std::vector<unsigned char>(_imgCount * _imgHeight * _imgWidth);

		for (int i = 0; i < numberOfImages * _imgHeight * _imgWidth; i++)
		{
			unsigned char temp = 0;
			file.read((char*)&temp, sizeof(temp));
			_data[i] = (unsigned char)temp;
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
	std::vector<int> counter(10);
	_categoryCount = 10;

	if (file.is_open())
	{
		file.clear();
		file.seekg(0, std::ios::beg);

		int magic, actualCount;

		file.read((char*)&magic, sizeof(magic));

		file.read((char*)&actualCount, sizeof(actualCount));

		_labels = std::vector<unsigned char>(ImageCount());

		unsigned char temp;
		for (int i = 0; i < ImageCount(); i++)
		{
			file.read((char*)&temp, sizeof(temp));
			_labels[i] = (unsigned char)temp;
			counter[(int)_labels[i]]++;
		}
		
		std::cout << "Test set:\n";
		for (int i = 0; i < 10; i++)
			std::cout << i << " -> " << counter[i] << '\n';
	}
	else
	{
		throw std::exception(("Unable to open file " + labelsLocation + ".").c_str());
	}
}

void DataSet::Swap(int x, int y)
{
	for (int i = 0; i < PixelCount(); i++)
		std::swap(_data[x + i], _data[y + i]);
	std::swap(_labels[x], _labels[y]);
}

#pragma endregion

void DataSet::Init()
{
	_data.clear();
	_labels.clear();
}

int DataSet::PixelCount() const
{
	return ImageHeight() * ImageWidth();
}

int DataSet::GetLabel(int index) const
{
	return (int)_labels[index];
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

int DataSet::ImageCount() const
{
	return _imgCount;
}

int DataSet::ImageHeight() const
{
	return _imgHeight;
}

int DataSet::ImageWidth() const
{
	return _imgWidth;
}

int DataSet::DataSize() const
{
	return _dataSize;
}

void DataSet::Load(std::string imagesLocation, std::string labelsLocation, int numberOfImages)
{
	LoadImages(imagesLocation, numberOfImages);
	LoadLabels(labelsLocation);
}

unsigned char DataSet::GetPixel(int img, int h) const
{
	return _data[img * (ImageHeight() * ImageWidth()) + h];
}

void DataSet::Shuffle(int start, int end)
{
	std::mt19937 gen;
	std::uniform_int_distribution<> dis;
	gen = std::mt19937(std::random_device()());
	dis = std::uniform_int_distribution<>(start, end - 2);

	for (int i = 0; i < (end - start / 2) / 16; i++)
		Swap(dis(gen), dis(gen));
}

DataSet::Data DataSet::operator[](const int index) const
{
	return{ _data.begin() + index * ImageWidth() * ImageHeight(), _labels.begin() + index };
}

Eigen::Matrix<double, Eigen::Dynamic, 1> DataSet::ToVector(unsigned char x) const
{
	auto result = Eigen::Matrix<double, Eigen::Dynamic, 1>(CategoryCount(), 1);
	result.setZero();
	result(x) = 1;
	return result;
}

// TODO: Implement
// WARNING: Broken!
std::vector<DataSet> DataSet::Split(const std::vector<int>& parts) const
{
	std::vector<DataSet> sets;
	int sum = parts[0], index = 0, x = 0, y = 0;

	for (int i = 0; i < (int)parts.size(); i++)
	{
		for (int j = 0; j < parts[i]; j++)
		{
			sets.push_back(DataSet());
			index++;
		}
	}

	return sets;
}

Eigen::Matrix<double, Eigen::Dynamic, 1> DataSet::GetInputVector(int ind) const
{
	Eigen::Matrix<double, Eigen::Dynamic, 1> ret(PixelCount(), 1);
	for (int i = 0; i < PixelCount(); i++)
		ret(i) = GetPixel(ind, i) / 255.0;
	return ret;
}
