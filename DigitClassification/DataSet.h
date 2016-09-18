#pragma once

#include <vector>
#include <string>

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
	static const std::string TRAINING_LABELS;
	static const std::string TRAINING_IMAGES;

	DataSet();
	DataSet(int);
	~DataSet();

	#pragma region Getters And Setters
	void SetImageWidth(int);
	void SetImageHeight(int);
	void SetImageCount(int);
	int GetImageCount();
	int GetImageHeight();
	int GetImageWidth();
	#pragma endregion

	void Load(std::string, std::string, int);
	unsigned char GetPixel(int, int, int);
};
