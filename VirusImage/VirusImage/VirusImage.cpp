/** Date : 2017.08.29 
 *  Author : jeon.tae.kyeong
 *  내용 : 바이너리 파일을 이미지파일로 변환하는 프로그램
 */
#include<afx.h>
#include<cstdlib>
#include<cmath>
#include<cstringt.h>
#include<cstring>
#include<string.h>
#include<io.h>
#include<iostream>
#include<string>
#include<fstream>
#include<stdio.h>
#include<vector>
#include<utility>
#include<direct.h>
#include<opencv/cv.h>
#include<opencv/highgui.h>

#define MAX_PATH           1000
#define KB                 1024

int ImageSize(size_t binarySize)
{
	size_t width;
	size_t total_Bit=0;

	total_Bit = binarySize*8;
	
	if(total_Bit < 10*KB)
		return 32;
	else if(total_Bit >= 10*KB && total_Bit < 30*KB)
		return 64;
	else if(total_Bit >= 30*KB && total_Bit < 60*KB)
		return 128;
	else if(total_Bit >= 60*KB && total_Bit < 100*KB)
		return 256;
	else if(total_Bit >= 100*KB && total_Bit < 200*KB)
		return 384;
	else if(total_Bit >= 200*KB && total_Bit < 500*KB)
		return 512;
	else if(total_Bit >= 500*KB && total_Bit < 1000*KB)
		return 768;
	else if(total_Bit >= 1000*KB)
		return 1024;
}

void SaveImage(std::string path, std::string fileName)
{
	char SaveFileName[MAX_PATH];

	std::ifstream stream(path, std::ios::binary);
	if(!stream)
	{
		std::cout << "fail to upload the file : " <<path<<std::endl;
		return;
	}
	size_t binarySize=0;
	{
		std::streampos const start = stream.tellg();
		stream.seekg(0, std::ios::end);
		binarySize = stream.tellg() - start;
		stream.seekg(start);
	}
	
	size_t width = 1024;// ImageSize(binarySize);
	size_t height = 1024; //(binarySize/width);
	
	IplImage* pImage = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

	for (size_t line = 0; line < height; ++line)
	{
		stream.read(pImage->imageData + pImage->widthStep * line, width);
		if (stream.eof())
		{
			break;
		}
		else if (!stream)
		{
			std::cout << "fail to read the file : " << path << std::endl;
			return;
		}
	}

	sprintf(SaveFileName, "%s_image.png", fileName.c_str());
	cvSaveImage(SaveFileName, pImage);
	cvReleaseImage(&pImage);
}

int main(void)
{
	_finddata_t fd;
	long handle;
	int result = 1;

	handle = _findfirst("C:\\Users\\JoWookJae\\Desktop\\test\\*.*", &fd);
	if(handle == -1)
	{
		puts("there is no files.");
		return 0;
	}

	while(result != -1)
	{
		std::string path = "C:\\Users\\JoWookJae\\Desktop\\test\\";
		std::string lowpath;
		std::string fileName;
		fileName = fd.name;
		lowpath = path + fileName;
		SaveImage(lowpath, fileName);
		result = _findnext(handle, &fd);
	}

	_findclose(handle);
	std::cout<<"File ends..."<<std::endl;

	return 0;
}
