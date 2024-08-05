#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <string>
#include <math.h>
#include "liveportrait.h"

using namespace cv;
using namespace std;


int main()
{
	LivePortraitPipeline mynet;
    
	string imgpath = "/home/wangbo/liveportrait-onnxrun/0.jpg";
	string videopath = "/home/wangbo/liveportrait-onnxrun/d0.mp4";
	
	mynet.execute(imgpath, videopath);

	return 0;
}