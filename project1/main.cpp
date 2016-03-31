#include <opencv2/opencv.hpp>
#include <iostream>

using namespace cv;
using namespace std;

const double angle(Point ptA, Point ptO, Point ptB)
{
	double dxA = ptA.x - ptO.x;
	double dyA = ptA.y - ptO.y;
	double dxB = ptB.x - ptO.x;
	double dyB = ptB.y - ptO.y;
	return (dxA*dxB + dyA*dyB) / sqrt((dxA*dxA + dyA*dyA)*(dxB*dxB + dyB*dyB));
}

const double distance(Point ptA, Point ptB)
{
	return sqrt((ptA.x - ptB.x)*(ptA.x - ptB.x) + (ptA.y - ptB.y)*(ptA.y - ptB.y));
}


int main(int argc, char** argv)
{	
	Mat M = imread(argv[1], 1);
	//Mat M = imread("tstImg02.jpg", 1);
	if (M.empty())
	{
		cout << "Can't open image." << endl;
		waitKey(0);
		return 0;
	}

	Mat pyr;
	Mat M_pyred;
	pyrDown(M, pyr);
	pyrUp(pyr, M_pyred);

	Mat M_smoothed;
	medianBlur(M_pyred, M_smoothed, 3);
	imshow("imgSmoothed", M_smoothed);

	Mat imgHSV;
	vector<Mat> hsvSplit;
	cvtColor(M_smoothed, imgHSV, COLOR_BGR2HSV);
	int height = imgHSV.size().height;
	int width = imgHSV.size().width;

	int iLowH = 90;
	int iHighH = 135;
	int iLowS = 64;
	int iHighS = 255;
	int iLowV = 32;
	int iHighV = 255;

	//Create trackbars in "Control" window  
	/*namedWindow("Control", CV_WINDOW_AUTOSIZE); //create a window called "Control"  
	cvCreateTrackbar("LowH", "Control", &iLowH, 179); //Hue (0 - 179)  
	cvCreateTrackbar("HighH", "Control", &iHighH, 179);

	cvCreateTrackbar("LowS", "Control", &iLowS, 255); //Saturation (0 - 255)  
	cvCreateTrackbar("HighS", "Control", &iHighS, 255);

	cvCreateTrackbar("LowV", "Control", &iLowV, 255); //Value (0 - 255)  
	cvCreateTrackbar("HighV", "Control", &iHighV, 255);*/

	split(imgHSV, hsvSplit);
	equalizeHist(hsvSplit[2], hsvSplit[2]);		//直方图均衡
	merge(hsvSplit, imgHSV);

	Mat imgThresholded;
	inRange(imgHSV, Scalar(iLowH, iLowS, iLowV), Scalar(iHighH, iHighS, iHighV), imgThresholded); //Threshold the image  
		
	Mat imgColorArea;
	morphologyEx(imgThresholded, imgColorArea, MORPH_CLOSE, NULL);
	imshow("imgBlueArea",imgColorArea);

	vector<Mat> vImgOriginRGB;
	split(M_pyred, vImgOriginRGB);
	//imshow("originB", vImgOriginRGB[0]);
	//imshow("originG", vImgOriginRGB[1]);
	//imshow("originR", vImgOriginRGB[2]);
	int N = 11;
	vector<vector<Point>> filteredContours;
	for (int c = 0; c < 3; c++)
	{
		Mat imgSglColor;
		Mat imgSglClrThd;
		vImgOriginRGB[c].copyTo(imgSglColor);
		for (int l = 0; l < N; l++)
		{
			if (l == 0)
			{
				imgColorArea.copyTo(imgSglClrThd);
				morphologyEx(imgSglClrThd, imgSglClrThd, MORPH_CLOSE, NULL,Point(-1,-1),3);
				imshow("imgTmp", imgSglClrThd);
			}
			else
			{
				threshold(imgSglColor, imgSglClrThd, l * 255 / N, 255, THRESH_BINARY);
			}

			Mat imgContours;
			M.copyTo(imgContours);
			vector<vector<Point>> contours;
			findContours(imgSglClrThd, contours, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE);
			for (int idx = 0; idx < contours.size(); idx++)
			{
				Scalar color(rand() & 255, rand() & 255, rand() & 255);
				drawContours(imgContours, contours, idx, color, 2);
			}
			imshow("Contours", imgContours);
			waitKey(0);

			for (int idx = 0; idx < contours.size(); idx++)
			{
				vector<Point> result;
				approxPolyDP(contours[idx], result, arcLength(contours[idx], true)*0.02, true);
				if (result.size() == 4 && fabs(contourArea(result))>1000 && fabs(contourArea(result))<30000 && isContourConvex(result))
				{
					double maxCos = 0;
					for (int i = 2; i < 6; i++)
					{
						double cosine = fabs(angle(result[i % 4], result[(i - 1)%4], result[(i - 2)%4]));
						maxCos = max(maxCos, cosine);
					}
					if (maxCos < 0.3)		//直角
					{
						Mat RecMask(M.size(), CV_8UC1, 255);
						polylines(RecMask, result, true, 0);
						
						Point innerPoint((result[0].x + result[2].x) / 2, (result[0].y + result[2].y) / 2);
						floodFill(RecMask, innerPoint, 0);

						int colorCnt = 0;
						for (int i=0; i < RecMask.rows; i++)
							for (int j=0; j < RecMask.cols; j++)
							{
								if (RecMask.at<uchar>(i, j) == 0 && imgColorArea.at<uchar>(i, j) == 255)
									colorCnt += 1;
							}
						double colorRatio = colorCnt / fabs(contourArea(result));

						double RecHeight = distance(result[0], result[1]);
						double RecWidth = distance(result[1], result[2]);
						if (RecHeight > RecWidth)
							swap(RecHeight, RecWidth);
						double WHRatio = double(RecWidth) / RecHeight;

						if (WHRatio > 2 && WHRatio<5 && colorRatio>0.6)
						{
							filteredContours.push_back(result);
							cout << "GET" << endl;
							imshow("RecMask", RecMask);
							//waitKey(0);
						}	//if filter*3
					}	//if filter*2
				}	//if filter
			}	//contours
		}	//layer
	}	//color
	Mat imgFilteredContours;
	M.copyTo(imgFilteredContours);
	if (filteredContours.size() == 0)
	{
		cout << "NO Filtered Contours!" << endl;
		return 0;
	}
	for (int idx = 0; idx < filteredContours.size(); idx++)
	{
		Scalar color(rand() & 255, rand() & 255, rand() & 255);
		drawContours(imgFilteredContours, filteredContours, idx, color, 2);
	}
	imshow("FilteredContours", imgFilteredContours);

	double maxArea = 0;
	int maxAreaIdx = 0;
	for (int i = 0; i < filteredContours.size(); i++)
	{
		double area = fabs(contourArea(filteredContours[i]));
		if (area > maxArea)
		{
			maxArea = area;
			maxAreaIdx = i;
		}
	}

	vector<Point> chosenContour = filteredContours[maxAreaIdx];
	Mat imgChosenContour;
	M.copyTo(imgChosenContour);
	polylines(imgChosenContour, chosenContour, true, Scalar(0, 0, 255), 2);
	imshow("imgChosenContour", imgChosenContour);
	waitKey(0);

	return 0;
}
