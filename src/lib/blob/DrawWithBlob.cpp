#include "DrawWithBlob.h"

namespace za {


///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<std::vector<cv::Point> > contours, std::string strImageName) 
{
    cv::Mat image(imageSize, CV_8UC3, za::SCALAR_BLACK);

    cv::drawContours(image, contours, -1, za::SCALAR_WHITE, -1);

    cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawAndShowContours(cv::Size imageSize, std::vector<za::Blob> blobs, std::string strImageName) 
{

    cv::Mat image(imageSize, CV_8UC3, za::SCALAR_BLACK);

    std::vector<std::vector<cv::Point> > contours;

    for (auto &blob : blobs) 
    {
        if (blob.blnStillBeingTracked == true) 
        {
            contours.push_back(blob.currentContour);
        }
    }

    cv::drawContours(image, contours, -1, za::SCALAR_WHITE, -1);

    cv::imshow(strImageName, image);
}

///////////////////////////////////////////////////////////////////////////////////////////////////
void drawBlobInfoOnImage(std::vector<za::Blob> &blobs, cv::Mat &imgFrame2Copy) 
{

    for (unsigned int i = 0; i < blobs.size(); i++) 
    {

        if (blobs[i].blnStillBeingTracked == true) 
        {
            cv::rectangle(imgFrame2Copy, blobs[i].currentBoundingRect, za::SCALAR_RED, 2);

            int intFontFace = cv::FONT_HERSHEY_SIMPLEX;
            double dblFontScale = blobs[i].dblCurrentDiagonalSize / 60.0;
            int intFontThickness = (int)std::round(dblFontScale * 1.0);

            cv::putText(imgFrame2Copy, std::to_string(i), blobs[i].centerPositions.back(), intFontFace, dblFontScale, za::SCALAR_GREEN, intFontThickness);
        }
    }
}
}