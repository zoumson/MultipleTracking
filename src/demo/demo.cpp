
#include<opencv2/core/core.hpp>
#include<opencv2/highgui/highgui.hpp>
#include<opencv2/imgproc/imgproc.hpp>
#include<iostream>
#include "DrawWithBlob.h"
#include "UseBlob.h"
#include "Blob.h"

int main(int argc, char const *argv[])
{

    cv::VideoCapture capVideo;

    cv::Mat imgFrame1;
    cv::Mat imgFrame2;

    std::vector<za::Blob> blobs;

   cv::String keys =
        "{v video |<none>           | video path}"                                                                                                                                                                                                                     
        "{help h usage ?    |      | show help message}";      
  
    cv::CommandLineParser parser(argc, argv, keys);
    parser.about("Car Counting");
    if (parser.has("help")) 
    {
        parser.printMessage();
        return 0;
    }
    
    cv::String videoPath = parser.get<cv::String>("video"); 

    if (!parser.check()) 
    {
        parser.printErrors();
        return -1;
    }

    capVideo.open(videoPath);

    if (!capVideo.isOpened()) 
    {                                                 
        std::cout << "error reading video file\n";                       
        return(-1);                                                              
    }

    if (capVideo.get(cv::CAP_PROP_FRAME_COUNT) < 2) 
    {
        std::cout << "error: video file must have at least two frames\n";
 
        return(-1);
    }

    capVideo.read(imgFrame1);
    capVideo.read(imgFrame2);

    char chCheckForEscKey = 0;

    bool blnFirstFrame = true;

    int frameCount = 2;

    while (capVideo.isOpened() && chCheckForEscKey != 27) 
    {

        std::vector<za::Blob> currentFrameBlobs;

        cv::Mat imgFrame1Copy = imgFrame1.clone();
        cv::Mat imgFrame2Copy = imgFrame2.clone();

        cv::Mat imgDifference;
        cv::Mat imgThresh;

        cv::cvtColor(imgFrame1Copy, imgFrame1Copy, cv::COLOR_BGR2GRAY);
        cv::cvtColor(imgFrame2Copy, imgFrame2Copy, cv::COLOR_BGR2GRAY);

        cv::GaussianBlur(imgFrame1Copy, imgFrame1Copy, cv::Size(5, 5), 0);
        cv::GaussianBlur(imgFrame2Copy, imgFrame2Copy, cv::Size(5, 5), 0);

        cv::absdiff(imgFrame1Copy, imgFrame2Copy, imgDifference);

        cv::threshold(imgDifference, imgThresh, 30, 255.0, cv::THRESH_BINARY);

        cv::imshow("imgThresh", imgThresh);

        cv::Mat structuringElement3x3 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3));
        cv::Mat structuringElement5x5 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(5, 5));
        cv::Mat structuringElement7x7 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(7, 7));
        cv::Mat structuringElement9x9 = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(9, 9));

        /*
        cv::dilate(imgThresh, imgThresh, structuringElement7x7);
        cv::erode(imgThresh, imgThresh, structuringElement3x3);
        */

        cv::dilate(imgThresh, imgThresh, structuringElement5x5);
        cv::dilate(imgThresh, imgThresh, structuringElement5x5);
        cv::erode(imgThresh, imgThresh, structuringElement5x5);

        
        cv::Mat imgThreshCopy = imgThresh.clone();

        std::vector<std::vector<cv::Point> > contours;

        cv::findContours(imgThreshCopy, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

        za::drawAndShowContours(imgThresh.size(), contours, "imgContours");

        std::vector<std::vector<cv::Point> > convexHulls(contours.size());

        for (unsigned int i = 0; i < contours.size(); i++) 
        {
            cv::convexHull(contours[i], convexHulls[i]);
        }

        za::drawAndShowContours(imgThresh.size(), convexHulls, "imgConvexHulls");

        for (auto &convexHull : convexHulls) 
        {
            za::Blob possibleBlob(convexHull);

            if (possibleBlob.currentBoundingRect.area() > 100 &&
                possibleBlob.dblCurrentAspectRatio >= 0.2 &&
                possibleBlob.dblCurrentAspectRatio <= 1.25 &&
                possibleBlob.currentBoundingRect.width > 20 &&
                possibleBlob.currentBoundingRect.height > 20 &&
                possibleBlob.dblCurrentDiagonalSize > 30.0 &&
                (cv::contourArea(possibleBlob.currentContour) / (double)possibleBlob.currentBoundingRect.area()) > 0.40) 
                {
                    currentFrameBlobs.push_back(possibleBlob);
                }
        }

        za::drawAndShowContours(imgThresh.size(), currentFrameBlobs, "imgCurrentFrameBlobs");

        if (blnFirstFrame == true) 
        {
            for (auto &currentFrameBlob : currentFrameBlobs) 
            {
                blobs.push_back(currentFrameBlob);
            }
        }
        else 
        {
            za::matchCurrentFrameBlobsToExistingBlobs(blobs, currentFrameBlobs);
        }

        za::drawAndShowContours(imgThresh.size(), blobs, "imgBlobs");

        // get another copy of frame 2 since we changed the previous frame 2 copy in the processing above
        imgFrame2Copy = imgFrame2.clone();        

        za::drawBlobInfoOnImage(blobs, imgFrame2Copy);

        cv::imshow("imgFrame2Copy", imgFrame2Copy);

        // uncomment this line to go frame by frame for debugging
        //cv::waitKey(0);                 

        // now we prepare for the next iteration

        currentFrameBlobs.clear();

        // move frame 1 up to where frame 2 is
        imgFrame1 = imgFrame2.clone();           

        if ((capVideo.get(cv::CAP_PROP_POS_FRAMES) + 1) < capVideo.get(cv::CAP_PROP_FRAME_COUNT)) 
        {
            capVideo.read(imgFrame2);
        }
        else 
        {
            std::cout << "end of video\n";
            break;
        }

        blnFirstFrame = false;
        frameCount++;
        chCheckForEscKey = cv::waitKey(1);
    }

    if (chCheckForEscKey != 27) 
    {              
         // if the user did not press esc (i.e. we reached the end of the video)
        cv::waitKey(0);                       
    }
    

    return(0);
}
