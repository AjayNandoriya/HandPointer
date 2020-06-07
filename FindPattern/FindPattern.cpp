#include "pch.h"
#include <opencv2/opencv.hpp>
#include <filesystem>
#include <iostream>
#include <sstream>
#include <string>
#include "StrokeRecognizer.hpp"

struct Setting
{
    std::string baseDirName = "D:\\git\\HandPointer\\dataset\\ds325\\fast_circles";
    std::string outputDirName = "D:\\git\\HandPointer\\dataset";
    std::string fileNameFormat = "%06d_depth.tiff";
    unsigned maxFileCount = 800;
    unsigned depthMin = 100;
    unsigned depthMax = 1300;
    unsigned confidenceMin = 100;
    int erodeKernelSize = 3;

};

const Setting setting;


void GenerateColorMap(size_t N, cv::Mat& colormap, int mapType = cv::COLORMAP_HSV)
{
    if (N == 0)
    {
        N = 1;
    }
    cv::Mat mapInd(N, 1, CV_8U, cv::Scalar(0));
    for (int bid = 0; bid < N; bid++)
    {
        mapInd.at<uchar>(bid) = bid * 255.0 / N;
    }
    cv::applyColorMap(mapInd, colormap, mapType);
}

void FindROIMask(const cv::Mat& img_depth, const cv::Mat& img_confidence, cv::Mat& mask)
{
    cv::Mat mask_depth;
    cv::Mat mask_confidence;
    cv::inRange(img_depth, setting.depthMin, setting.depthMax, mask_depth);
    mask_confidence = img_confidence > setting.confidenceMin;
    cv::bitwise_and(mask_depth, mask_confidence, mask);
    cv::erode(mask, mask, cv::getStructuringElement(cv::MORPH_RECT, cv::Size(setting.erodeKernelSize, setting.erodeKernelSize)));

}
void FilterMeasurementArea(const cv::Mat& img, cv::Mat& img_inner, cv::Mat& mask)
{
    cv::subtract(setting.depthMax, img, img_inner, mask);
}

bool FindFingerTip(cv::Mat img, cv::Point& finger_tip_point)
{
    double maxVal;
    cv::minMaxLoc(img, nullptr, &maxVal, nullptr, &finger_tip_point);
    return (maxVal > 10);
}
std::filesystem::path GetFileName(unsigned id, const std::string postFix = "_depth.tiff")
{
    std::stringstream baseName;
    baseName << std::setfill('0') << std::setw(6) << id << postFix;
    std::filesystem::path file_path(baseName.str());
    return file_path;
}

struct TrajectoryPoint
{
    cv::Point finger_tip;
    unsigned fileID;
    unsigned depth;

    TrajectoryPoint() {}
    TrajectoryPoint(cv::Point p, unsigned id, unsigned depth)
    {
        this->finger_tip = p;
        this->fileID = id;
        this->depth = depth;
    }

    static void DumpToCSV(const std::filesystem::path file_path, const std::vector<TrajectoryPoint>& finger_tip_trajectory)
    {
        std::ofstream csvFile(file_path.string());
        if (!csvFile.is_open())
        {
            std::cout << "Error opening file:" << file_path.string();
            return;
        }
        csvFile << "file ID, u, v, depth\n";
        for (TrajectoryPoint data : finger_tip_trajectory)
        {
            csvFile << data.fileID << "," << data.finger_tip.x << "," << data.finger_tip.y << "," << data.depth <<"\n";
        }
        csvFile.close();
    }
};



std::vector<std::string> trackerTypes = { "BOOSTING", "MIL", "KCF", "TLD", "MEDIANFLOW", "GOTURN", "MOSSE", "CSRT" };
//// create tracker by name
//cv::Ptr<cv::tracker> createTrackerByName(string trackerType)
//{
//    Ptr<tracker> tracker;
//    if (trackerType == trackerTypes[0])
//        tracker = TrackerBoosting::create();
//    else if (trackerType == trackerTypes[1])
//        tracker = TrackerMIL::create();
//    else if (trackerType == trackerTypes[2])
//        tracker = TrackerKCF::create();
//    else if (trackerType == trackerTypes[3])
//        tracker = TrackerTLD::create();
//    else if (trackerType == trackerTypes[4])
//        tracker = TrackerMedianFlow::create();
//    else if (trackerType == trackerTypes[5])
//        tracker = TrackerGOTURN::create();
//    else if (trackerType == trackerTypes[6])
//        tracker = TrackerMOSSE::create();
//    else if (trackerType == trackerTypes[7])
//        tracker = TrackerCSRT::create();
//    else
//    {
//        cout << "Incorrect tracker name" << endl;
//        cout << "Available trackers are: " << endl;
//        for (vector<string>::iterator it = trackerTypes.begin(); it != trackerTypes.end(); ++it)
//            std::cout << " " << *it << endl;
//    }
//    return tracker;
//}

void ProcessSequence(const std::filesystem::path baseDirName)
{
    std::filesystem::path csv_file_name(setting.outputDirName);
    std::filesystem::path trajectory_img_file_name(setting.outputDirName);
    std::filesystem::path trajectory_video_file_name(setting.outputDirName);
    {
        std::string filename = baseDirName.string();
        std::replace(filename.begin(), filename.end(), '\\', '#');
        csv_file_name /= filename + ".csv";
        trajectory_img_file_name /= filename + "_trajectory.png";
        trajectory_video_file_name /= filename + "_trajectory.avi";
    }

    cv::VideoWriter outputVideo;
    bool status = outputVideo.open(trajectory_video_file_name.string(), -1, 60, cv::Size(320, 240));
    if (!outputVideo.isOpened())
    {
        cout << "video file not opened.";
    }
    std::vector<TrajectoryPoint> finger_tip_trajectory;
    finger_tip_trajectory.reserve(setting.maxFileCount);
    cv::Mat trajectory_img, trajectory_video_frame, color_map;
    GenerateColorMap(setting.maxFileCount, color_map);
    for (unsigned fileID = 0; fileID < setting.maxFileCount; fileID++)
    {
        // file sanity check
        const std::filesystem::path depthFileName = baseDirName / GetFileName(fileID, "_depth.tiff");
        const std::filesystem::path confidenceFileName = baseDirName / GetFileName(fileID, "_confidence.tiff");
        const std::filesystem::path resultFileName = baseDirName / GetFileName(fileID, "_result.png");

        if (!std::filesystem::exists(depthFileName) || !std::filesystem::exists(confidenceFileName))
        {
            std::cout << "depth file not found:" << depthFileName.string();
            break;
        }

        // read image
        cv::Mat img_depth = cv::imread(depthFileName.string(), cv::IMREAD_UNCHANGED);
        cv::Mat img_confidence = cv::imread(confidenceFileName.string(), cv::IMREAD_UNCHANGED);
        cv::Mat img_inner = cv::Mat::zeros(img_depth.size(), img_depth.type());

        // prepare trajectory image for display
        if (fileID == 0)
        {
            trajectory_img = cv::Mat::zeros(img_depth.size(), CV_8UC3);
        }

        // filter ROI
        cv::Mat mask;
        FindROIMask(img_depth, img_confidence, mask);
        FilterMeasurementArea(img_depth, img_inner, mask);

        // find finger tip
        cv::Point finger_tip_point;
        bool isValid = FindFingerTip(img_inner, finger_tip_point);
        if (isValid)
        {
            int depthVal = img_depth.at<ushort>(finger_tip_point);
            TrajectoryPoint trajPoint(finger_tip_point, fileID, depthVal);
            finger_tip_trajectory.push_back(trajPoint);
        }

        // prepare display
        cv::Mat cimg, img_c_u8, img_d_u8;
        // img_inner.convertTo(cimg, CV_8U, 255.0 / (setting.depthMax - setting.depthMin));
        img_confidence.convertTo(img_c_u8, CV_8U, 0.1);
        img_depth.convertTo(img_d_u8, CV_8U, 0.1);
        std::vector<cv::Mat> channels;
        channels.push_back(img_d_u8);
        channels.push_back(img_c_u8);
        channels.push_back(img_c_u8);
        cv::merge(channels, cimg);
        //cv::cvtColor(cimg, cimg, cv::COLOR_GRAY2BGR);
        cv::putText(cimg, baseDirName.string().substr(33, baseDirName.string().length()-33),cv::Point(10,10), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        if (isValid)
        {
            cv::drawMarker(cimg, finger_tip_point, color_map.at<cv::Vec3b>(fileID));
            if (finger_tip_trajectory.size() > 1)
            {
                cv::line(trajectory_img, finger_tip_point, finger_tip_trajectory[finger_tip_trajectory.size() - 2].finger_tip, color_map.at<cv::Vec3b>(fileID));
            }
            for (int i = 1; i < finger_tip_trajectory.size(); i++)
            {
                cv::line(cimg, finger_tip_trajectory[i].finger_tip, finger_tip_trajectory[i - 1].finger_tip, color_map.at<cv::Vec3b>(i));
            }
            outputVideo << cimg;
            cv::imwrite(resultFileName.string(), cimg);
        }


        // display
        cv::Mat cimg_depth;
        img_depth.convertTo(cimg_depth, CV_8U, -250.0 / setting.depthMax, 250.0);
        cv::imshow("confidence", cimg);
        cv::imshow("depth", cimg_depth);
        cv::imshow("trajectory", trajectory_img);
        int key = cv::waitKey(60);
        if (key == ' ')
        {
            key = cv::waitKey(0);
        }
        
    }


    // Save Results
    cv::imwrite(trajectory_img_file_name.string(), trajectory_img);
    outputVideo.release();
    TrajectoryPoint::DumpToCSV(csv_file_name, finger_tip_trajectory);
}
int main()
{
    test_stroke_recognizer();
    std::vector<std::string> datasets = {
        "D:\\git\\HandPointer\\dataset\\ds325\\fast_circles",
        "D:\\git\\HandPointer\\dataset\\ds325\\gestures_two_hands",
        "D:\\git\\HandPointer\\dataset\\ds325\\gestures_two_hands_swap",
        "D:\\git\\HandPointer\\dataset\\ds325\\sequence_closed_hand",
        "D:\\git\\HandPointer\\dataset\\ds325\\sequence_open_hand",
        "D:\\git\\HandPointer\\dataset\\ds325\\sequence_small_shapes",
        "D:\\git\\HandPointer\\dataset\\ds536\\circle_ccw",
        "D:\\git\\HandPointer\\dataset\\ds536\\circle_ccw_far",
        "D:\\git\\HandPointer\\dataset\\ds536\\circle_ccw_hand",
        "D:\\git\\HandPointer\\dataset\\ds536\\circle_sequence",
        "D:\\git\\HandPointer\\dataset\\ds536\\multiple_shapes_1",
        "D:\\git\\HandPointer\\dataset\\ds536\\rectangle_ccw",
        "D:\\git\\HandPointer\\dataset\\ds536\\rectangle_cw",
        "D:\\git\\HandPointer\\dataset\\ds536\\star",
        "D:\\git\\HandPointer\\dataset\\ds536\\zigzag"
    };
    for (std::string dataset : datasets)
    {
        // iterate over files
        ProcessSequence(dataset);
    }
    return 0;
}
