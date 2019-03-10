
#include <stdio.h>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/shape_utils.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/videoio.hpp>
#include <fstream>
#include <iostream>
#include <algorithm>
#include <cstdlib>
#include <iomanip>    

using namespace std;
using namespace cv;
using namespace cv::dnn;

int main(int argc, char *argv[])
{    

    Mat image;
    image = imread(argv[1], IMREAD_COLOR);   // Read the file

    cout<< "image size"<<image.cols<<"x"<<image.rows<<endl;

    if(! image.data )
    {
        cout <<  "Could not open or find the image" << std::endl ;
        return -1;
    }

    Mat inputBlob = blobFromImage(image, 1 , Size(image.cols, image.rows), Scalar(103.939, 116.779, 123.680), false, false); //Convert Mat to batch of images

    cout<<"loading net"<<endl;

    dnn::Net net;
    net= readNetFromTorch(argv[2]);
    vector< String > layers;

    layers=net.getLayerNames ();
    for (std::vector<String>::iterator it = layers.begin() ; it != layers.end(); ++it)
    {
        std::cout << "-> " << *it<<endl;
    }

    if (net.empty())
    {
        cout<<"Failed to load network"<<endl;
        return -1;
    }

    net.setInput(inputBlob);
    Mat styledMat = net.forward();
    // convert tensor blob to  mat images
    std::vector<cv::Mat> outImg;
    imagesFromBlob(styledMat,outImg);
    
    Mat styledImgOut;

    outImg[0].convertTo(styledImgOut,0);

    cout<<"Channels :"<<outImg[0].channels()<<endl;
    cout<<"Depth :"<<styledImgOut.depth()<<endl;
    cout<<"Res :"<<outImg[0].cols<<"x"<<outImg[0].rows<<endl;

    styledImgOut =styledImgOut+ Scalar(103.939, 116.779, 123.680);

    namedWindow( "Style transfer", -1 );
    imshow( "Style transfer", styledImgOut );
    waitKey(0);


    imwrite( "style_out.jpg", styledImgOut );

}
