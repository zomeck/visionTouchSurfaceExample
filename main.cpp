#include <iostream>
#include "kinecthandler.h"
#include "leaphandler.h"

#include "gesturemanager.h"
#include "doubleswipemanager.h"
#include "swipemanager.h"
#include "tappingmanager.h"
#include "monitorhandsmanager.h"

#include "ioeventmanager.h"
#include "scrollupdownmanager.h"
#include "singleclickholdeventmanager.h"
#include "storehandmask.h"
#include "alttabeventmanager.h"

#include "visionprocessor.h"
//#include "hand.h"
using namespace std;

//old version
/*
int segmentScreen(const Mat & grayFrame, int Ith,float epsilon, vector<Point2i> & screenContours, Mat & maskShow ){


    //segment screen
    Mat IThres;
    vector<vector<Point2i> > contours;
    Canny( grayFrame, IThres, Ith, Ith, 3 );

    findContours( IThres, contours, CV_RETR_EXTERNAL,CV_CHAIN_APPROX_SIMPLE );

    float areaMax=0,area;
    int idContourMax=-1;
    for( int i=0; i<contours.size(); i++ ){
        area=contourArea( contours[i] );
        if( area>areaMax ){
            areaMax=area;
            idContourMax=i;
        }
    }
    //cout<<contours.size()<<" "<<idContourMax<<endl;

    //Mat drawing = Mat::zeros( IThres.size(), CV_8UC3 );
    Mat mask = Mat::zeros(IThres.size(), CV_8UC1);
    //Mat maskShow;//= Mat::zeros(IThres.size(), CV_8UC3);
    if( idContourMax>=0 ){
        //cout<<contours[idContourMax].size()<<endl;
        drawContours(mask, contours, idContourMax, Scalar(255), CV_FILLED);
        //drawContours(drawing,contours,idContourMax,Scalar(0,255,0));

        approxPolyDP(contours[idContourMax],screenContours,epsilon,true);

        //cout<<contoursFiltered.size()<<endl;
        //mask.copyTo(maskShow);

        //to show aproximation diferences
        vector<Mat> maskvect;
        maskvect.push_back(mask);
        maskvect.push_back(mask);
        maskvect.push_back(mask);
        merge(maskvect,maskShow);

        vector<vector<Point2i> > tmpVect;
        tmpVect.push_back(screenContours);

        if( screenContours.size()==4 ){
            drawContours(maskShow, tmpVect, 0, Scalar(0,255,0),3 );
        }else{
            drawContours(maskShow, tmpVect, 0, Scalar(0,0,255) );
        }



    }

    return idContourMax>=0;
}

void getScreenTransformation( KinectHandler & kinect, ProjectiveMapping & kinect2screen,float errorTolerance , Mat & mask){


    //segmenting screen
    vector<Point2i> screenContours ;
    Mat maskShow, colorFrame,grayFrame;

    namedWindow( "ScreenSegmentation", CV_WINDOW_FREERATIO );


    int Ith=140;//70 with les light
    cvCreateTrackbar("IThreshold","ScreenSegmentation",&Ith,255,0);

    //cvCreateTrackbar("Tolerance","ScreenSegmentation",&errorTolerance,10,0);
    vector<Mat> bgrChannels;
    vector< vector<Point2i> > contoursVect;
    while(true){
        kinect.waitFrame();
        kinect.getImage( colorFrame );

        //cvtColor( colorFrame, hsvFrame, CV_BGR2HSV_FULL );
        //split(hsvFrame ,hsvChannels );
        cvtColor(colorFrame,grayFrame,CV_BGR2GRAY);

        split( colorFrame, bgrChannels );
        segmentScreen( grayFrame, Ith, errorTolerance, screenContours,maskShow );

        mask = Mat::zeros( colorFrame.size(), CV_8UC1);

        contoursVect.clear();
        contoursVect.push_back(screenContours);

        drawContours( mask, contoursVect, 0, Scalar(255), CV_FILLED );
        Mat dilateK=getStructuringElement( MORPH_DILATE, Point(100,100) );
        dilate( mask, mask, dilateK );
        //cout<<screenContours.size()<<endl;


        imshow("ScreenSegmentation", maskShow );

        usleep(100*1000);

        char key =cvWaitKey(15);
        if (key==13) {
            break;
        }
    }
    cout<<"size found "<<screenContours.size() <<endl;

    vector<Point2f> screenContoursF;

    for( int i=0; i < screenContours.size(); i++ ){

        Point2f tmp;
        tmp.x=screenContours.at(i).x;
        tmp.y=screenContours.at(i).y;
        screenContoursF.push_back( tmp );

    }

    cout<< screenContoursF<<endl;
    char key =cvWaitKey(0);


    kinect2screen.calculateMapping( screenContoursF );
}

void createMaxDepthWithHistogram( KinectHandler & kinect, Mat & maxDepth, int numFrames = 200 ){

    //int numFrames=200;
    int minBinBound=-255;
    int maxBinBound=-minBinBound;
    int numBins=abs(minBinBound)+abs(maxBinBound)+1;

    int threshold=0.01*numFrames; //min depth that gives at least this value in hist is d_max

    //Mat kinectImg(5, 5, CV_32S);
    //randu(kinectImg, Scalar::all(100), Scalar::all(255));
    Mat surfaceDepth;
    kinect.waitFrame();
    kinect.getDepthMap16( surfaceDepth );
    KinectHandler::depthMapCorrection( surfaceDepth, surfaceDepth );

    Mat kinectImg=surfaceDepth.clone();
    //cout<<kinectImg<<endl;


    int size_histogram[]={kinectImg.rows, kinectImg.cols, numBins};

    Mat temporalHistogram(3, size_histogram,CV_32S, Scalar::all(0));

    for(int t=0; t<numFrames; t++) //at each time step
    {
        //read frame
        //Mat frame = Mat(kinectImg.rows, kinectImg.cols, CV_32S);
        //randu(frame, Scalar::all(0), Scalar::all(255));

        Mat frame;
        kinect.waitFrame();
        kinect.getDepthMap16( frame );
        KinectHandler::depthMapCorrection( frame, frame );

        Mat frameS, kinectImgS;
        frame.convertTo( frameS, CV_32S );
        kinectImg.convertTo( kinectImgS, CV_32S );
        Mat diff=frameS-kinectImgS;
        //cout<<diff<<endl;
        //for each pixel
        for(int i=0; i<kinectImg.rows; i++)
        {
            for(int j=0; j<kinectImg.cols; j++)
            {
                int temp=diff.at<int>(i,j)-minBinBound;
                //cout<<diff.at<int>(i,j)<<"\t"<<temp<<endl;
                temporalHistogram.at<int>(i,j,temp)+=1;

            }
        }
    }

    //cout << temporalHistogram.at<int>(0,0,255)<< endl;
    Mat d_max=kinectImg.clone();

    //for each pixel
    for(int i=0; i<kinectImg.rows; i++)
    {
        for(int j=0; j<kinectImg.cols; j++)
        {
            //find d_max (starting search at closest pixels)
            int idx=minBinBound;
            int cumulativeHist=0;
            bool d_maxFound=false;
            while(idx<maxBinBound && !d_maxFound)
            {

                int diffVal=temporalHistogram.at<int>(i,j,idx-minBinBound);

                //cout<<diffVal<<" ";

                cumulativeHist+=diffVal;
                d_maxFound=(cumulativeHist>threshold);
                if( d_maxFound){
                    d_max.at<u_int8_t>(i,j)=kinectImg.at<u_int8_t>(i,j)+idx;
                    //cout<<kinectImg.at<int>(i,j);
                    //cout<<endl<<idx<<endl;

                }
                idx++;
                //cout<<idx<<endl;
            }
        }
    }

    maxDepth= d_max.clone()-1;
}


void testDoubleSwipeManager(){

    //GestureManager *dSwiper=new DoubleSwipeManager;

    int NManagers=3;
    GestureManager ** managers=new GestureManager*[NManagers];

    managers[0]=new DoubleSwipeManager;
    managers[1]=new SwipeManager;
    managers[2]=new TappingManager;

    ScrollUpDownManager *scroll=new ScrollUpDownManager;
    SingleClickHoldEventManager *drag=new SingleClickHoldEventManager;
    SingleClickHoldEventManager *drag2=new SingleClickHoldEventManager;



    managers[0]->setIOManager( scroll );
    managers[1]->setIOManager( drag );
    managers[2]->setIOManager( drag2 );


    drag->setButton( 1 );//setting Left button
    scroll->initData();
    drag->initData();




    //creating mask params
    float thicknessHand = 12;//2.5
    float thicknesFinger = 2.5;

    //hand area size boundaries
    int *sizes=new int[2];
    sizes[1]=640;
    sizes[0]=480;
    int minAreaContour=850;
    int maxAreaContour=(sizes[0]*sizes[1]/16);
    //hand aprox polyfit error allowed
    int tolerance=1;


    //screen corners params
    float epsilon=4;

    //histogram params
    int numFrames=200;
    bool useHist=true;

    //use only screen area
    bool useScreenSeg=true;

    //uncertanty of kinect
    float radiusUncertantyKinect=50;


    //initializing kinect and its mapping
    ProjectiveMapping kinect2screen( 1024, 768 );

    KinectHandler kinect;

    kinect.init();

    //initializing leap and its mapping
    bool useLeap=false;
    LeapHandler *leapHandler=NULL; //will initialise
    if( useLeap ){
        leapHandler=new LeapHandler;
    }
    LeapMapping leapMapping(1024, 768, 932.2, 520.7);
    leapMapping.calculateMapping(Point(512, -10));//leap centerposition

    //getting data ready for creating the background for the inRange
    cv::Mat surfaceDepth;
    kinect.waitFrame();
    kinect.getDepthMap16( surfaceDepth );

    KinectHandler::depthMapCorrection( surfaceDepth, surfaceDepth );

    cv::Mat maxDepth = Mat::zeros(surfaceDepth.rows,surfaceDepth.cols, CV_8UC1 )*1;//avoid segfault by *1


    if( useHist ){
        createMaxDepthWithHistogram( kinect, maxDepth, numFrames );
    }else{
        maxDepth = surfaceDepth*1;
        maxDepth-=1;
    }

    //creating minimum thres masks for segmenting hands and fingertips
    cv::Mat minDepthHand = maxDepth - thicknessHand;
    cv::Mat minDepthFinger = maxDepth - thicknesFinger;

    //creating screenmask
    Mat mask;
    if( useScreenSeg ){
        getScreenTransformation( kinect, kinect2screen, epsilon, mask );
    }

    //vars for storing loop information
    Mat depth16,depth8,colorFrame,grayFrame,touchHand,depth,touchFinger, touchFingerCopy, touchHandCopy;
    Mat erodeK=getStructuringElement(MORPH_ERODE,Size(3,3) );
    FrameData frameData;

    frameData.kinect2screen=&kinect2screen;

    while( true ) {
        // Wait for new data to be available
        kinect.waitFrame();

        if( useLeap ){
        leapHandler->waitFrame();
        }

        // Take current depth map with 16 bits int
        kinect.getDepthMap16( depth16 );
        // Take current depth map with 8 bits int
        //kinect.getDepthMap08( depth );
        // Take current color image
        kinect.getImage( colorFrame );

        depth16.convertTo(depth8, CV_8UC1, 255.0/2048);


        cvtColor(colorFrame,grayFrame,CV_BGR2GRAY);

        touchHand =cv::Mat::zeros( depth.size(), CV_8UC1 );


        depth8.copyTo(depth, mask );

        if( useScreenSeg ){
            depth.copyTo( touchHand,mask );
            depth.copyTo( touchFinger,mask );
        }else{
            depth.copyTo( touchHand );
            depth.copyTo( touchFinger );

        }



        inRange(touchHand,minDepthHand,maxDepth,touchHand);//SEGMENTING hand
        erode(touchHand,touchHand,erodeK);
        //erode(touchHand,touchHand,erodeK);

        inRange( touchFinger, minDepthFinger, maxDepth, touchFinger );//SEGMENTING finger
        //erode(touchFinger,touchFinger,erodeK);

        touchFinger.copyTo( touchFingerCopy );
        //bitwise_and( prevTouchFinger, touchFinger, touchFinger );//temporal filtering

        touchHand.copyTo(touchHandCopy);
        //bitwise_and(prevTouchHand,touchHand,touchHand);//temporal filtering

        vector< vector<Point> > contours;
        vector<Vec4i> hierarchy;

        Mat touchHandCopy2=touchHand.clone();

        findContours( touchHandCopy2, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

        /// Draw contours
        //Mat drawing = Mat::zeros( touchCopy.size(), CV_8UC3 );
        Mat drawingHand = colorFrame;

        vector< vector<Point> > contoursHandFiltered;
        vector<Hand> hands;
        contoursHandFiltered.clear();
        //vector< Mat > contourMaskVect;
        for( int i = 0; i< contours.size(); i++ )
        {
            float area=contourArea( contours.at(i) );
            if( area>minAreaContour && area < maxAreaContour ){
                Scalar color = Scalar( 100, 0, 0 );
                vector<Point> tmpHand;
                approxPolyDP( contours.at(i),tmpHand , tolerance, true );
                contoursHandFiltered.push_back( tmpHand );

                Hand hand;
                hand.shapeType=-1;
                hand.contour=tmpHand;
                hands.push_back( hand );

                drawContours( drawingHand, contoursHandFiltered, contoursHandFiltered.size()-1, color, CV_FILLED);//, hierarchy);//, 0, Point() );


            }
        }

        //Mat handMask=depth.clone();
        //frameData.contourHandsFiltered=&contoursHandFiltered;

        //now fingertips of hands

        // vector< handFingertipsContour >, handFingertipsContour=vector<vector<point>>
        vector< vector< vector< Point > > > contoursFingersHands;
        vector< bool >foundPalmHands;

        for( int i=0; i< contoursHandFiltered.size(); i++ ){

            Mat handMask=Mat::zeros(depth8.rows,depth8.cols, cv::THRESH_BINARY );
            drawContours(handMask,contoursHandFiltered,i,Scalar(255,255,255),CV_FILLED);

            Mat touchFingerHand_i=touchFinger.clone();
            if( useScreenSeg ){
                bitwise_and(handMask,touchFingerHand_i,touchFingerHand_i,mask);
            }else{
                bitwise_and(handMask,touchFingerHand_i,touchFingerHand_i );
            }

            vector< vector<Point> > contoursF,contoursFFiltered;
            vector<Vec4i> hierarchyF;

            Mat touchFingerC= touchFingerHand_i.clone();
            findContours( touchFingerC, contoursF, hierarchyF, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, Point(0, 0) );

            int minExc=1,exc;
            int minIdx=-1;

            bool foundPalm=false;
            for( int j=0; j<contoursF.size(); j++ ){
                RotatedRect bBoxf=minAreaRect( contoursF.at(j) );
                exc=std::min(bBoxf.boundingRect().width,bBoxf.boundingRect().height)/std::max(bBoxf.boundingRect().width,bBoxf.boundingRect().height);
                float areaF= contourArea( contoursF.at(j) );


                foundPalm=foundPalm || areaF>200;

                if( exc<0.6 && areaF>=60 ){
                    //minExc=exc;
                    //minIdx=i;
                    contoursFFiltered.push_back( contoursF.at(j) );
                    Finger finger;
                    finger.contour=contoursF.at(j) ;
                    hands.at(i).fingers.push_back( finger );
                }

            }

            contoursFingersHands.push_back( contoursFFiltered );
            foundPalmHands.push_back( foundPalm );
            //hands.at(i).fingers=contoursFFiltered;
            //hands.at(i).fingers.insert()
            //copy( contoursFFiltered.begin(), contoursFFiltered.end(), hands.at(i).fingers.be)
            hands.at(i).foundPalm=foundPalm;

        }

        //chance to get a pointable object
        bool foundMatch=false;
        Point3i matchPointable;
        matchPointable.x=-1;
        matchPointable.y=-1;
        matchPointable.z=-1;
        if(contoursHandFiltered.size()==1 && contoursFingersHands.at(0).size() == 1 && useLeap ){

            Rect bboxF=boundingRect(contoursFingersHands.at(0).at(0) );

            radiusUncertantyKinect=bboxF.width+400;

            Point2i pointableCandidate;

            pointableCandidate.x=bboxF.tl().x+bboxF.width/2;
            pointableCandidate.y=bboxF.br().y;

            kinect2screen.mapPoints( pointableCandidate, pointableCandidate);

            foundMatch=leapHandler->findBestMatchOnCurrentFrame( pointableCandidate, radiusUncertantyKinect, leapMapping, matchPointable );

        }

        frameData.matchPointable=matchPointable;
        //frameData.contoursFingersHands=&contoursFingersHands;
        //frameData.foundPalmHands=&foundPalmHands;
        frameData.hands=&hands;

        for( int i=0; i<NManagers; i++ ){
            managers[i]->updateStatus( frameData );
        }

        contoursFingersHands.clear();

        hands.clear();

        imshow( "contours",drawingHand );

        char key =cvWaitKey(15);
        if (key==27) {
            break;
        }

    }

}


*/

Point2f sampleMouseGetPos(){

    Display *dpy = NULL;
    XEvent event;
    dpy = XOpenDisplay (NULL);
    /* Get the current pointer position */


    XQueryPointer (dpy, RootWindow (dpy, 0), &event.xbutton.root,
                   &event.xbutton.window, &event.xbutton.x_root,
                   &event.xbutton.y_root, &event.xbutton.x, &event.xbutton.y,
                   &event.xbutton.state);

    //cout<< event.xbutton.x_root<<", "<<event.xbutton.y_root<<endl;


    XCloseDisplay (dpy);

    return Point2f(event.xbutton.x_root,event.xbutton.y_root );

}

void testVisionProcessor(){


    //initializing leap and its mapping
/*
    bool useLeap=false;
    LeapHandler *leapHandler=NULL; //will initialise
    if( useLeap ){
        leapHandler=new LeapHandler;
    }
    */
    //LeapMapping leapMapping(1024, 768, 932.2, 520.7);
    //leapMapping.calculateMapping(Point(512, -10));//leap centerposition


    int NManagers=3;
    GestureManager ** managers=new GestureManager*[NManagers];

    managers[0]=new DoubleSwipeManager;
    managers[1]=new SwipeManager;
    managers[2]=new TappingManager;
    //managers[3]=new MonitorHandsManager;

    ScrollUpDownManager *scroll=new ScrollUpDownManager;
    SingleClickHoldEventManager *drag=new SingleClickHoldEventManager;
    SingleClickHoldEventManager *drag2=new SingleClickHoldEventManager;
    StoreHandMask * store=new StoreHandMask;
    AltTabEventManager *altTab=new AltTabEventManager;


    string folderHand( "defaultOutput" );
    store->setFolder(folderHand);


    managers[0]->setIOManager( scroll );
    managers[1]->setIOManager( drag );
    //managers[1]->setIOManager( altTab );
    managers[2]->setIOManager( drag2 );
    //managers[3]->setIOManager( store );


    drag->setButton( 1 );//setting Left button
    scroll->initData();
    drag->initData();
    drag2->initData();
    altTab->initData();


    VisionProcessor visionProcessor;
    //visionProcessor.setVerboseVideo(true);

    //visionProcessor.init(1280,1024);
    visionProcessor.init();

    //getting data ready for creating the background for the inRange
    visionProcessor.createMaxDepthWithHistogram();

    visionProcessor.getScreenTransformation();

    FrameData frameData;

    //ProjectiveMapping kinect2creen=visionProcessor.getKinect2screen();

    //frameData.kinect2screen=&kinect2creen;

/*
    Point3i matchPointable;
    matchPointable.x=-1;
    matchPointable.y=-1;
    matchPointable.z=-1;
    frameData.matchPointable=matchPointable;
*/
    //Mat drawingHand;


    vector<Hand> hands;
    frameData.hands=&hands;


    VideoWriter outputVideo;
    bool verboseVideo;

    verboseVideo=true;

    if( verboseVideo ){
        time_t rawtime;
        struct tm * timeinfo;
        char buffer [100];

        time (&rawtime);
        timeinfo = localtime (&rawtime);

        strftime (buffer,80,"video_%Y%B%d_%Hh_%Mm_%Ss.avi",timeinfo);

        string debugVideoName(buffer);
        Size2i kinectRes=visionProcessor.getKinectResolution();
        kinectRes.width*=2;
        kinectRes.height*=2;
        outputVideo.open( debugVideoName,CV_FOURCC('M', 'P', '4', '2')  ,30,kinectRes ,true );
    }


    double acumSec=0;
    long nframes=0;
    int MaxFrames=1500;
    while( true ) {


        double duration=0;

        hands.clear();
        Mat depth;
        visionProcessor.processFrame( frameData, duration );

        //frameData.hands=&hands;
        //frameData.depthMap=depth;

        for( int i=0; i<NManagers; i++ ){
            managers[i]->updateStatus( frameData );
        }

        //hands.clear();



        acumSec+=duration;
        nframes++;

        if( nframes == MaxFrames ){

            cout<<"Avg processing frame rate: "<< ((float)MaxFrames)/acumSec<<endl;
            nframes=0;
            acumSec=0;
        }





        if( verboseVideo ){

            Mat tmp;
            Mat depthTmp;
            Mat handTmp;
            Mat fingersTmp;
            Mat tmp2,tmpF;

            //frameData.depthMap.convertTo( depthTmp, CV_8UC3);
            cvtColor( frameData.depthMap, depthTmp, CV_GRAY2RGB );
            cvtColor( frameData.handsMaskRaw, handTmp, CV_GRAY2RGB );
            cvtColor( frameData.fingersMaskRaw, fingersTmp, CV_GRAY2RGB );

            hconcat( frameData.rgbDebug, depthTmp, tmp );
            hconcat( handTmp, fingersTmp, tmp2 );
            vconcat( tmp,tmp2,tmpF );

            outputVideo << tmpF;
            imshow( "contours",tmpF );
        }else{
            imshow( "contours",frameData.rgbDebug );
        }

        char key =cvWaitKey(1);
        if (key==27/*ESC*/) {
            break;
        }

    }

}

void testTappingAccuracy(){

    int screenResolutionW=1024;
    int screenResolutionH=768;

    int NManagers=1;
    GestureManager ** managers=new GestureManager*[NManagers];

    managers[0]=new TappingManager;
    //managers[1]=new MonitorHandsManager;

    //ScrollUpDownManager *scroll=new ScrollUpDownManager;
    SingleClickHoldEventManager *drag=new SingleClickHoldEventManager;
    //SingleClickHoldEventManager *drag2=new SingleClickHoldEventManager;
    StoreHandMask * store=new StoreHandMask;

    string folderHand( "output" );
    store->setFolder(folderHand);


    // managers[0]->setIOManager( scroll );
    managers[0]->setIOManager( drag );
    //managers[2]->setIOManager( drag2 );
    //managers[3]->setIOManager( store );


    drag->setButton( 1 );//setting Left button
    drag->initData();

    VisionProcessor visionProcessor;


    visionProcessor.init();


    //getting data ready for creating the background for the inRange
    visionProcessor.createMaxDepthWithHistogram();

    visionProcessor.getScreenTransformation();

    FrameData frameData;

    ProjectiveMapping kinect2creen=visionProcessor.getKinect2screen();
/*
    frameData.kinect2screen=&kinect2creen;


    Point3i matchPointable;
    matchPointable.x=-1;
    matchPointable.y=-1;
    matchPointable.z=-1;
    frameData.matchPointable=matchPointable;
*/
    Mat drawingHand;
    /*
    while( true ) {

        vector<Hand> hands;
        Mat depth;
        visionProcessor.processFrame( hands, drawingHand,depth );

        frameData.hands=&hands;
        frameData.depthMap=depth;

        for( int i=0; i<NManagers; i++ ){
            managers[i]->updateStatus( frameData );
        }

        hands.clear();

        //imshow( "contours",drawingHand );

        char key =cvWaitKey(1);
        if (key==27) {
            break;
        }

    }
    */

    vector< Point2f > screenPoints;

    int stepW=floor( screenResolutionW/8.0 );
    int stepH=floor( (screenResolutionH-50)/6.0 );//substract window bars height

    for( int i=stepW; i<(screenResolutionW-stepW); i+=stepW ){

        for( int j=stepH; j<(screenResolutionH-50-stepH); j+=stepH ){
            screenPoints.push_back( Point2f( i, j ) );
        }
    }


    int attempsPerPoint=150;
    vector<Point2f> capturedPoints;
    int minDist=50;
    namedWindow( "WindowName", CV_WINDOW_AUTOSIZE );
    cv::moveWindow("WindowName", 0, 0);
    vector<Hand> hands;
    frameData.hands=&hands;

    for( int i=0; i< screenPoints.size(); i++ ){

        Mat testMat=Mat::zeros(screenResolutionH,screenResolutionW-50, CV_8UC3 )*1;

        imshow( "WindowName",testMat );
        char key =cvWaitKey(250);
        //points shifted the window header width and height
        circle( testMat, Point(screenPoints.at(i).x+1, screenPoints.at(i).y+50 ), 5, Scalar(0,255,0),2 );
        circle( testMat, Point(screenPoints.at(i).x+1, screenPoints.at(i).y+50 ), 10, Scalar(0,255,0),2 );
        imshow( "WindowName",testMat );

        cout<<"Showing new point"<<endl;

        key =cvWaitKey(250);
        if (key==27) {
            break;
        }
        Point2f acumMeasure;
        acumMeasure.x=0;
        acumMeasure.y=0;
        int counterMeasured=0;
        double duration;
        for( int j=0;j<attempsPerPoint;j++ ){


            hands.clear();
            Mat depth;
            //visionProcessor.processFrame( hands, drawingHand,depth );
            visionProcessor.processFrame( frameData,duration );
            //frameData.hands=&hands;
            //frameData.depthMap=depth;

            for( int id=0; id<NManagers; id++ ){
                managers[id]->updateStatus( frameData );
            }



            //get mouse position and assign it to acumMeasure
            Point2f mousePosition=sampleMouseGetPos();

            float dist=norm( mousePosition - screenPoints.at(i) );
            if( dist<minDist ){
                //cout<<"Match found"<<endl;
                acumMeasure.x +=mousePosition.x;
                acumMeasure.y +=mousePosition.y;
                counterMeasured++;
            }



            usleep( 10*1000 );
            //sleep( 1 );

        }

        acumMeasure.x/=counterMeasured;
        acumMeasure.y/=counterMeasured;

        capturedPoints.push_back( acumMeasure );


        key =cvWaitKey(15);
        if (key==27) {
            break;
        }


    }

    cout<<"Screen points"<<endl;
    cout<<screenPoints<<endl;
    cout<<"Observed points"<<endl;
    cout<<capturedPoints<<endl;

}

int main()
{
    //cout << "Hello World!" << endl;


    testVisionProcessor();
    //testTappingAccuracy();
    //testDoubleSwipeManager();



    return 0;
}

