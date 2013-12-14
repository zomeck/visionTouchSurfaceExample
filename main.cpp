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


        for( int i=0; i<NManagers; i++ ){
            managers[i]->updateStatus( frameData );
        }

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

    Mat drawingHand;


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
    testVisionProcessor();
    //testTappingAccuracy();


    return 0;
}

