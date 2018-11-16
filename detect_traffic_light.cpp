#include <opencv2/opencv.hpp>
#include <iostream>

#define min(d, s) ((d < s) ? (d) : (s))
#define max(d, s) ((d > s) ? (d) : (s))

using namespace cv;
using namespace std;

int low_threshold = 70;
int local_minima_threshold = 50;
int test_thresh = 5;

RNG rng(12345);

void theFunction(int, void *);

//TODO merge multiple processing of hsv of source

char *file_number;

void erosion(Mat &src, Mat &dst, int erosion_elem, int erosion_size) {

  int erosion_type;
  if (erosion_elem == 0) erosion_type = MORPH_RECT;
  else if (erosion_elem == 1) erosion_type = MORPH_CROSS;
  else if (erosion_elem == 2) erosion_type = MORPH_ELLIPSE;

  Mat element = getStructuringElement(erosion_type, Size(2 * erosion_size + 1, 2 * erosion_size + 1), Point(erosion_size, erosion_size));

  erode(src, dst, element);
}

void dilation(Mat &src, Mat &dst, int dilation_elem, int dilation_size) {
  
    int dilation_type;
    if (dilation_elem == 0) dilation_type = MORPH_RECT;
    else if (dilation_elem == 1) dilation_type = MORPH_CROSS;
    else if (dilation_elem == 2) dilation_type = MORPH_ELLIPSE;

    Mat element = getStructuringElement(dilation_type, Size(2 * dilation_size + 1, 2 * dilation_size + 1), Point(dilation_size, dilation_size));

    dilate(src, dst, element);
}

void findLocalMinima(Mat& input_image, vector<Point> &local_minimas, double threshold_value) {
    Mat local_minima;
    Mat eroded_input_image,thresholded_input_image,thresholded_input_8bit;
    erode(input_image, eroded_input_image, Mat());
    compare(input_image, eroded_input_image, local_minima, CMP_EQ);
    threshold(input_image, thresholded_input_image, threshold_value, 1, THRESH_BINARY_INV);
    thresholded_input_image.convertTo(thresholded_input_8bit, CV_8U);
    bitwise_and(local_minima, thresholded_input_8bit, local_minima);

    for (int i = 0; i < local_minima.rows; i++)
        for (int j = 0; j < local_minima.cols; j++)
            if (local_minima.at<uchar>(i, j) > 0)
                local_minimas.push_back(Point(j, i));
}


void getHist(Mat src, Mat &hist) {

    //Mat hsv; cvtColor(src, hsv, CV_BGR2HSV);
    int histSize = 2; //no quantization


/*    // Quantize the hue to 30 levels
    // and the saturation to 32 levels
    int hbins = 30, sbins = 32;
    int histSize[] = {hbins, sbins};
    // hue varies from 0 to 179, see cvtColor
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };*/
    
     /// Set the ranges ( for B,G,R) )
    float range[] = { 0, histSize } ;
    const float* histRange = { range };
    
    // we compute the histogram from the 0-th and 1-st channels
    int channels[] = {0};

    //vector<Mat> bgr_planes;
    //split( src, bgr_planes );
    calcHist(&src, 1, channels, Mat(), hist, 1, &histSize, &histRange, true, false);

/*
    int hist_w = 512; int hist_h = 400;
    int bin_w = cvRound( (double) hist_w/histSize );


    Mat histImage( hist_h, hist_w, CV_8UC3, Scalar( 0,0,0) );*/

     //normalize(hist, hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  //normalize(g_hist, g_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );
  //normalize(r_hist, r_hist, 0, histImage.rows, NORM_MINMAX, -1, Mat() );

 // Draw the histograms for B, G and R
   /* for (int i = 1; i < histSize; i++) {
      line( histImage, Point( bin_w*(i-1), hist_h - cvRound(hist.at<float>(i-1)) ),
                       Point( bin_w*(i), hist_h - cvRound(hist.at<float>(i)) ),
                       Scalar( 255, 0, 0), 2, 8, 0  );
    //namedWindow(, CV_WINDOW_AUTOSIZE );
  //waitKey(0);
  }*/

//    imshow("calcHist Demo: " + to_string(loop), histImage );
  //  imshow("calcHist Source: " + to_string(loop), src );
   /* for (int i = 0; i < histSize; i++) {
        cout << "Loop: " << loop << endl;
        cout << hist.at<float>(i) << endl;
    }
    imshow( "H-S Histogram", histImage );

    waitKey(0);*/


    /*calcHist( &hsv, 1, channels, Mat(), // do not use mask
             hist, 1, histSize, ranges,
             true, // the histogram is uniform
             false );*/
    //double maxVal = 0;
    //minMaxLoc(hist, 0, &maxVal, 0, 0);

/*    int scale = 10;
    Mat histImg = Mat::zeros(sbins * scale, hbins * 10, CV_8UC3);

    for( int h = 0; h < hbins; h++)
        for( int s = 0; s < sbins; s++) {
            float binVal = hist.at<float>(h, s);
            int intensity = cvRound(binVal*255/maxVal);
            rectangle( histImg, Point(h*scale, s*scale),
                        Point( (h+1)*scale - 1, (s+1)*scale - 1),
                        Scalar::all(intensity),
                        CV_FILLED );
        }

    imshow( "H-S Histogram", histImg );*/
}

void backProjection(string filename, Mat &target, Mat &thresh) {

    Mat roi = imread(filename); //region of interest
    Mat roi_hsv, target_hsv;
    cvtColor(roi, roi_hsv, COLOR_BGR2HSV, 0);
    //hsv = cv2.cvtColor(roi,cv2.COLOR_BGR2HSV)
    //target = cv2.imread('CS4053/CamVidLights/CamVidLights05.png')
//    hsvt = cv2.cvtColor(target,cv2.COLOR_BGR2HSV)
    cvtColor(target, target_hsv, COLOR_BGR2HSV, 0);
    // calculating object histogram
//    C++: void calcHist(const Mat* images, int nimages, const int* channels, InputArray mask, OutputArray hist, int dims, const int* histSize, const float** ranges, bool uniform=true, bool accumulate=false )

    MatND roi_hist;
    int channels[] = {0, 1};
    int hbins = 180, sbins = 256;
    int histSize[] = {hbins, sbins};
    float hranges[] = { 0, 180 };
    // saturation varies from 0 (black-gray-white) to
    // 255 (pure spectrum color)
    float sranges[] = { 0, 256 };
    const float* ranges[] = { hranges, sranges };
    calcHist(&roi_hsv, 1, channels, Mat(), roi_hist, 2, histSize, ranges, true, false);
    //roihist = cv2.calcHist([hsv],[0, 1], None, [180, 256], [0, 180, 0, 256])
    // normalize histogram and apply backprojection
    normalize(roi_hist, roi_hist, 0, 255, NORM_MINMAX);
    //cv2.normalize(roihist,roihist,0,255,cv2.NORM_MINMAX)
    //Mat dst;
    calcBackProject(&target_hsv, 1, channels, roi_hist, thresh, ranges, 1, true);
    //C++: void calcBackProject(const Mat* images, int nimages, const int* channels, InputArray hist, OutputArray backProject, const float** ranges, double scale=1, bool uniform=true )
    //dst = cv2.calcBackProject([hsvt],[0,1],roihist,[0,180,0,256],1)
    // Now convolute with circular disc

    //dilation(thresh, thresh, 2, 1); erosion(thresh, thresh, 2, 1);
    Mat disc = getStructuringElement(MORPH_ELLIPSE, Size(5, 5), Point(-1,-1)); //disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    filter2D(thresh, thresh, -1, disc, Point(-1, -1), 0, BORDER_DEFAULT);
    
    //cv2.filter2D(dst,-1,disc,dst) //Python: cv2.filter2D(src, ddepth, kernel[, dst[, anchor[, delta[, borderType]]]]) → dst¶
    //C++: void filter2D(InputArray src, OutputArray dst, int depth, InputArray kernel, Point anchor=Point(-1,-1), double delta=0, int borderType=BORDER_DEFAULT );

    double ret = threshold(thresh, thresh, 100, 255, THRESH_BINARY);

    //# threshold and binary AND
    //ret,thresh = cv2.threshold(dst,50,255,0)
    //thresh = cv2.merge((thresh,thresh,thresh))
    //merge();
    //C++: void bitwise_and(InputArray src1, InputArray src2, OutputArray dst, InputArray mask=noArray())

    // Mat res;
    // bitwise_and(thresh, target, res);
    //res = cv2.bitwise_and(target,thresh)
    //res = np.vstack((target,thresh,res))
    //cv2.imwrite('res.jpg',res)

    namedWindow("Back Project: " + filename, CV_WINDOW_AUTOSIZE);
    int a;
    createTrackbar("Does Nothing Yet", "Back Project: " + filename, &a, 1, theFunction);
    imshow("Back Project: " + filename, thresh);
}

void chamferMatching(Mat &chamfer_image, Mat &model, Mat &matching_image) {
    // Extract the model points (as they are sparse).
    vector<Point> model_points;
    int image_channels = model.channels();
    for (int model_row = 0; (model_row < model.rows); model_row++) {
        uchar *curr_point = model.ptr<uchar>(model_row);
        for (int model_column = 0; model_column < model.cols; model_column++) {
            if (*curr_point > 0) {
                Point new_point = Point(model_column, model_row);
                model_points.push_back(new_point);
                /*cout << "pushed a new point: ";
                cout << new_point;
                cout << endl;*/
            }
            curr_point += image_channels;
        }
    }
    int num_model_points = model_points.size();
    image_channels = chamfer_image.channels();
    // Try the model in every possible position
    matching_image = Mat(chamfer_image.rows - model.rows + 1, chamfer_image.cols - model.cols + 1, CV_32FC1);
    for (int search_row = 0; search_row <= chamfer_image.rows - model.rows; search_row++) {
        float *output_point = matching_image.ptr<float>(search_row);
        for (int search_column = 0; search_column <= chamfer_image.cols - model.cols; search_column++) {
            float matching_score = 0.0;
            for (int point_count = 0; point_count < num_model_points; point_count++) 
                matching_score += (float) *(chamfer_image.ptr<float>(model_points[point_count].y + search_row) + search_column + model_points[point_count].x * image_channels);
            *output_point = matching_score;
            output_point++;
        }
    }
}

void getBlackVsWhitePixels(Mat &bin_image, int &white, int &black) {
    white = black = 0;
    for (int i = 0; i < bin_image.rows; i++) {
        for (int j = 0; j < bin_image.cols; j++) {
            int val = (int) (bin_image.at<uchar>(i, j));
            if (val == 255) {
                white++;
            } else
                black++;
        }
    }
}

//find ellipses around bitmap
void findLights(Mat &source, Mat &edge_image, vector<Rect> &lights_r, vector<Rect> &lights_a, vector<Rect> &lights_g, Mat src_gray) {

    vector< vector<Point> > circle_contours;
    Mat circle = imread("CS4053/circle.png", CV_LOAD_IMAGE_COLOR);
    cvtColor(circle, circle, CV_BGR2GRAY);
    Canny(circle, circle, 50, 100, 3);
    vector<Vec4i> circle_hierarchy;

    findContours(circle, circle_contours, circle_hierarchy, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0));
    if (circle_contours.size() < 1)
        cout << "error circle in png not found" << endl;

    vector<Point> circle_contour = circle_contours[0];

    Mat b_proj_lights_map_red, b_proj_lights_map_amber, b_proj_lights_map_green;
    backProjection("CS4053/red-all-2.png", source, b_proj_lights_map_red);
    backProjection("CS4053/amber-all-1.png", source, b_proj_lights_map_amber);
    backProjection("CS4053/green-all-6.png", source, b_proj_lights_map_green);
    Mat drawing = src_gray; //Mat::zeros(edge_image.size(), CV_8UC3);

    // Mat canny_output;
    vector< vector<Point> > contours;
    vector<Vec4i> hierarchy;

    //findContours(InputOutputArray image, OutputArrayOfArrays contours, OutputArray hierarchy, int mode, int method, Point offset=Point());
    findContours(edge_image, contours, hierarchy, RETR_TREE/*RETR_EXTERNAL*//*RETR_TREE*/, CHAIN_APPROX_NONE, Point(0, 0));

    for (int i = 0; i< contours.size(); i++) {
        vector<Point> contour = contours[i];

        //cout << matchShapes(contour, circle_contour, CV_CONTOURS_MATCH_I1, 0);
    //approxPolyDP(Mat(contours[i]), approx[k], 3, true);
        const float epsilon = 0.01f;

        vector<Point> approx;
        approxPolyDP(contour, approx, epsilon * arcLength(contour, true), true);
             /*   if (approx.size() > test_thresh && hierarchy[i][2] > 0 && !isContourConvex(contour)) {
            Rect rect = boundingRect(contour);
            lights.push_back(rect);
            RotatedRect rotatedRect = fitEllipse(contour);
            float rr_ratio = rotatedRect.size.width / rotatedRect.size.height;
            if (rr_ratio > 0.6) {*/

        if (approx.size() > test_thresh && hierarchy[i][2] > 0 &&
    //C++: double matchShapes(InputArray contour1, InputArray contour2, int method, double parameter)
            matchShapes(contour, circle_contour, CV_CONTOURS_MATCH_I1, 0) < 0.1) {
      //      /*isContourConvex(contour)*/) {

            Rect light = boundingRect(contour);
            //rectangle(source, point, Point(point.x + model.cols, point.y + model.rows), color, 2, 8, 0);

            //Mat hist;
            int white, black;
            Mat bin_image = b_proj_lights_map_green(light);
            getBlackVsWhitePixels(bin_image, white, black);

            //threshold(bin_image, bin_image, 0, 255, THRESH_BINARY);

            //imshow("Potential Light: " + to_string(i), bin_image);
            //getHist(bin_image, hist);

            //float x = ; //white ratio
            //cout << "H Potential Light: i: " << hist.at<float>(1) << " vs " << hist.at<float>(0) << endl;
            //cout << "> Potential Light: i: " << white << " vs " << black << endl;
            if (1.0f * white / black > 0.7f) {
            
                /*if (hist.at<float>(0) < best_hist && point.x < source.cols/2) {
                    best_hist = hist.at<float>(0);
                    best_model = point;
                }*/

                lights_g.push_back(light);

                /*//TODO remove
                for (int p = 0; p < contour.size(); p++) {
                    Point point = contour[p];
                    //Vec3b color = image.at<Vec3b>(Point(x,y));
                    drawing.at<Vec3b>(point) = Vec3b(0, 0, 255);
                }*/    
                drawContours(drawing, contours, i, Scalar(74, 195, 139), CV_FILLED, 4, hierarchy, 0, Point());
                continue; //since green is unique we don't need to check for red
            }

            bin_image = b_proj_lights_map_amber(light);
            getBlackVsWhitePixels(bin_image, white, black); 
            if (1.0f * white / black > 0.5f) {
                lights_a.push_back(light);

                drawContours(drawing, contours, i, /*Scalar(255, 0, 0)*/ Scalar(7, 193, 255), CV_FILLED, 4, hierarchy, 0, Point());
                continue;
            }

            bin_image = b_proj_lights_map_red(light);
            getBlackVsWhitePixels(bin_image, white, black);
            if (1.0f * white / black > 0.7f) {
                lights_r.push_back(light);

                drawContours(drawing, contours, i, Scalar(54, 67, 244), CV_FILLED, 4, hierarchy, 0, Point());
            }
        }
    }
/*
/// Get the moments
  vector<Moments> mu(contours.size() );
  for( int i = 0; i < contours.size(); i++ )
     { mu[i] = moments( contours[i], false ); }

  ///  Get the mass centers:
  vector<Point2f> mc( contours.size() );
  for( int i = 0; i < contours.size(); i++ )
     { mc[i] = Point2f( mu[i].m10/mu[i].m00 , mu[i].m01/mu[i].m00 ); }

  /// Draw contours
 // Mat drawing = Mat::zeros( canny_output.size(), CV_8UC3 );
  for( int i = 0; i< contours.size(); i++ )
     {
        cout << mc[i] << endl;
       Scalar color = Scalar( rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255) );
       drawContours( drawing, contours, i, color, 2, 8, hierarchy, 0, Point() );
       circle( drawing, mc[i], 4, color, -1, 8, 0 );
     }

    for (size_t i = 0; i < contours.size(); i++) {
        //C++: void approxPolyDP(InputArray curve, OutputArray approxCurve, double epsilon, bool closed)¶
        //C++: double arcLength(InputArray curve, bool closed);
        vector<Point> contour = contours[i];

    }*/
        //const float epsilon = 0.01f;
        //vector<Point> approx;
        //approxPolyDP(contour, approx, epsilon * arcLength(contour, true), true);

      //Moments  moments(contour);

        //approxPolyDP(Mat(contours[i]), approx[k], 3, true);
    //    if (
//C++: double matchShapes(InputArray contour1, InputArray contour2, int method, double parameter)
//matchShapes(contour, circle_contour, CV_CONTOURS_MATCH_I1)

  //      /*isContourConvex(contour)*/) {

//cout << matchShapes(contour, circle_contour, CV_CONTOURS_MATCH_I1) << endl;
            /*Rect rect = boundingRect(contour);
            lights.push_back(rect);
            RotatedRect rotatedRect = fitEllipse(contour);
            float rr_ratio = rotatedRect.size.width / rotatedRect.size.height;
            if (rr_ratio > 0.6) {*/
                //Scalar color = Scalar(rng.uniform(50, 255), rng.uniform(50, 255), rng.uniform(50, 255));
//                ellipse(drawing, fitEllipse(contours[i]), color, 2);//, int thickness=1, int lineType=8);

                /*for (int j = 0; j < contour.size(); j++) {
                    drawing.at<uchar>(contour[j].y, contour[j].x) = color.val[0];
                }*/
                //ellipse(drawing, rotatedRect, color);//, int thickness=1, int lineType=8);
               // cout << rotatedRect << endl;
            //cout << "(x, y): " << rotatedRect.size.width << ", " <<  << endl; 

            //purely for drawing
//            for (size_t i = 0; i < contours.size(); i++) {
           

               // We take the edges that OpenCV calculated for us
       /*     cv::Point2f vertices2f[4];
            rotatedRect.points(vertices2f);

            // Convert them so we can use them in a fillConvexPoly
            cv::Point vertices[4];    
            for (int i = 0; i < 4; ++i) {
                vertices[i] = vertices2f[i];
            }*/

            // Now we can fill the rotated rectangle with our specified color
         /*   cv::fillConvexPoly(drawing,
                               vertices,
                               4,
                               color);*/
                            //drawContours(drawing, contours, i, color, 2);
       //    drawContours(drawing, contours, i, Scalar(0, 0, 255), CV_FILLED, 8, hierarchy, 2, Point());   } 
        

//            rectangle(drawing, rect, color);//, int thickness=1, int lineType=8, int shift=0 )¶

        
            //ellipse(drawing, fitEllipse(contours[i]), color);//, int thickness=1, int lineType=8);
  //          }
      //  }
    

    
//void cv::drawContours (   InputOutputArray    image,
/*InputArrayOfArrays  contours,
int     contourIdx,
const Scalar &  color,
int     thickness = 1,
int     lineType = LINE_8,
InputArray  hierarchy = noArray(),
int     maxLevel = INT_MAX,
Point   offset = Point() 
)   */    

    namedWindow("Back Project Drawing", CV_WINDOW_AUTOSIZE);
    createTrackbar("Approx Poly:", "Back Project Drawing", &test_thresh, 24, theFunction);
    imshow("Back Project Drawing", drawing);


    //imshow("Back Project Drawing", drawing);
}

void theFunction(int, void *) {

    int const max_low_threshold = 100;
    int ratio1 = 4;
    int kernel_size = 3;
    
    Mat chamfer_image;
    Mat chamfer_image_normalized;

    Mat src_gray;
    Mat edge_image;

    char buf[1024];
    sprintf(buf, "CS4053/CamVidLights/CamVidLights%s.png", file_number);
    Mat source = imread(buf, CV_LOAD_IMAGE_COLOR);
    Mat model_o = imread("CS4053/Template-TrafficLight02.png");
    Mat model;

    float h_ratio = 0.20f, w_ratio = 0.40f; // light to frame (inc borders)
    int model_p_height = INT_MAX, model_p_width = INT_MAX; //model ratio
    //test

    cvtColor(source, src_gray, CV_BGR2GRAY);

    /// Reduce noise with a kernel 3x3
    //    C++: void GaussianBlur(InputArray src, OutputArray dst, Size ksize, double sigmaX, double sigmaY=0, int borderType=BORDER_DEFAULT )
    
    Mat disc = getStructuringElement(MORPH_ELLIPSE, Size(2, 2), Point(-1,-1)); //disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    
    //filter2D(src_gray, src_gray, -1, disc, Point(-1, -1), 0, BORDER_DEFAULT);
    GaussianBlur(src_gray, src_gray, cv::Size(0, 0), 1.5);
    addWeighted(src_gray, 3.5, src_gray, -1.5, 0, src_gray);

    //blur(src_gray, src_gray, Size(2,2)); //TODO try gaussian blur?
    //filter2D(src_gray, src_gray, -1, disc, Point(-1, -1), 0, BORDER_DEFAULT);
    //blur(src_gray, src_gray, Size(2,2)); //TODO try gaussian blur?
    
    imshow("Source Gray", src_gray);


    low_threshold = 50;
        /// Canny detector
    
    Canny(src_gray, edge_image, low_threshold, low_threshold * ratio1, kernel_size);
    
    vector<Rect> lights_r; //could use RotatedRect (for ellipses)
    vector<Rect> lights_a; //could use RotatedRect (for ellipses)
    vector<Rect> lights_g; //could use RotatedRect (for ellipses)
    findLights(source, edge_image, lights_r, lights_a, lights_g, source);

    /*for (int i = 0; i < lights.size(); i++) {
        float wh_ratio = 1.0f * model_o.rows / model_o.cols; // h / w
        model_p_height = min(model_p_height, lights[i].height / h_ratio);
        model_p_width = min(model_p_width, lights[i].width / w_ratio);
    }

    resize(model_o, model_o, Size(model_p_width, model_p_height));    imshow("Model Resized", model_o);
    */

    threshold(edge_image, edge_image, 127, 255, THRESH_BINARY_INV);
    distanceTransform(edge_image, chamfer_image, CV_DIST_L2, 3);
    normalize(chamfer_image, chamfer_image, 0, 1.0, NORM_MINMAX);
    //imshow("Edge Map", chamfer_image_normalized);

    imshow("Edge Image", edge_image);
    imshow("Chamfer Image", chamfer_image);

    namedWindow("Local Minima", CV_WINDOW_AUTOSIZE);
    createTrackbar("Local Minima Value:", "Local Minima", &local_minima_threshold, 100, theFunction);

    int model_width = model_o.cols;
    int model_height = model_o.rows;

    for (float i = 1; i > 0.6; i = i - 0.5) {
        Mat matching_image;
        Mat model;

        resize(model_o, model, Size(i * model_width, i * model_height));

        //Canny(model, model_edges, low_threshold, low_threshold * ratio1, kernel_size);
        threshold(model, model, 127, 255, THRESH_BINARY);
        imshow("Model Edges", model);
        vector<Point> local_minima;

        chamferMatching(chamfer_image, model, matching_image);
        for(int i=0; i<matching_image.rows; i++)
            for(int j=0; j<matching_image.cols; j++) {
                float p = matching_image.at<float>(i,j);
              //  if (p > 0) {
                    //Point matchLoc = Point(j, i);
                if (p < 400) {
                    cout << "matching image pixel weight: " << p << endl;
                    local_minima.push_back(Point(j, i));
                }
                   // rectangle(source, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 2, 8, 0);
                    //rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 2, 8, 0);
                    //rectangle(dst, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 2, 8, 0);
                //}
            }

        //findLocalMinima(matching_image, local_minima, 1.0f * local_minima_threshold / 100);
        //normalize(local_minima, local_minima, 0, 255, NORM_MINMAX);
        //imshow("Local Minima", local_minima);

        /*for (int i = 0; i < local_minima.rows; i++)
            for (int j = 0; j < local_minima.cols; j++) {
                int p = local_minima.at<uchar>(i, j);
                if (p > 0) {
                    Point matchLoc = Point(j, i);
                    //printf("pixel val: %d\n", p);
                    rectangle(source, matchLoc, Point(matchLoc.x + model.cols, matchLoc.y + model.rows), Scalar(0, 0, 255), 2, 8, 0);
                    //rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 2, 8, 0);
                    //rectangle(dst, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 2, 8, 0);
                }
            }*/
        
        /*for (vector<Point>::size_type i = 0; i != local_minima.size(); i++) {
            Point point = local_minima[i];
            int mw = model.cols;
            int mh = model.rows;

            bool found = false;
            for (int i = 0; i < local_minima.rows; i++)
                for (int j = 0; j < local_minima.cols; j++) {
                    int p = local_minima.at<uchar>(i, j);
                    found = found || (p > 0);
                }

            if (!found) {
                local_minima.erase(i);
            }
        }*/

        vector<Point> local_minimas_filtered;

        for (vector<Point>::iterator it = local_minima.begin(); it != local_minima.end(); it++) {
        //for (vector<Point>::size_type i = 0; i != local_minima.size(); i++) {
            Point p_model = *it;//local_minima.at(it);
            int model_width = model.cols;
            int model_height = model.rows;

            bool found = false;
            /*for (int y = p_model.y; y < min(lights_map.rows, p_model.y + mh); y++)
                for (int x = p_model.x; x < min(lights_map.cols, p_model.x + mw); x++) {
                    int p = lights_map.at<uchar>(y, x);
                    found = found || (p > 0);
                }*/

            for (int i = 0; i < lights_g.size(); i++) {
                Rect light = lights_g[i];
                if (p_model.x < light.x && p_model.x + model_width > light.x + light.width
                    && p_model.y < light.y && p_model.y + model_height > light.y + light.height) {
                    found = true;
                    break;
                }
            }

            if (found) {
                local_minimas_filtered.push_back(p_model);
                //local_minima.erase(it);
            }
        }

        Point best_model;
        float best_ratio = 0;

        //TODO merge with above
        for (vector<Point>::size_type i = 0; i != local_minimas_filtered.size(); i++) {
            Point point = local_minimas_filtered[i];
            Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
            rectangle(source, point, Point(point.x + model.cols, point.y + model.rows), color, 2, 8, 0);
            //Mat hist;

            // Crop the full image to that image contained by the rectangle myROI
            // Note that this doesn't copy the data
            //cv::Mat croppedImage = image(myROI);
            Mat bin_image = src_gray(Rect(point.x, point.y, model.cols, model.rows));
            threshold(bin_image, bin_image, 10, 255, THRESH_BINARY);
            int white, black;
            getBlackVsWhitePixels(bin_image, white, black);
            //cout << "i: " << hist << endl;
            float bw_ratio = black / (white+black);
            if (bw_ratio > best_ratio) {
                best_ratio = bw_ratio;
                best_model = point;
            }
        }

        Point point = best_model;
        rectangle(source, point, Point(point.x + model.cols, point.y + model.rows), Scalar(0, 0, 255), 2, 8, 0);

          /// Draw for each channel

        /*for (vector<Point>::iterator it = local_minima.begin(); it != local_minima.end(); it++) {
             {
                        Point matchLoc = Point(j, i);
                        //printf("pixel val: %d\n", p);
                        bool found = false;

                        if (found)
                            rectangle(source, matchLoc, Point(matchLoc.x + model.cols, matchLoc.y + model.rows), Scalar(0, 0, 255), 2, 8, 0);
                        //rectangle(result, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 2, 8, 0);
                        //rectangle(dst, matchLoc, Point(matchLoc.x + templ.cols, matchLoc.y + templ.rows), Scalar(0, 0, 255), 2, 8, 0);
                    }
        }*/
    }

    namedWindow("Matching Image", CV_WINDOW_AUTOSIZE);
    createTrackbar("Low Threshold:", "Matching Image", &low_threshold, max_low_threshold, theFunction);
    imshow("Matching Image", source);
}

int main( int argc, char** argv ) {

    file_number = argv[1];
    theFunction(0, 0);

    //namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    //imshow( "Display window", model );                   // Show our image inside it.

    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
