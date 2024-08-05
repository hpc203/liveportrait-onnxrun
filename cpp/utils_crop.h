#ifndef UTILS
#define UTILS
#include <iostream>
#include <string>
#include <map>
#include <algorithm>
#include <vector>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#define PI 3.14159265358979323846

typedef struct
{
    float xmin;
    float ymin;
    float xmax;
    float ymax;
    float score;
    float kps[10];
    float landmark_2d_106[106 * 2];
} Bbox;

void crop_image(const cv::Mat &img, const float *pts, const int num_pts, std::map<std::string, cv::Mat> &ret_dct, const int dsize, const float scale, const float vy_ratio);
cv::Mat src_preprocess(const cv::Mat &img);
void preprocess(const cv::Mat &img, std::vector<float> &input_tensor, std::vector<int64_t> &input_shape);
void softmax(const float *x, const std::vector<int64_t> shape, std::vector<float> &y);
cv::Mat get_rotation_matrix(const cv::Mat &pitch_, const cv::Mat &yaw_, const cv::Mat &roll_);
cv::Mat transform_keypoint(std::map<std::string, cv::Mat> kp_info);
cv::Mat prepare_paste_back(const cv::Mat &mask_crop, const cv::Mat &crop_M_c2o, cv::Size dsize);
float calculate_distance_ratio(const cv::Mat &lmk, const int idx1, const int idx2, const int idx3, const int idx4, const float eps = 1e-6);
void concat_frame(const cv::Mat &driving_img, const cv::Mat &src_img, const cv::Mat &I_p, cv::Mat &out);
void paste_back(const cv::Mat &img_crop, const cv::Mat &M_c2o, const cv::Mat &img_ori, const cv::Mat &mask_ori, cv::Mat &result);

float GetIoU(const Bbox box1, const Bbox box2);
void nms_boxes(std::vector<Bbox> &boxes, const float nms_thresh);

#endif