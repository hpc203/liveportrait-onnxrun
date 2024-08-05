#include "utils_crop.h"

using namespace std;
using namespace cv;

void crop_image(const Mat &img, const float *pts, const int num_pts, std::map<string, Mat> &ret_dct, const int dsize, const float scale, const float vy_ratio)
{
    ////_estimate_similar_transform_from_pt
    const bool flag_do_rot = true;
    const bool use_lip = true;
    const bool need_square = true;
    const float vx_ratio = 0.f;
    const bool use_deg_flag = false;

    float pt_left_eye[2] = {0.f, 0.f};
    const int left_inds[4] = {33, 35, 40, 39};
    for (int i = 0; i < 4; i++)
    {
        pt_left_eye[0] += pts[left_inds[i] * 2];
        pt_left_eye[1] += pts[left_inds[i] * 2 + 1];
    }
    pt_left_eye[0] /= 4;
    pt_left_eye[1] /= 4;
    const int right_inds[4] = {87, 89, 94, 93};
    float pt_right_eye[2] = {0.f, 0.f};
    for (int i = 0; i < 4; i++)
    {
        pt_right_eye[0] += pts[right_inds[i] * 2];
        pt_right_eye[1] += pts[right_inds[i] * 2 + 1];
    }
    pt_right_eye[0] /= 4;
    pt_right_eye[1] /= 4;
    ////已调试验证通过

    float pt2[2 * 2];
    if (use_lip)
    {
        pt2[0] = (pt_left_eye[0] + pt_right_eye[0]) * 0.5;
        pt2[1] = (pt_left_eye[1] + pt_right_eye[1]) * 0.5;
        pt2[2] = (pts[52 * 2] + pts[61 * 2]) * 0.5;
        pt2[3] = (pts[52 * 2 + 1] + pts[61 * 2 + 1]) * 0.5;
    }
    else
    {
        pt2[0] = pt_left_eye[0];
        pt2[1] = pt_left_eye[1];
        pt2[2] = pt_right_eye[0];
        pt2[3] = pt_right_eye[1];

        const float v[2] = {pt2[2] - pt2[0], pt2[3] - pt2[1]};
        pt2[2] = pt2[0] - v[1];
        pt2[3] = pt2[1] + v[0];
    }

    float uy[2] = {pt2[2] - pt2[0], pt2[3] - pt2[1]};
    float l = sqrt(uy[0] * uy[0] + uy[1] * uy[1]);
    if (l <= 1e-3)
    {
        uy[0] = 0;
        uy[1] = 1;
    }
    else
    {
        uy[0] /= l;
        uy[1] /= l;
    }
    float ux[2] = {uy[1], -uy[0]};
    float angle = acos(ux[0]);
    if (ux[1] < 0)
    {
        angle = -angle;
    }

    float M_T[2 * 2] = {ux[0], uy[0], ux[1], uy[1]};
    float center0[2] = {0, 0};
    for (int i = 0; i < num_pts; i++)
    {
        center0[0] += pts[i * 2];
        center0[1] += pts[i * 2 + 1];
    }
    center0[0] /= num_pts;
    center0[1] /= num_pts;
    float lt_pt[2] = {10000, 10000};   ////lt_pt = np.min(rpts, axis=0)
    float rb_pt[2] = {-10000, -10000}; ////rb_pt = np.max(rpts, axis=0)
    for (int i = 0; i < num_pts; i++)
    {
        const float x = pts[i * 2] - center0[0];
        const float y = pts[i * 2 + 1] - center0[1];
        float rpts_x = x * M_T[0] + y * M_T[2];
        float rpts_y = x * M_T[1] + y * M_T[3];

        lt_pt[0] = min(lt_pt[0], rpts_x);
        lt_pt[1] = min(lt_pt[1], rpts_y);
        rb_pt[0] = max(rb_pt[0], rpts_x);
        rb_pt[1] = max(rb_pt[1], rpts_y);
    }
    float center1[2] = {(lt_pt[0] + rb_pt[0]) * 0.5f, (lt_pt[1] + rb_pt[1]) * 0.5f};
    float size[2] = {rb_pt[0] - lt_pt[0], rb_pt[1] - lt_pt[1]};
    if (need_square)
    {
        float m = max(size[0], size[1]);
        size[0] = m;
        size[1] = m;
    }
    size[0] *= scale;
    size[1] *= scale;
    float center[2] = {center0[0] + ux[0] * center1[0] + uy[0] * center1[1], center0[1] + ux[1] * center1[0] + uy[1] * center1[1]};
    center[0] = center[0] + ux[0] * vx_ratio * size[0] + uy[0] * vy_ratio * size[0];
    center[1] = center[1] + ux[1] * vx_ratio * size[1] + uy[1] * vy_ratio * size[1];
    if (use_deg_flag)
    {
        angle = angle * 180.f / PI; ////degrees
    }

    float s = (float)dsize / size[0];
    float tgt_center[2] = {dsize * 0.5f, dsize * 0.5f};

    Mat M_INV;
    if (flag_do_rot)
    {
        float costheta = cos(angle), sintheta = sin(angle);
        float cx = center[0], cy = center[1];
        float tcx = tgt_center[0], tcy = tgt_center[1];
        M_INV = (Mat_<float>(2, 3) << s * costheta, s * sintheta, tcx - s * (costheta * cx + sintheta * cy), -s * sintheta, s * costheta, tcy - s * (-sintheta * cx + costheta * cy));
    }
    else
    {
        M_INV = (Mat_<float>(2, 3) << s, 0, tgt_center[0] - s * center[0], 0, s, tgt_center[1] - s * center[1]);
    }
    ////_estimate_similar_transform_from_pts
    ////_transform_img
    Mat img_crop;
    warpAffine(img, img_crop, M_INV, cv::Size(dsize, dsize));
    ////_transform_img
    ////_transform_pts
    Mat pt_crop = Mat::zeros(num_pts, 2, CV_32FC1);
    for (int i = 0; i < num_pts; i++)
    {
        pt_crop.at<float>(i, 0) = pts[i * 2] * M_INV.at<float>(0, 0) + pts[i * 2 + 1] * M_INV.at<float>(0, 1) + M_INV.at<float>(0, 2);
        pt_crop.at<float>(i, 1) = pts[i * 2] * M_INV.at<float>(1, 0) + pts[i * 2 + 1] * M_INV.at<float>(1, 1) + M_INV.at<float>(1, 2);
    }
    ////_transform_pts
    Mat M_o2c = Mat::eye(3, 3, CV_32FC1);
    M_INV.copyTo(M_o2c.rowRange(0, 2));
    Mat M_c2o = M_o2c.inv();

    ret_dct["M_o2c"] = M_o2c;
    ret_dct["M_c2o"] = M_c2o;
    ret_dct["img_crop"] = img_crop;
    ret_dct["pt_crop"] = pt_crop;
}

Mat src_preprocess(const Mat &img)
{
    const int h = img.rows, w = img.cols;
    const int max_dim = 1280;
    const int max_hw = max(h, w);
    Mat resize_img = img.clone();
    if (max_hw > max_dim)
    {
        int new_w = max_dim;
        int new_h = int((float)h * ((float)max_dim / w));
        if (h > w)
        {
            new_h = max_dim;
            new_w = int((float)w * ((float)max_dim / h));
        }
        resize(img, resize_img, Size(new_w, new_h));
    }

    const int division = 2;
    int new_h = resize_img.rows - (resize_img.rows % division);
    int new_w = resize_img.cols - (resize_img.cols % division);

    if (new_h == 0 || new_w == 0)
    {
        return resize_img;
    }

    if (new_h != resize_img.rows or new_w != resize_img.cols)
    {
        Mat new_img;
        resize_img(Rect(0, 0, new_w, new_h)).copyTo(new_img);
        return new_img;
    }
    return resize_img;
}

void preprocess(const Mat &img, vector<float> &input_tensor, vector<int64_t> &input_shape)
{
    const int h = img.rows;
    const int w = img.cols;
    input_shape = {1, 3, h, w};
    const int image_area = h * w;
    input_tensor.clear();
    input_tensor.resize(input_shape[0] * input_shape[1] * image_area);
    vector<Mat> bgrChannels(3);
    split(img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 255.f);
        bgrChannels[c].setTo(0, bgrChannels[c] < 0);
        bgrChannels[c].setTo(1, bgrChannels[c] > 1);
    }

    size_t single_chn_size = image_area * sizeof(float);
    memcpy(input_tensor.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(input_tensor.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(input_tensor.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
}

void softmax(const float *x, const vector<int64_t> shape, vector<float> &y)
{
    const int64_t batch_size = shape[0];
    const int64_t len = shape[1];
    y.clear();
    y.resize(batch_size * len);
    for (int i = 0; i < batch_size; i++)
    {
        float sum = 0;
        for (int j = 0; j < len; j++)
        {
            sum += exp(x[i * len + j]);
        }
        for (int j = 0; j < len; j++)
        {
            y[i * len + j] = exp(x[i * len + j]) / sum;
        }
    }
}

/// 传值引用避免参数拷贝
Mat get_rotation_matrix(const Mat &pitch_, const Mat &yaw_, const Mat &roll_)
{
    float x = pitch_.at<float>(0, 0) / 180.f * PI;
    float y = yaw_.at<float>(0, 0) / 180.f * PI;
    float z = roll_.at<float>(0, 0) / 180.f * PI;
    ////batchsize=1，不考虑batchsize
    Mat rot_x = (Mat_<float>(3, 3) << 1, 0, 0, 0, cos(x), -sin(x), 0, sin(x), cos(x));
    Mat rot_y = (Mat_<float>(3, 3) << cos(y), 0, sin(y), 0, 1, 0, -sin(y), 0, cos(y));
    Mat rot_z = (Mat_<float>(3, 3) << cos(z), -sin(z), 0, sin(z), cos(z), 0, 0, 0, 1);
    Mat rot = rot_z * rot_y * rot_x;
    rot = rot.t();
    // vector<int> newshape = {1, 3, 3};   ////由于在c++的opencv里不支持3维Mat的矩阵乘法,此处不考虑batchsize维度
    // rot.reshape(0, newshape);
    return rot;
}

Mat transform_keypoint(std::map<string, Mat> kp_info)
{
    const int num_kp = kp_info["kp"].size[0]; ////不考虑batchsize
    Mat rot_mat = get_rotation_matrix(kp_info["pitch"], kp_info["yaw"], kp_info["roll"]);
    Mat kp_transformed = kp_info["kp"] * rot_mat + kp_info["exp"];
    kp_transformed *= kp_info["scale"].at<float>(0, 0);
    for (int i = 0; i < num_kp; i++)
    {
        kp_transformed.at<float>(i, 0) += kp_info["t"].at<float>(0, 0);
        kp_transformed.at<float>(i, 1) += kp_info["t"].at<float>(0, 1);
    }

    return kp_transformed;
}

Mat prepare_paste_back(const Mat &mask_crop, const Mat &crop_M_c2o, Size dsize)
{
    Mat mask_ori;
    warpAffine(mask_crop, mask_ori, crop_M_c2o.rowRange(0, 2), dsize);
    mask_ori.convertTo(mask_ori, CV_32FC3, 1 / 255.0);
    return mask_ori;
}

void concat_frame(const Mat &driving_img, const Mat &src_img, const Mat &I_p, Mat &out)
{
    const int h = I_p.rows;
    const int w = I_p.cols;
    Size dsize = Size(w, h);
    vector<Mat> himgs;
    Mat temp_img;
    resize(driving_img, temp_img, dsize);
    himgs.push_back(temp_img);
    resize(src_img, temp_img, dsize);
    himgs.push_back(temp_img);
    himgs.push_back(I_p);
    cv::hconcat(himgs, out);
}

void paste_back(const Mat &img_crop, const Mat &M_c2o, const Mat &img_ori, const Mat &mask_ori, Mat &result)
{
    Size dsize = Size(img_ori.cols, img_ori.rows);
    warpAffine(img_crop, result, M_c2o.rowRange(0, 2), dsize);
    vector<Mat> rgb_result(3);
    split(result, rgb_result);
    vector<Mat> rgb_mask_ori(3);
    split(mask_ori, rgb_mask_ori);
    vector<Mat> rgb_img_ori(3);
    split(img_ori, rgb_img_ori);
    vector<Mat> rgbs(3);
    for (int c = 0; c < 3; c++)
    {
        rgb_result[c].convertTo(rgb_result[c], CV_32FC1); ////注意数据类型转换，不然在下面的矩阵点乘运算时会报错的
        rgb_img_ori[c].convertTo(rgb_img_ori[c], CV_32FC1);
        
        rgbs[c] = rgb_mask_ori[c].mul(rgb_result[c]) + (1 - rgb_mask_ori[c]).mul(rgb_img_ori[c]);
        rgbs[c].setTo(0, rgbs[c] < 0);
        rgbs[c].setTo(255, rgbs[c] > 255);
    }
    merge(rgbs, result);
    result.convertTo(result, CV_8UC3);
}

float calculate_distance_ratio(const Mat &lmk, const int idx1, const int idx2, const int idx3, const int idx4, const float eps)
{
    float d1 = sqrt(pow(lmk.at<float>(idx1, 0) - lmk.at<float>(idx2, 0), 2) + pow(lmk.at<float>(idx1, 1) - lmk.at<float>(idx2, 1), 2));
    float d2 = sqrt(pow(lmk.at<float>(idx3, 0) - lmk.at<float>(idx4, 0), 2) + pow(lmk.at<float>(idx3, 1) - lmk.at<float>(idx4, 1), 2));
    return d1 / (d2 + eps);
}

float GetIoU(const Bbox box1, const Bbox box2)
{
    float x1 = max(box1.xmin, box2.xmin);
    float y1 = max(box1.ymin, box2.ymin);
    float x2 = min(box1.xmax, box2.xmax);
    float y2 = min(box1.ymax, box2.ymax);
    float w = max(0.f, x2 - x1);
    float h = max(0.f, y2 - y1);
    float over_area = w * h;
    if (over_area == 0)
        return 0.0;
    float union_area = (box1.xmax - box1.xmin) * (box1.ymax - box1.ymin) + (box2.xmax - box2.xmin) * (box2.ymax - box2.ymin) - over_area;
    return over_area / union_area;
}

void nms_boxes(vector<Bbox> &boxes, const float nms_thresh)
{
    sort(boxes.begin(), boxes.end(), [](Bbox a, Bbox b)
         { return a.score > b.score; });
    const int num_box = boxes.size();
    vector<bool> isSuppressed(num_box, false);
    for (int i = 0; i < num_box; ++i)
    {
        if (isSuppressed[i])
        {
            continue;
        }
        for (int j = i + 1; j < num_box; ++j)
        {
            if (isSuppressed[j])
            {
                continue;
            }

            float ovr = GetIoU(boxes[i], boxes[j]);
            if (ovr >= nms_thresh)
            {
                isSuppressed[j] = true;
            }
        }
    }

    int idx_t = 0;
    boxes.erase(remove_if(boxes.begin(), boxes.end(), [&idx_t, &isSuppressed](const Bbox &f)
                          { return isSuppressed[idx_t++]; }),
                boxes.end());
}