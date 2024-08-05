#include "liveportrait.h"
#include <opencv2/highgui.hpp>
#include <numeric>

using namespace cv;
using namespace std;
using namespace Ort;

LivePortraitPipeline::LivePortraitPipeline()
{
    /// OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注释
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);

    string model_path = "/home/wangbo/liveportrait-onnxrun/weights/appearance_feature_extractor.onnx";
    appearance_feature_extractor_ort_session = new Session(env, model_path.c_str(), sessionOptions);

    model_path = "/home/wangbo/liveportrait-onnxrun/weights/motion_extractor.onnx";
    motion_extractor_ort_session = new Session(env, model_path.c_str(), sessionOptions);

    model_path = "/home/wangbo/liveportrait-onnxrun/weights/warping_spade.onnx";
    warping_spade_ort_session = new Session(env, model_path.c_str(), sessionOptions);

    model_path = "/home/wangbo/liveportrait-onnxrun/weights/stitching.onnx";
    stitching_module_ort_session = new Session(env, model_path.c_str(), sessionOptions);

    model_path = "/home/wangbo/liveportrait-onnxrun/weights/landmark.onnx";
    /// std::wstring widestr = std::wstring(model_path.begin(), model_path.end());  ////windows写法
    /// landmark_runner_ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
    landmark_runner_ort_session = new Session(env, model_path.c_str(), sessionOptions); ////linux写法
    /// 输出和输出节点名称在头文件里写死,在这里就不调用函数获取了

    this->face_analysis = std::make_shared<FaceAnalysis>("/home/wangbo/liveportrait-onnxrun/weights/retinaface_det_static.onnx", "/home/wangbo/liveportrait-onnxrun/weights/face_2dpose_106_static.onnx");

    this->mask_crop = imread("/home/wangbo/liveportrait-onnxrun/mask_template.png");
    cvtColor(this->mask_crop, this->mask_crop, COLOR_BGRA2BGR);
}

Mat LivePortraitPipeline::landmark_runner(const Mat &img, const float *lmk)
{
    std::map<string, Mat> crop_dct;
    crop_image(img, lmk, 106, crop_dct, 224, 1.5, -0.1);

    std::vector<int64_t> input_img_shape = {1, 3, crop_dct["img_crop"].rows, crop_dct["img_crop"].cols};
    vector<cv::Mat> bgrChannels(3);
    split(crop_dct["img_crop"], bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 255.0);
    }
    const int image_area = input_img_shape[2] * input_img_shape[3];
    this->landmark_runner_input_tensor.clear();
    this->landmark_runner_input_tensor.resize(input_img_shape[0] * input_img_shape[1] * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->landmark_runner_input_tensor.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->landmark_runner_input_tensor.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->landmark_runner_input_tensor.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);

    Value input_tensor = Value::CreateTensor<float>(memory_info_handler, this->landmark_runner_input_tensor.data(), this->landmark_runner_input_tensor.size(), input_img_shape.data(), input_img_shape.size());
    vector<Value> ort_outputs = this->landmark_runner_ort_session->Run(runOptions, this->landmark_runner_input_names.data(), &input_tensor, 1, this->landmark_runner_output_names.data(), this->landmark_runner_output_names.size());
    float *out_pts = ort_outputs[2].GetTensorMutableData<float>();
    const int num_pts = ort_outputs[2].GetTensorTypeAndShapeInfo().GetShape()[1] / 2;
    Mat lmk_mat(num_pts, 2, CV_32FC1);
    for (int i = 0; i < num_pts; i++)
    { ////btachsize=1，不考虑batchsize维度
        lmk_mat.at<float>(i, 0) = out_pts[i * 2] * 224 * crop_dct["M_c2o"].at<float>(0, 0) + out_pts[i * 2 + 1] * 224 * crop_dct["M_c2o"].at<float>(0, 1) + crop_dct["M_c2o"].at<float>(0, 2);
        lmk_mat.at<float>(i, 1) = out_pts[i * 2] * 224 * crop_dct["M_c2o"].at<float>(1, 0) + out_pts[i * 2 + 1] * 224 * crop_dct["M_c2o"].at<float>(1, 1) + crop_dct["M_c2o"].at<float>(1, 2);
    }
    return lmk_mat;
}

void LivePortraitPipeline::crop_src_image(const Mat &srcimg, std::map<string, Mat> &crop_info)
{
    vector<Bbox> boxes = this->face_analysis->detect(srcimg);
    if (boxes.size() == 0)
    {
        cout << "No face detected in the source image." << endl;
        return;
    }
    else if (boxes.size() > 1)
    {
        cout << "More than one face detected in the image, only pick one face." << endl;
        return;
    }

    Bbox src_face = boxes[0];
    float *lmk = src_face.landmark_2d_106;

    crop_image(srcimg, lmk, 106, crop_info, 512, 2.3, -0.125);

    Mat lmk_crop = this->landmark_runner(srcimg, lmk);
    
    crop_info["lmk_crop"] = lmk_crop;
    Mat img_crop_256x256;
    resize(crop_info["img_crop"], img_crop_256x256, Size(256, 256), INTER_AREA);
    crop_info["img_crop_256x256"] = img_crop_256x256;
    crop_info["lmk_crop_256x256"] = crop_info["lmk_crop"] * 256 / 512;
}

void LivePortraitPipeline::get_kp_info(vector<float> x, vector<int64_t> shape, std::map<string, Mat> &kp_info)
{
    Value input_tensor = Value::CreateTensor<float>(memory_info_handler, x.data(), x.size(), shape.data(), shape.size());
    vector<Value> ort_outputs = this->motion_extractor_ort_session->Run(runOptions, this->motion_extractor_input_names.data(), &input_tensor, 1, this->motion_extractor_output_names.data(), this->motion_extractor_output_names.size());
    ////pitch, yaw, roll, t, exp, scale, kp
    float *pitch = ort_outputs[0].GetTensorMutableData<float>();
    float *yaw = ort_outputs[1].GetTensorMutableData<float>();
    float *roll = ort_outputs[2].GetTensorMutableData<float>();
    float *t = ort_outputs[3].GetTensorMutableData<float>();
    float *exp = ort_outputs[4].GetTensorMutableData<float>();
    float *scale = ort_outputs[5].GetTensorMutableData<float>();
    float *kp = ort_outputs[6].GetTensorMutableData<float>();

    vector<float> pred;
    softmax(pitch, this->motion_extractor_output_shape[0], pred);
    ////const int bs = this->motion_extractor_output_shape[0][0];  ////batchsize=1,不考虑多图片输入
    const int len = 66; ////66
    float sum = 0;
    for (int i = 0; i < len; i++)
    {
        sum += (i * pred[i]);
    }
    float degree = sum * 3 - 97.5;
    kp_info["pitch"] = (Mat_<float>(1, 1) << degree);

    softmax(yaw, this->motion_extractor_output_shape[1], pred);
    sum = 0;
    for (int i = 0; i < len; i++)
    {
        sum += (i * pred[i]);
    }
    degree = sum * 3 - 97.5;
    kp_info["yaw"] = (Mat_<float>(1, 1) << degree);

    softmax(roll, this->motion_extractor_output_shape[2], pred);
    sum = 0;
    for (int i = 0; i < len; i++)
    {
        sum += (i * pred[i]);
    }
    degree = sum * 3 - 97.5;
    kp_info["roll"] = (Mat_<float>(1, 1) << degree);

    kp_info["t"] = Mat(1, 3, CV_32FC1, t);
    ////vector<int> sizes = {1, 21, 3};   ////由于在c++的opencv里不支持3维Mat的矩阵乘法,此处不考虑batchsize维度
    vector<int> sizes = {21, 3};
    kp_info["exp"] = Mat(sizes, CV_32FC1, exp);
    kp_info["scale"] = Mat(1, 1, CV_32FC1, scale);
    kp_info["kp"] = Mat(sizes, CV_32FC1, kp);
}

void LivePortraitPipeline::extract_feature_3d(vector<float> x, vector<int64_t> shape, vector<float> &f_s)
{
    Value input_tensor = Value::CreateTensor<float>(memory_info_handler, x.data(), x.size(), shape.data(), shape.size());
    vector<Value> ort_outputs = this->appearance_feature_extractor_ort_session->Run(runOptions, this->appearance_feature_extractor_input_names.data(), &input_tensor, 1, this->appearance_feature_extractor_output_names.data(), this->appearance_feature_extractor_output_names.size());
    int numel = ort_outputs[0].GetTensorTypeAndShapeInfo().GetElementCount();
    float *out = ort_outputs[0].GetTensorMutableData<float>();

    f_s.clear();
    f_s.resize(numel);
    memcpy(f_s.data(), out, numel * sizeof(float));
}

void LivePortraitPipeline::stitching(const Mat &kp_source, Mat &kp_driving_new)
{
    ////不考虑batchsize维度
    const int num_kp = kp_source.size[0];
    const int numel = kp_source.total();

    const int len = std::accumulate(this->stitching_module_input_shape.begin(), this->stitching_module_input_shape.end(), 1, std::multiplies<int64_t>());
    vector<float> feat(len);
    memcpy(feat.data(), (float *)kp_source.data, numel * sizeof(float));
    memcpy(feat.data() + numel, (float *)kp_driving_new.data, (len - numel) * sizeof(float));

    Value input_tensor = Value::CreateTensor<float>(memory_info_handler, feat.data(), feat.size(), this->stitching_module_input_shape.data(), this->stitching_module_input_shape.size());
    vector<Value> ort_outputs = this->stitching_module_ort_session->Run(runOptions, this->stitching_module_input_names.data(), &input_tensor, 1, this->stitching_module_output_names.data(), this->stitching_module_output_names.size());
    float *delta = ort_outputs[0].GetTensorMutableData<float>();
    const float delta_tx_ty[2] = {delta[num_kp * 3], delta[num_kp * 3 + 1]};
    for (int i = 0; i < num_kp; i++)
    {
        kp_driving_new.at<float>(i, 0) += delta[i * 3];
        kp_driving_new.at<float>(i, 1) += delta[i * 3 + 1];
        kp_driving_new.at<float>(i, 2) += delta[i * 3 + 2];

        kp_driving_new.at<float>(i, 0) += delta_tx_ty[0];
        kp_driving_new.at<float>(i, 1) += delta_tx_ty[1];
    }
}

Mat LivePortraitPipeline::warping_spade(vector<float> feature_3d, const Mat &kp_source, const Mat &kp_driving)
{
    vector<Ort::Value> inputTensors;
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, feature_3d.data(), feature_3d.size(), this->warping_spade_input_shape[0].data(), this->warping_spade_input_shape[0].size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, (float *)kp_driving.data, kp_driving.total(), this->warping_spade_input_shape[1].data(), this->warping_spade_input_shape[1].size()));
    inputTensors.emplace_back(Ort::Value::CreateTensor<float>(memory_info_handler, (float *)kp_source.data, kp_source.total(), this->warping_spade_input_shape[2].data(), this->warping_spade_input_shape[2].size()));
    
    vector<Value> ort_outputs = this->warping_spade_ort_session->Run(runOptions, this->warping_spade_input_names.data(), inputTensors.data(), inputTensors.size(), this->warping_spade_output_names.data(), this->warping_spade_output_names.size());
    float *out = ort_outputs[0].GetTensorMutableData<float>();
    Mat out_mat = Mat(this->warping_spade_output_shape, CV_32FC1, out);
    return out_mat;
}

Mat LivePortraitPipeline::predict(const int frame_id, std::map<string, Mat> x_s_info, const Mat &R_s, vector<float> f_s, const Mat &x_s, const Mat &frame)
{
    Mat lmk;
    if (frame_id > 0)
    {
        lmk = this->landmark_runner(frame, (float*)this->pred_info_lmk.data);
    }
    else
    {
        vector<Bbox> boxes = this->face_analysis->detect(frame);
        if (boxes.size() == 0)
        {
            cout << "No face detected in the frame." << endl;
            exit(-1);
        }
        else if (boxes.size() > 1)
        {
            cout << "More than one face detected in the driving frame, only pick one face." << endl;
            exit(-1);
        }
        Bbox src_face = boxes[0];
        lmk = this->landmark_runner(frame, src_face.landmark_2d_106);
    }
    lmk.copyTo(this->pred_info_lmk);

    float c_d_eyes[2] = {calculate_distance_ratio(lmk, 6, 18, 0, 12), calculate_distance_ratio(lmk, 30, 42, 24, 36)};
    float c_d_lip = calculate_distance_ratio(lmk, 90, 102, 48, 66);

    Mat img;
    resize(frame, img, Size(256, 256));
    vector<float> I_d;
    vector<int64_t> I_d_shape;
    preprocess(img, I_d, I_d_shape);

    std::map<string, Mat> x_d_info;
    this->get_kp_info(I_d, I_d_shape, x_d_info);
    Mat R_d = get_rotation_matrix(x_d_info["pitch"], x_d_info["yaw"], x_d_info["roll"]);
    x_d_info["R_d"] = R_d;
    x_d_info.erase("pitch");
    x_d_info.erase("yaw");
    x_d_info.erase("roll");
    x_d_info.erase("kp");

    if (frame_id == 0)
    {
        this->pred_info_x_d_0_info["scale"] = x_d_info["scale"].clone();   ///也可以定义结构体的方式打包参数
        this->pred_info_x_d_0_info["R_d"] = R_d.clone();
        this->pred_info_x_d_0_info["exp"] = x_d_info["exp"].clone();
        this->pred_info_x_d_0_info["t"] = x_d_info["t"].clone();
    }

    Mat R_new = (R_d * this->pred_info_x_d_0_info["R_d"].t()) * R_s;
    Mat delta_new = x_s_info["exp"] + (x_d_info["exp"] - this->pred_info_x_d_0_info["exp"]);
    Mat scale_new = x_s_info["scale"].mul(x_d_info["scale"] / this->pred_info_x_d_0_info["scale"]); /// scale是1x1矩阵,也就是单个数值
    Mat t_new = x_s_info["t"] + (x_d_info["t"] - this->pred_info_x_d_0_info["t"]);
    
    t_new.at<float>(0, 2) = 0;
    Mat temp = repeat(t_new, 21, 1);
    Mat x_d_new = scale_new.at<float>(0, 0) * (x_s_info["kp"] * R_new + delta_new) + temp;
    
    this->stitching(x_s, x_d_new);
    
    Mat out = this->warping_spade(f_s, x_s, x_d_new);   ////形状是[1,3,512,512]
    const int image_erea = out.size[2] * out.size[3];
    float *pdata = (float *)out.data;
    Mat rmat = Mat(out.size[2], out.size[3], CV_32FC1, pdata);
    Mat gmat = Mat(out.size[2], out.size[3], CV_32FC1, pdata + image_erea);
    Mat bmat = Mat(out.size[2], out.size[3], CV_32FC1, pdata + image_erea * 2);
    rmat.setTo(0, rmat < 0);
    rmat.setTo(1, rmat > 1);
    gmat.setTo(0, gmat < 0);
    gmat.setTo(1, gmat > 1);
    bmat.setTo(0, bmat < 0);
    bmat.setTo(1, bmat > 1);
    vector<Mat> channel_mats(3);
    channel_mats[0] = rmat;
    channel_mats[1] = gmat;
    channel_mats[2] = bmat;
    Mat I_p;
    merge(channel_mats, I_p);
    I_p *= 255;
    I_p.convertTo(I_p, CV_8UC3);
    return I_p;
}

int LivePortraitPipeline::execute(string imgpath, string videopath)
{
    Mat srcimg = imread(imgpath);
    if (srcimg.empty())
    {
        cout << "opencv读取图片为空, 请检查输入图片的路径" << endl;
        return -1;
    }

    Mat img;
    cvtColor(srcimg, img, COLOR_BGRA2BGR);
    cvtColor(img, img, COLOR_BGR2RGB);
    Mat src_img = src_preprocess(img);
    std::map<string, Mat> crop_info;
    crop_src_image(src_img, crop_info);

    Mat img_crop_256x256 = crop_info["img_crop_256x256"];
    vector<float> I_s;
    vector<int64_t> I_s_shape;
    preprocess(img_crop_256x256, I_s, I_s_shape);

    std::map<string, Mat> x_s_info;
    this->get_kp_info(I_s, I_s_shape, x_s_info);
    Mat R_s = get_rotation_matrix(x_s_info["pitch"], x_s_info["yaw"], x_s_info["roll"]);  ////返回结果已验证通过

    vector<float> f_s;
    this->extract_feature_3d(I_s, I_s_shape, f_s);
    Mat x_s = transform_keypoint(x_s_info);

    cv::VideoCapture capture(videopath);
    if (!capture.isOpened())
    {
        cout << "VideoCapture,open video file failed, " << videopath << endl;
        return -1;
    }
    const int fps = capture.get(cv::CAP_PROP_FPS);
    const int video_length = capture.get(cv::CAP_PROP_FRAME_COUNT);
    cout<<"video total have "<<video_length<<" frames"<<endl;
    int f_h = src_img.rows;
    int f_w = src_img.cols;
    if (this->flg_composite)
    {
        f_h = 512;
        f_w = 512 * 3;
    }

    ////prepare for pasteback
    Mat mask_ori = prepare_paste_back(this->mask_crop, crop_info["M_c2o"], Size(src_img.cols, src_img.rows));

    VideoWriter video_writer;
    video_writer.open("output.mp4", cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, Size(f_w, f_h));
    Mat frame;
    int frame_id = 0;
    while (capture.read(frame))
    {
        if (frame.empty())
        {
            break;
        }

        Mat img_rgb;
        cvtColor(frame, img_rgb, COLOR_BGR2RGB);
        auto a = std::chrono::high_resolution_clock::now();
        Mat I_p = this->predict(frame_id, x_s_info, R_s, f_s, x_s, img_rgb);
        auto b = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> c = b - a;
        cout<<"frame_id="<<frame_id<<", predict waste time="<<to_string(c.count())<<" s"<<endl;
        
        frame_id += 1;
        Mat driving_img;
        if (this->flg_composite)
        {
            concat_frame(img_rgb, img_crop_256x256, I_p, driving_img);
        }
        else
        {
            paste_back(I_p, crop_info["M_c2o"], src_img, mask_ori, driving_img);
        }
        cvtColor(driving_img, driving_img, COLOR_RGB2BGR);
        video_writer.write(driving_img);
    }
    video_writer.release();
    capture.release();
    ///destroyAllWindows();
    return 0;
}