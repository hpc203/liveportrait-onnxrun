#include "faceanalysis.h"


using namespace cv;
using namespace std;
using namespace Ort;


FaceAnalysis::FaceAnalysis(string model_patha, string model_pathb)
{
    /// OrtStatus* status = OrtSessionOptionsAppendExecutionProvider_CUDA(sessionOptions, 0);   ///如果使用cuda加速，需要取消注释
    sessionOptions.SetGraphOptimizationLevel(ORT_ENABLE_BASIC);
    /// std::wstring widestr = std::wstring(model_patha.begin(), model_patha.end());  ////windows写法
    /// ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
    det_face_ort_session = new Session(env, model_patha.c_str(), sessionOptions); ////linux写法

    size_t numInputNodes = det_face_ort_session->GetInputCount();
    size_t numOutputNodes = det_face_ort_session->GetOutputCount();
    AllocatorWithDefaultOptions allocator;
    
    for (int i = 0; i < numOutputNodes; i++)
    {
        Ort::TypeInfo output_type_info = det_face_ort_session->GetOutputTypeInfo(i);
        auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
        auto output_dims = output_tensor_info.GetShape();
        det_face_output_node_dims.push_back(output_dims);
    }

    /// std::wstring widestr = std::wstring(model_pathb.begin(), model_pathb.end());  ////windows写法
    /// ort_session = new Session(env, widestr.c_str(), sessionOptions); ////windows写法
    landmark_ort_session = new Session(env, model_pathb.c_str(), sessionOptions); ////linux写法
    numInputNodes = landmark_ort_session->GetInputCount();
    numOutputNodes = landmark_ort_session->GetOutputCount();
    
    Ort::TypeInfo input_type_info = landmark_ort_session->GetInputTypeInfo(0);
	auto input_tensor_info = input_type_info.GetTensorTypeAndShapeInfo();
	auto input_dims = input_tensor_info.GetShape();
	this->landmark_input_height = input_dims[2];
    this->landmark_input_width = input_dims[3];
	this->landmark_input_tensor_shape = { 1, 3, this->landmark_input_height, this->landmark_input_width };
    
    Ort::TypeInfo output_type_info = landmark_ort_session->GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    auto output_dims = output_tensor_info.GetShape();
    landmark_output_node_dims.push_back(output_dims);  
}

void FaceAnalysis::preprocess(Mat srcimg)
{
    const float im_ratio = float(srcimg.rows) / (float)srcimg.cols;
    int new_width = this->input_size;
    int new_height = int(new_width * im_ratio);
    if(im_ratio>1)
    {
        new_height = this->input_size;
        new_width = int(new_height / im_ratio);
    }
    this->det_scale = float(new_height) / (float)srcimg.rows;
    Mat resized_img;
    resize(srcimg, resized_img, Size(new_width, new_height));
    Mat det_img;
    copyMakeBorder(resized_img, det_img, 0, this->input_size - new_height, 0, this->input_size - new_width, BORDER_CONSTANT, 0);

    vector<cv::Mat> bgrChannels(3);
    split(det_img, bgrChannels);
    for (int c = 0; c < 3; c++)
    {
        bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1, 1 / 128.0, -127.5 / 128.0);
    }

    const int image_area = this->input_size * this->input_size;
    this->input_image.resize(3 * image_area);
    size_t single_chn_size = image_area * sizeof(float);
    memcpy(this->input_image.data(), (float *)bgrChannels[0].data, single_chn_size);
    memcpy(this->input_image.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
    memcpy(this->input_image.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);
}

void FaceAnalysis::generate_proposal(const float* p_box, const float* p_scores, const float* p_kps, const int stride, vector<Bbox>& boxes)
{
	const int feat_h = this->input_size / stride;
	const int feat_w = this->input_size / stride;
    const int num_anchors = 2;
	for (int i = 0; i < feat_h; i++)
	{
		for (int j = 0; j < feat_w; j++)
		{
			for(int n=0; n<num_anchors; n++)
            {
                const int index = i * feat_w*num_anchors + j*num_anchors+n;
                if(p_scores[index] >= this->det_thresh)
                {
                    Bbox box;
                    box.xmin = (j - p_box[index * 4]) * stride;
                    box.ymin = (i - p_box[index * 4 + 1]) * stride;
                    box.xmax = (j + p_box[index * 4 + 2]) * stride;
                    box.ymax = (i + p_box[index * 4 + 3]) * stride;
                    box.xmin /= this->det_scale;
                    box.ymin /= this->det_scale;
                    box.xmax /= this->det_scale;
                    box.ymax /= this->det_scale;

                    for(int k=0;k<5;k++)
                    {
                        float px = (j + p_kps[index * 10 + k * 2]) * stride;
                        float py = (i + p_kps[index * 10 + k * 2 + 1]) * stride;
                        px /= this->det_scale;
                        py /= this->det_scale;
                        box.kps[k * 2] = px;
                        box.kps[k * 2 + 1] = py;
                    }
                    box.score = p_scores[index];
                    boxes.emplace_back(box);
                }

            }
		}
	}
}

bool cmp(Bbox a, Bbox b)
{
    float area_a = (a.xmax - a.xmin) * (a.ymax - a.ymin);
    float area_b = (b.xmax - b.xmin) * (b.ymax - b.ymin);
    return area_a > area_b;
}
vector<Bbox> FaceAnalysis::detect(const Mat& srcimg)
{
    this->preprocess(srcimg);

    std::vector<int64_t> input_img_shape = {1, 3, this->input_size, this->input_size};  ///也可以写在构造函数里, det_face的输入是动态的
    Value input_tensor_ = Value::CreateTensor<float>(memory_info_handler, this->input_image.data(), this->input_image.size(), input_img_shape.data(), input_img_shape.size());

    vector<Value> det_face_ort_outputs = this->det_face_ort_session->Run(runOptions, this->det_face_input_names.data(), &input_tensor_, 1, this->det_face_output_names.data(), this->det_face_output_names.size());
    vector<Bbox> boxes;
    for(int i=0;i<3;i++)
    {
        float *p_scores = det_face_ort_outputs[i].GetTensorMutableData<float>();
        float *p_bbox = det_face_ort_outputs[i + this->fmc].GetTensorMutableData<float>();
        float *p_kps = det_face_ort_outputs[i + this->fmc*2].GetTensorMutableData<float>();
        
        this->generate_proposal(p_bbox, p_scores, p_kps, this->feat_stride_fpn[i], boxes);
    }
    nms_boxes(boxes, this->nms_thresh);

    for(int i=0;i<boxes.size();i++)
    {
        ////get_landmark
        float w = boxes[i].xmax - boxes[i].xmin;
        float h = boxes[i].ymax - boxes[i].ymin;
        float center[2] = {(boxes[i].xmin + boxes[i].xmax) * 0.5f, (boxes[i].ymin + boxes[i].ymax) * 0.5f};
        float rot = 0.f*PI/180.f;
        float scale_ratio = this->landmark_input_size / (max(w, h) * 1.5);
        ////face_align
        Mat M = (Mat_<float>(2, 3) << scale_ratio*cos(rot), -scale_ratio*sin(rot), this->landmark_input_size*0.5-center[0]*scale_ratio, scale_ratio*sin(rot), scale_ratio*cos(rot), this->landmark_input_size*0.5-center[1]*scale_ratio);
        Mat cropped;
	    warpAffine(srcimg, cropped, M, cv::Size(this->landmark_input_size, this->landmark_input_size));
        ////face_align
        vector<cv::Mat> bgrChannels(3);
        split(cropped, bgrChannels);
        for (int c = 0; c < 3; c++)
        {
            bgrChannels[c].convertTo(bgrChannels[c], CV_32FC1);
        }

        const int image_area = this->landmark_input_size * this->landmark_input_size;
        this->aimg.resize(3 * image_area);
        size_t single_chn_size = image_area * sizeof(float);
        memcpy(this->aimg.data(), (float *)bgrChannels[0].data, single_chn_size);
        memcpy(this->aimg.data() + image_area, (float *)bgrChannels[1].data, single_chn_size);
        memcpy(this->aimg.data() + image_area * 2, (float *)bgrChannels[2].data, single_chn_size);

        Value input_tensor2 = Value::CreateTensor<float>(memory_info_handler, this->aimg.data(), this->aimg.size(), this->landmark_input_tensor_shape.data(), this->landmark_input_tensor_shape.size());
        
        vector<Value> landmark_ort_outputs = this->landmark_ort_session->Run(runOptions, this->landmark_input_names.data(), &input_tensor2, 1, this->landmark_output_names.data(), this->landmark_output_names.size());
        float *p_landmark = landmark_ort_outputs[0].GetTensorMutableData<float>();
        Mat IM;
	    invertAffineTransform(M, IM);
        for(int k=0;k<106;k++)
        {
            float px = (p_landmark[k * 2] + 1) * this->landmark_input_size*0.5;
            float py = (p_landmark[k * 2 + 1] + 1) * this->landmark_input_size*0.5;
            boxes[i].landmark_2d_106[k * 2] = IM.at<float>(0, 0) * px + IM.at<float>(0, 1) * py + IM.at<float>(0, 2);
            boxes[i].landmark_2d_106[k * 2 + 1] = IM.at<float>(1, 0) * px + IM.at<float>(1, 1) * py + IM.at<float>(1, 2);
        }
        ////get_landmark
    }
    sort(boxes.begin(), boxes.end(), cmp);
    return boxes;
}