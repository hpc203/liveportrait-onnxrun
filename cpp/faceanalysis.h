# ifndef FACE_ANALYSIS
# define FACE_ANALYSIS
//#include <cuda_provider_factory.h>  ///如果使用cuda加速，需要取消注释
#include <onnxruntime_cxx_api.h>
#include "utils_crop.h"


class FaceAnalysis
{
public:
	FaceAnalysis(std::string modelpatha, std::string modelpathb);
	std::vector<Bbox> detect(const cv::Mat& srcimg);   
private:
	void preprocess(cv::Mat img);
	std::vector<float> input_image;
	float det_scale;
	void generate_proposal(const float* p_box, const float* p_scores, const float* p_kps, const int stride, std::vector<Bbox>& boxes);

	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "Face Analysis");
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();

	Ort::Session *det_face_ort_session = nullptr;
	const std::vector<const char*> det_face_input_names = {"input.1"};
	const std::vector<const char*> det_face_output_names = {"448", "471", "494", "451", "474", "497", "454", "477", "500"};
	const int input_size = 512;     ///单个输入
	std::vector<std::vector<int64_t>> det_face_output_node_dims;
	const float det_thresh = 0.5;
    const int fmc = 3;
    const int feat_stride_fpn[3] = {8, 16, 32};
	const float nms_thresh = 0.4;

	Ort::Session *landmark_ort_session = nullptr;
	const std::vector<const char*> landmark_input_names = {"data"};
	const std::vector<const char*> landmark_output_names = {"fc1"};
	const int landmark_input_size = 192;     ///单个输入
	std::vector<float> aimg;
	int landmark_input_height;
	int landmark_input_width;                ///单个输入
	std::vector<std::vector<int64_t>> landmark_output_node_dims;
	std::vector<int64_t> landmark_input_tensor_shape;

	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::RunOptions runOptions;
};

#endif