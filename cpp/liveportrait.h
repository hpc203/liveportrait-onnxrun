#ifndef LIVEPORTRAIT
#define LIVEPORTRAIT
#include <math.h>
// #include <cuda_provider_factory.h>
#include <onnxruntime_cxx_api.h>
#include "faceanalysis.h"

class LivePortraitPipeline
{
public:
	LivePortraitPipeline();
	int execute(std::string imgpath, std::string videopath);

private:
	Ort::Env env = Ort::Env(ORT_LOGGING_LEVEL_ERROR, "LivePortrait");
	Ort::SessionOptions sessionOptions = Ort::SessionOptions();

	Ort::Session *appearance_feature_extractor_ort_session = nullptr;
	const std::vector<const char *> appearance_feature_extractor_input_names = {"img"};
	const std::vector<const char *> appearance_feature_extractor_output_names = {"output"};
	const std::vector<int64_t> appearance_feature_extractor_input_shape = {1, 3, 256, 256};
	const std::vector<int64_t> appearance_feature_extractor_output_shape = {1, 32, 16, 64, 64};

	Ort::Session *motion_extractor_ort_session = nullptr;
	const std::vector<const char *> motion_extractor_input_names = {"img"};
	const std::vector<const char *> motion_extractor_output_names = {"pitch", "yaw", "roll", "t", "exp", "scale", "kp"};
	const std::vector<int64_t> motion_extractor_input_shape = {1, 3, 256, 256};
	const std::vector<std::vector<int64_t>> motion_extractor_output_shape = {{1, 66}, {1, 66}, {1, 66}, {1, 3}, {1, 63}, {1, 1}, {1, 63}};

	Ort::Session *warping_spade_ort_session = nullptr;
	const std::vector<const char *> warping_spade_input_names = {"feature_3d", "kp_driving", "kp_source"};
	const std::vector<const char *> warping_spade_output_names = {"out"};
	const std::vector<std::vector<int64_t>> warping_spade_input_shape = {{1, 32, 16, 64, 64}, {1, 21, 3}, {1, 21, 3}};
	const std::vector<int> warping_spade_output_shape = {1, 3, 512, 512};

	Ort::Session *stitching_module_ort_session = nullptr;
	const std::vector<const char *> stitching_module_input_names = {"input"};
	const std::vector<const char *> stitching_module_output_names = {"output"};
	const std::vector<int64_t> stitching_module_input_shape = {1, 126};
	const std::vector<int64_t> stitching_module_output_shape = {1, 65};

	Ort::Session *landmark_runner_ort_session = nullptr;
	const std::vector<const char *> landmark_runner_input_names = {"input"};
	const std::vector<const char *> landmark_runner_output_names = {"output", "853", "856"};
	const std::vector<int64_t> landmark_runner_input_shape = {1, 3, 224, 224};
	const std::vector<std::vector<int>> landmark_runner_output_shape = {{1, 214}, {1, 262}, {1, 406}};
	std::vector<float> landmark_runner_input_tensor;
	std::shared_ptr<FaceAnalysis> face_analysis{nullptr};

	void crop_src_image(const cv::Mat &img, std::map<std::string, cv::Mat> &crop_info);
	cv::Mat landmark_runner(const cv::Mat &img, const float *lmk);
	void get_kp_info(std::vector<float> x, std::vector<int64_t> shape, std::map<std::string, cv::Mat> &kp_info);
	void extract_feature_3d(std::vector<float> x, std::vector<int64_t> shape, std::vector<float> &f_s);
	void stitching(const cv::Mat &kp_source, cv::Mat &kp_driving_new);
	cv::Mat warping_spade(std::vector<float> feature_3d, const cv::Mat &kp_source, const cv::Mat &kp_driving);
	cv::Mat predict(const int frame_id, std::map<std::string, cv::Mat> x_s_info, const cv::Mat &R_s, std::vector<float> f_s, const cv::Mat &x_s, const cv::Mat &frame);

	Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);
	Ort::RunOptions runOptions;

	cv::Mat mask_crop;
	const bool flg_composite = false;
	cv::Mat pred_info_lmk;
	std::map<std::string, cv::Mat> pred_info_x_d_0_info;
};

#endif