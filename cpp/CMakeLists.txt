project(test)
set(CMAKE_CXX_STANDARD 17)

add_executable(${PROJECT_NAME}  ${CMAKE_CURRENT_SOURCE_DIR}/main.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/liveportrait.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/faceanalysis.cpp
                                ${CMAKE_CURRENT_SOURCE_DIR}/utils_crop.cpp)

target_include_directories(${PROJECT_NAME}
        PUBLIC "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/include/opencv4"
        PUBLIC "/home/wangbo/libs/onnxruntime-linux-x64-1.18.1/include")

target_link_libraries(${PROJECT_NAME}
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_imgcodecs.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_highgui.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_imgproc.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_core.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_videoio.so.4.8.0"
        "/home/wangbo/libs/mylibs/opencv-4.8.0/temp_install_dir/lib/libopencv_video.so.4.8.0"
        "/home/wangbo/libs/onnxruntime-linux-x64-1.18.1/lib/libonnxruntime.so.1.18.1")