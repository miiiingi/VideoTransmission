cmake_minimum_required(VERSION 3.8)  # CUDA 지원을 위해 3.8 이상 필요
project(VideoTransmission CUDA C)    # CUDA 언어 지원 추가

# C 표준 설정
set(CMAKE_C_STANDARD 11)

# CUDA 컴파일러 설정
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)  # nvcc 경로 지정

find_package(CUDA REQUIRED)
enable_language(CUDA)

# FFmpeg 라이브러리 찾기

find_package(PkgConfig REQUIRED)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(SWSCALE REQUIRED libswscale)

# 소스 파일 설정
set(SERVER_SOURCE_FILES v4l2_cap_tcp_server_encode.c)
set(CLIENT_SOURCE_FILES v4l2_cap_tcp_client_encode.c)
set(SERVER_SOURCE_FILES_BASIC v4l2_cap_tcp_server.c)
set(CLIENT_SOURCE_FILES_BASIC v4l2_cap_tcp_client.c)
set(CAPTURE_FILES v4l2_cap.cu)  # .cu 파일 추가

# 헤더 및 라이브러리 포함 디렉토리 추가
include_directories(
    /usr/local/cuda/include  # CUDA 헤더 경로
    ${AVUTIL_INCLUDE_DIRS} 
    ${AVFORMAT_INCLUDE_DIRS} 
    ${AVCODEC_INCLUDE_DIRS} 
    ${SWSCALE_INCLUDE_DIRS}
)

# CUDA 라이브러리 경로 추가
link_directories(/usr/local/cuda/lib64)

# Capture 실행 파일에 대한 CUDA 설정
add_executable(Capture ${CAPTURE_FILES})
set_target_properties(Capture PROPERTIES CUDA_SEPARABLE_COMPILATION ON)    # 분리 컴파일 활성화

# 라이브러리 링크
target_link_libraries(Capture 
    ${CUDA_LIBRARIES} 
    ${AVUTIL_LIBRARIES} 
    ${AVFORMAT_LIBRARIES} 
    ${AVCODEC_LIBRARIES} 
    ${SWSCALE_LIBRARIES} 
    cudart  # CUDA 런타임 라이브러리
    pthread
)

# 나머지 실행 파일들은 그대로 유지
add_executable(VideoServer ${SERVER_SOURCE_FILES})
target_link_libraries(VideoServer ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)

add_executable(VideoServerBasic ${SERVER_SOURCE_FILES_BASIC})
target_link_libraries(VideoServerBasic ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)

add_executable(VideoClient ${CLIENT_SOURCE_FILES})
target_link_libraries(VideoClient ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)

add_executable(VideoClientBasic ${CLIENT_SOURCE_FILES_BASIC})
target_link_libraries(VideoClientBasic ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)