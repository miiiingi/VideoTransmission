cmake_minimum_required(VERSION 3.8)  # CUDA 지원을 위해 3.8 이상 필요
project(VideoTransmission CUDA C CXX)    # CUDA 언어 지원 추가

# C 표준 설정
set(CMAKE_C_STANDARD 17)

# CUDA 컴파일러 설정
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)  # nvcc 경로 지정
set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
enable_language(CUDA)

# CUDA 라이브러리 경로 추가
link_directories(/usr/local/cuda/lib64)

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

# 헤더 및 라이브러리 포함 디렉토리 추가
include_directories(
    /usr/local/cuda/include  # CUDA 헤더 경로
    /usr/lib/gcc/aarch64-linux-gnu/11/include
    ${AVUTIL_INCLUDE_DIRS} 
    ${AVFORMAT_INCLUDE_DIRS} 
    ${AVCODEC_INCLUDE_DIRS} 
    ${SWSCALE_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}  # OpenCV 헤더 경로 추가
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/DS_Timer
    ${PROJECT_SOURCE_DIR}/NEON
)

# CUDA 소스 파일과 C++ 소스 파일 분리
set(CUDA_SOURCES v4l2_cap.cu)  # CUDA 소스 파일
set(CPP_SOURCES DS_Timer/DS_timer.cpp NEON/neon.cpp)  # C++ 소스 파일

# C++ 파일로 컴파일
# set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE CXX)

# Capture 실행 파일에 대한 CUDA 설정
# add_executable(Capture ${CUDA_SOURCES} ${CPP_SOURCES})

# CUDA 특정 속성 설정
# set_target_properties(Capture PROPERTIES 
#     CUDA_SEPARABLE_COMPILATION ON
# )

# 라이브러리 링크
# target_link_libraries(Capture 
#     ${CUDA_LIBRARIES} 
#     ${AVUTIL_LIBRARIES} 
#     ${AVFORMAT_LIBRARIES} 
#     ${AVCODEC_LIBRARIES} 
#     ${SWSCALE_LIBRARIES} 
#     ${OpenCV_LIBRARIES} 
#     cudart  # CUDA 런타임 라이브러리
#     pthread
# )

# CUDA와 C++ 파일에 대해 별도의 타겟 생성
add_executable(CaptureCUDA ${CUDA_SOURCES})
add_executable(CaptureCPP ${CPP_SOURCES})

# CUDA 타겟에 대한 라이브러리 링크
target_link_libraries(CaptureCUDA 
    ${CUDA_LIBRARIES} 
    ${AVUTIL_LIBRARIES} 
    ${AVFORMAT_LIBRARIES} 
    ${AVCODEC_LIBRARIES} 
    ${SWSCALE_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    cudart  # CUDA 런타임 라이브러리
    pthread
)

# C++ 타겟에 대한 라이브러리 링크
target_link_libraries(CaptureCPP 
    ${AVUTIL_LIBRARIES} 
    ${AVFORMAT_LIBRARIES} 
    ${AVCODEC_LIBRARIES} 
    ${SWSCALE_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    pthread
)

# 필요한 경우 두 타겟을 하나의 실행 파일로 합치기
add_dependencies(Capture CaptureCUDA CaptureCPP)



# 나머지 실행 파일들은 그대로 유지
# add_executable(VideoServer ${SERVER_SOURCE_FILES})
# target_link_libraries(VideoServer ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)
# set_source_files_properties(${SERVER_SOURCE_FILES} PROPERTIES LANGUAGE C)

# add_executable(VideoServerBasic ${SERVER_SOURCE_FILES_BASIC})
# target_link_libraries(VideoServerBasic ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)
# set_source_files_properties(${SERVER_SOURCE_FILES_BASIC} PROPERTIES LANGUAGE C)

# add_executable(VideoClient ${CLIENT_SOURCE_FILES})
# target_link_libraries(VideoClient ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)
# set_source_files_properties(${CLIENT_SOURCE_FILES} PROPERTIES LANGUAGE C)

# add_executable(VideoClientBasic ${CLIENT_SOURCE_FILES_BASIC})
# target_link_libraries(VideoClientBasic ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)
# set_source_files_properties(${CLIENT_SOURCE_FILES_BASIC} PROPERTIES LANGUAGE C)