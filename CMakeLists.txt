cmake_minimum_required(VERSION 3.8)  # CUDA 지원을 위해 3.8 이상 필요
project(VideoTransmission CUDA C CXX)    # CUDA 언어 지원 추가

# C 표준 설정
set(CMAKE_C_STANDARD 11)

# CUDA 컴파일러 설정
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)  # nvcc 경로 지정
set(CMAKE_CUDA_ARCHITECTURES 87)

find_package(CUDA REQUIRED)
find_package(OpenCV REQUIRED)
enable_language(CUDA)


# FFmpeg 라이브러리 찾기

find_package(PkgConfig REQUIRED)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(SWSCALE REQUIRED libswscale)

# CUDA 라이브러리 경로 추가
link_directories(
    /usr/local/cuda/lib64
)


# 소스 파일 설정
#set(SERVER_SOURCE_FILES_BASIC v4l2_cap_tcp_server.c)
#set(CLIENT_SOURCE_FILES_BASIC v4l2_cap_tcp_client.c)

# 헤더 및 라이브러리 포함 디렉토리 추가
include_directories(
    /usr/local/cuda/include  # CUDA 헤더 경로
    ${AVUTIL_INCLUDE_DIRS} 
    ${AVFORMAT_INCLUDE_DIRS} 
    ${AVCODEC_INCLUDE_DIRS} 
    ${SWSCALE_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}  # OpenCV 헤더 경로 추가
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/DS_Timer
)

# CUDA 소스 파일과 C++ 소스 파일 분리
set(CUDA_SOURCES v4l2_cap.cu)  # CUDA 소스 파일
set(CPP_SOURCES DS_Timer/DS_timer.cpp)  # C++ 소스 파일

# DS_timer.cpp를 C++ 파일로 컴파일
set_source_files_properties(${CPP_SOURCES} PROPERTIES LANGUAGE CXX)

# Capture 실행 파일에 대한 CUDA 설정
add_executable(Capture ${CUDA_SOURCES} ${CPP_SOURCES})
set_target_properties(Capture PROPERTIES CUDA_SEPARABLE_COMPILATION ON)    # 분리 컴파일 활성화

# 라이브러리 링크
target_link_libraries(Capture 
    ${CUDA_LIBRARIES} 
    ${AVUTIL_LIBRARIES} 
    ${AVFORMAT_LIBRARIES} 
    ${AVCODEC_LIBRARIES} 
    ${SWSCALE_LIBRARIES} 
    ${OpenCV_LIBRARIES} 
    cudart  # CUDA 런타임 라이브러리
    pthread
)

set(SERVER_SOURCE_FILES v4l2_cap_tcp_server_encode.cu)
# 나머지 실행 파일들은 그대로 유지
add_executable(Server ${SERVER_SOURCE_FILES})
target_link_libraries(Server
    ${CUDA_LIBRARIES} 
    ${AVUTIL_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    cudart
    pthread
)
set_target_properties(Server PROPERTIES CUDA_SEPARABLE_COMPILATION ON)    # 분리 컴파일 활성화


set(CLIENT_SOURCE_FILES v4l2_cap_tcp_client_encode.cu DS_Timer/DS_timer.cpp)
add_executable(Client ${CLIENT_SOURCE_FILES})
target_link_libraries(Client
    ${CUDA_LIBRARIES} 
    ${AVUTIL_LIBRARIES}
    ${AVFORMAT_LIBRARIES}
    ${AVCODEC_LIBRARIES}
    ${SWSCALE_LIBRARIES}
    ${OpenCV_LIBRARIES} 
    cudart
    pthread
)

set_target_properties(Client PROPERTIES CUDA_SEPARABLE_COMPILATION ON)    # 분리 컴파일 활성화

#add_executable(ClientBasic ${CLIENT_SOURCE_FILES_BASIC})
#target_link_libraries(ClientBasic ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)
#set_source_files_properties(${CLIENT_SOURCE_FILES_BASIC} PROPERTIES LANGUAGE C)

#add_executable(ServerBasic ${SERVER_SOURCE_FILES_BASIC})
#target_link_libraries(ServerBasic ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)
#set_source_files_properties(${SERVER_SOURCE_FILES_BASIC} PROPERTIES LANGUAGE C)