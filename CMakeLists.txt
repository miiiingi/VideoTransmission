cmake_minimum_required(VERSION 3.5)

# 프로젝트 이름 설정
project(VideoTransmission C)

# C 표준 설정
set(CMAKE_C_STANDARD 11)

# FFmpeg 라이브러리 찾기
find_package(PkgConfig REQUIRED)
pkg_check_modules(AVUTIL REQUIRED libavutil)
pkg_check_modules(AVFORMAT REQUIRED libavformat)
pkg_check_modules(AVCODEC REQUIRED libavcodec)
pkg_check_modules(SWSCALE REQUIRED libswscale)

# 소스 파일 설정 - 여러 개의 소스 파일을 추가
set(SERVER_SOURCE_FILES v4l2_cap_tcp_server_encode.c)
set(CLIENT_SOURCE_FILES v4l2_cap_tcp_client_encode.c)
set(CAPTURE_FILES v4l2_cap.c)

# 헤더 및 라이브러리 포함 디렉토리 추가
include_directories(${AVUTIL_INCLUDE_DIRS} ${AVFORMAT_INCLUDE_DIRS} ${AVCODEC_INCLUDE_DIRS} ${SWSCALE_INCLUDE_DIRS})

# 서버 실행 파일 생성 및 라이브러리 링크
add_executable(VideoServer ${SERVER_SOURCE_FILES})
target_link_libraries(VideoServer ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)

# 클라이언트 실행 파일 생성 및 라이브러리 링크
add_executable(VideoClient ${CLIENT_SOURCE_FILES})
target_link_libraries(VideoClient ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)

# 클라이언트 실행 파일 생성 및 라이브러리 링크
add_executable(Capture ${CAPTURE_FILES})
target_link_libraries(Capture ${AVUTIL_LIBRARIES} ${AVFORMAT_LIBRARIES} ${AVCODEC_LIBRARIES} ${SWSCALE_LIBRARIES} pthread)
