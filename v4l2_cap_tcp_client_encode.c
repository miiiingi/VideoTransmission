#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <linux/fb.h>
#include <arpa/inet.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <linux/videodev2.h>

#define WIDTH 		    640
#define HEIGHT 		    480

#define VIDEO_DEV     "/dev/video0"
#define FB_DEV        "/dev/fb0"
#define SERVER_PORT 8080
#define SERVER_IP "127.0.0.1"

typedef unsigned char uchar;
static int cond = 1;
/*
 * framebuffer에 비디오 프레임을 그리기 위해 포인터에 대한 변수 선언
*/
unsigned short *rgbBuffer, *fbPtr;
/*
 * 프레임 버퍼에 대한 다양한 화면 속성 정보를 담고있다.
 * xres, yres, xres_virtual, yres_virtual
 * bits_per_pixel: 한 픽셀을 표현하는 데 사용되는 비트수(e.g. 16, 24, 32 비트)
*/
static struct fb_var_screeninfo vinfo;
/*
 * 카메라로부터 온 비디오 프레임 YUYV 포맷을 프레임 버퍼에서 출력하기 위해 RGB 16bit로 변환하는 함수
 * */
static void yuyv2Rgb565(uchar *yuyv, unsigned short *fbPtr, int width, int height);
/*
 * 프레임 버퍼를 초기화하는 함수
 * */
static int init_framebuffer(unsigned short **fbPtr, int *size);
/*
 * 디버깅을 위해 프레임 버퍼에 출력된 데이터를 bmp 파일로 저장
 * */
static int decode_and_save_frame(AVCodecContext *dec_ctx, AVPacket *pkt, AVFrame *frame, uint8_t **output_data, int frame_num);
static void save_framebuffer_as_bmp(const char *filename, unsigned short *fbPtr, int width, int height);

static void save_yuv420p_as_bmp(const char *filename, AVFrame *frame, int width, int height);
static inline int clip(int value, int min, int max);
static void sigHandler(int signo);
static void save_yuyv_as_bmp(const char *filename, unsigned char *yuyv, int width, int height);
static void yuv420p_to_rgb565(unsigned char *yuv420p_data[3], unsigned short *rgb565_data, int width, int height, int fb_width);
static void draw_char_on_framebuffer(char c, int x, int y, unsigned short *fbPtr, int fb_width, int fb_height);
static void draw_text_on_framebuffer(const char* text, int x, int y, unsigned short *fbPtr, int fb_width, int fb_height);
int main(int argc, char** argv) 
{
    signal(SIGINT, sigHandler);
    int cam_fd, fb_fd;
    int fbSize, BUFFER_SIZE = WIDTH * HEIGHT * 2; // YUYV는 픽셀당 2바이트 
    unsigned char* frame_buffer = malloc(BUFFER_SIZE);

    int sock;
    struct sockaddr_in server_addr;
    ssize_t bytes_received = 0;

    // 소켓 생성
    sock = socket(AF_INET, SOCK_STREAM, 0);
    if (sock < 0) {
        perror("socket() failed");
	close(sock);
	return -1;
    }

    // 서버 주소 설정
    memset(&server_addr, 0, sizeof(server_addr));
    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    server_addr.sin_port = htons(SERVER_PORT);

    // 서버에 연결
    if (connect(sock, (struct sockaddr *)&server_addr, sizeof(server_addr)) < 0) {
        perror("connect() failed");
	close(sock);
	return -1;
    }

    printf("Connected to server...\n");

    // FFmpeg 디코더 초기화
    const AVCodec *codec = avcodec_find_decoder(AV_CODEC_ID_H264);
    if (!codec) {
        fprintf(stderr, "Codec not found\n");
        return -1;
    }

    AVCodecContext *dec_ctx = avcodec_alloc_context3(codec);
    if (!dec_ctx) {
        fprintf(stderr, "Could not allocate video codec context\n");
        return -1;
    }
    dec_ctx->time_base = (AVRational){1, 120};
    dec_ctx->framerate = (AVRational){120, 1};

    if (avcodec_open2(dec_ctx, codec, NULL) < 0) {
        fprintf(stderr, "Could not open codec\n");
        return -1;
    }

    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
        fprintf(stderr, "Could not allocate packet\n");
        return -1;
    }

    AVFrame *frame = av_frame_alloc();
    if (!frame) {
        fprintf(stderr, "Could not allocate frame\n");
        return -1;
    }

    // 프레임버퍼 초기화
    fb_fd = init_framebuffer(&fbPtr, &fbSize);
    if (fb_fd < 0) {
        fprintf(stderr, "Failed to initialize framebuffer\n"); 
        return -1;
    }

    // 영상을 저장할 메모리 할당
    // framebuffer는 rgb 16 비트라서 unsigned short로 변환
    rgbBuffer = (unsigned short *)malloc(fbSize);
    if (!rgbBuffer) {
        perror("Failed to allocate buffers");
        close(fb_fd);
        return -1;
    }

    int frame_num = 0;

    // .h264 파일로 저장할 파일 포인터 열기
    FILE *h264_file = fopen("client_received.h264", "wb");
    if (!h264_file) {
        perror("Failed to open output .h264 file");
        close(sock);
        return -1;
    }

    uint8_t *packet_data = malloc(BUFFER_SIZE);
    int decode_fn_check = -2;
    // Y, U, V 평면 복사를 위한 메모리 할당
    uint8_t *yuv420p_data[3];
    yuv420p_data[0] = malloc(WIDTH * HEIGHT);
    yuv420p_data[1] = malloc((WIDTH * HEIGHT) / 4);
    yuv420p_data[2] = malloc((WIDTH * HEIGHT) / 4);
    int fb_width = vinfo.xres;
    int fb_height = vinfo.yres;

    while (cond) {
        // recv()로 패킷을 수신
        bytes_received = recv(sock, packet_data, WIDTH * HEIGHT * 2, 0);
        if (bytes_received < 0) {
            perror("recv failed");
            break;
        } else if (bytes_received == 0) {
            printf("Server disconnected\n");
            break;
        }

	/*
	char text_buffer[50];
	snprintf(text_buffer, sizeof(text_buffer), "\rReceived %zd bytes", bytes_received);
	// 프레임 버퍼 하단에 텍스트 출력 (예: 화면의 마지막 16줄 아래쪽에 출력)
	draw_text_on_framebuffer(text_buffer, 10, fb_height - 16, fbPtr, fb_width, fb_height);
	*/
	printf("\rReceived %zd bytes", bytes_received);
	fflush(stdout);

        // NAL 유닛인지 확인하고 기록
        if (packet_data[0] == 0x00 && packet_data[1] == 0x00 && packet_data[2] == 0x00 && packet_data[3] == 0x01) {
            fwrite(packet_data, 1, bytes_received, h264_file);
	    pkt->size = bytes_received;
	    pkt->data = packet_data;
        }

	// H.264 디코딩 후 YUV420P 프레임을 BMP로 저장
       	decode_fn_check = decode_and_save_frame(dec_ctx, pkt, frame, yuv420p_data, frame_num++);
        fflush(h264_file);  // 데이터를 즉시 파일로 기록
	// RGB565 데이터를 프레임버퍼로 출력
	yuv420p_to_rgb565(yuv420p_data, rgbBuffer, WIDTH, HEIGHT, fb_width);

	// 프레임버퍼에 RGB 데이터를 복사
	memcpy(fbPtr, rgbBuffer, fb_height * fb_width * 2);

	/*
	 * 아래의 방법을 통해서 해결했는데 왜 그럴까 ?? 이 문제 해결한 방법 찾아보기
	 */
	//unsigned char *yuyv_copy = (unsigned char *)malloc(WIDTH * HEIGHT * 2);
	//memcpy(yuyv_copy, yuyv_data, WIDTH * HEIGHT * 2);
	// YUV420P 데이터가 제대로 들어있는지 확인
	// YUV420P → YUYV 변환

	// YUYV 데이터를 BMP 파일로 저장
	// save_yuyv_as_bmp("yuyv_output.bmp", yuyv_copy, WIDTH, HEIGHT);
	// YUYV → RGB565 변환 후 프레임 버퍼로 출력
	// yuyv2Rgb565(yuyv_copy, fbPtr, WIDTH, HEIGHT);

	// free(yuyv_copy);  // 변환된 YUYV 데이터 해제

    }

    printf("\nGood Bye!!!\n");
    // 파일디스크립터 정리
    munmap(fbPtr, fbSize);
    free(rgbBuffer);
    free(packet_data);
    free(yuv420p_data[0]);
    free(yuv420p_data[1]);
    free(yuv420p_data[2]);
    av_packet_free(&pkt);
    av_frame_free(&frame);
    avcodec_free_context(&dec_ctx);
    close(cam_fd);
    close(fb_fd);

    return 0;
}
// YUYV 데이터를 RGB565로 변환하는 함수
void yuyv2Rgb565(uchar *yuyv, unsigned short *fbPtr, int width, int height) 
{
    uchar* in = (uchar*)yuyv;
    unsigned short pixel;
    int istride = width*2;     /* 이미지의 폭을 넘어가면 다음 라인으로 내려가도록 설정 */
    int x, y, j;
    int y0, u, y1, v, r, g, b;
    long loc = 0;

    for (y = 0; y < height; ++y) {
	// j는 한 라인의 바이트 , x는 한 라인의 픽셀 수 
        for (j = 0, x = 0; j < vinfo.xres * 2; j += 4, x += 2) {
            if (j >= width*2) {                 /* 현재의 화면에서 이미지를 넘어서는 빈 공간을 처리 */
                 loc++; loc++;
                 continue;
            }
            /* YUYV 성분을 분리 */
            y0 = in[j];
            u = in[j + 1] - 128;
            y1 = in[j + 2];
            v = in[j + 3] - 128;

            /* YUV를 RGB로 전환: Y0 + U + V */
            r = clip((298 * y0 + 409 * v + 128) >> 8, 0, 255);
            g = clip((298 * y0 - 100 * u - 208 * v + 128) >> 8, 0, 255);
            b = clip((298 * y0 + 516 * u + 128) >> 8, 0, 255);
            pixel = ((r>>3)<<11)|((g>>2)<<5)|(b>>3);      /* 16비트 컬러로 전환 */
            fbPtr[loc++] = pixel;

            /* YUV를 RGB로 전환 : Y1 + U + V */
            r = clip((298 * y1 + 409 * v + 128) >> 8, 0, 255);
            g = clip((298 * y1 - 100 * u - 208 * v + 128) >> 8, 0, 255);
            b = clip((298 * y1 + 516 * u + 128) >> 8, 0, 255);
            pixel = ((r>>3)<<11)|((g>>2)<<5)|(b>>3);      /* 16비트 컬러로 전환*/
            fbPtr[loc++] = pixel;
        }
        in += istride;
    }
}

// 프레임버퍼를 설정하는 함수
int init_framebuffer(unsigned short **fbPtr, int *size) 
{
    //frame buffer 장치 열기
    int fd = open(FB_DEV, O_RDWR);
    if (fd < 0) {
        perror("Failed to open framebuffer device");
        return -1;
    }

    // ioctl은 디바이스와 소통하기 위한 시스템 호출로 하드웨어 장치의 설정과 상태를 제어할 때 사용.
    // 파일디스크립터를 통해 장치에 명령을 전달하고 특정 명령에 대한 처리를 요청
    // 아래의 함수는 프레임버퍼 장치에 FBIOGET_VSCREENINFO 명령을 보내 화면 정보를 가져오라는 요청을 하는 것
    // 성공적으로 호출되면 프레임버퍼의 화면 설정 정보가 vinfo라는 fb_var_screeninfo 구조체에 저장됨
    if (ioctl(fd, FBIOGET_VSCREENINFO, &vinfo)) {
        perror("Error reading variable information");
        close(fd);
        return -1;
    }
    /*
    printf("vinfo.yres: %d\n", vinfo.yres);
    printf("vinfo.yres_virtual: %d\n", vinfo.yres_virtual);
    */

    // mmap을 통해 메모리를 매핑하여 물리 메모리와 프레임 버퍼를 연결, 매핑된 메모리 주소가 반환되며, 이주소를 사용해 프레임버퍼에 데이터를 직접 쓸 수 있다.
    *size = vinfo.yres_virtual * vinfo.xres_virtual * vinfo.bits_per_pixel / 8;

    *fbPtr = (unsigned short *)mmap(0, *size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (*fbPtr == MAP_FAILED) {
        perror("Failed to mmap framebuffer");
        close(fd);
        return -1;
    }

    return fd;
}

// H.264 디코딩 후 프레임을 BMP로 저장하는 함수
static int decode_and_save_frame(AVCodecContext *dec_ctx, AVPacket *pkt, AVFrame *frame, uint8_t **output_data, int frame_num) {
    int Y_SIZE = WIDTH * HEIGHT;
    int UV_SIZE = (WIDTH / 2) * (HEIGHT / 2);
    while(cond){
	int ret = avcodec_send_packet(dec_ctx, pkt);
	if (ret < 0) {
	    fprintf(stderr, "Error sending a packet for decoding\n");
	    return -1;
	}
	while(ret >= 0){
	    ret = avcodec_receive_frame(dec_ctx, frame);
	    if (ret == AVERROR(EAGAIN)){
		return AVERROR(EAGAIN);
	    } else if(ret == AVERROR_EOF){ 
		return AVERROR_EOF;
	    } else if (ret < 0) {
		perror("Error during decoding\n");
		return -1;
	    }
	    // PTS가 유효한지 확인 후 조정
	    if (frame->pts != AV_NOPTS_VALUE) {
		frame->pts = av_rescale_q(frame->pts, dec_ctx->time_base, (AVRational){1, 120});
	    }
	    memcpy(output_data[0], frame->data[0], Y_SIZE);
	    memcpy(output_data[1], frame->data[1], UV_SIZE);
	    memcpy(output_data[2], frame->data[2], UV_SIZE);
	    /*
	     * yuv420p 데이터 bmp 파일로 저장하는 코드
	    char filename[1024];
	    snprintf(filename, sizeof(filename), "output_frame_%03d.bmp", frame_num);
	    save_yuv420p_as_bmp(filename, frame, WIDTH, HEIGHT);
	    */
	}
    }
    return 0;
}

void save_yuv420p_as_bmp(const char *filename, AVFrame *frame, int width, int height) {
    struct SwsContext *img_convert_ctx = sws_getContext(
        width, height, AV_PIX_FMT_YUV420P,
        width, height, AV_PIX_FMT_RGB24,
        SWS_BICUBIC, NULL, NULL, NULL
    );

    if (!img_convert_ctx) {
        fprintf(stderr, "Error initializing sws context\n");
        return;
    }

    AVFrame *rgb_frame = av_frame_alloc();
    int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, width, height, 1);
    uint8_t *rgb_buffer = (uint8_t *)av_malloc(num_bytes);

    av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, rgb_buffer, AV_PIX_FMT_RGB24, width, height, 1);

    // YUV420P -> RGB24 변환
    sws_scale(img_convert_ctx, (const uint8_t * const *)frame->data, frame->linesize, 0, height,
              rgb_frame->data, rgb_frame->linesize);

    // BMP로 저장
    FILE *f;
    unsigned int filesize = 54 + 3 * width * height;  // 54바이트 헤더 + 픽셀 데이터 크기

    unsigned char bmpfileheader[14] = {
        'B', 'M',              // 매직 넘버
        filesize & 0xFF, (filesize >> 8) & 0xFF, (filesize >> 16) & 0xFF, (filesize >> 24) & 0xFF,
        0, 0, 0, 0,            // 예약된 필드
        54, 0, 0, 0            // 데이터 시작 위치(헤더 크기)
    };
    unsigned char bmpinfoheader[40] = {
        40, 0, 0, 0,           // 정보 헤더 크기
        width & 0xFF, (width >> 8) & 0xFF, (width >> 16) & 0xFF, (width >> 24) & 0xFF,   // 이미지 너비
        height & 0xFF, (height >> 8) & 0xFF, (height >> 16) & 0xFF, (height >> 24) & 0xFF,  // 이미지 높이
        1, 0,                  // 플레인 수
        24, 0,                 // 비트 수(24 비트)
        0, 0, 0, 0,            // 압축 방식(없음)
        0, 0, 0, 0,            // 이미지 크기(비압축)
        0, 0, 0, 0,            // 수평 해상도
        0, 0, 0, 0,            // 수직 해상도
        0, 0, 0, 0,            // 색상 수
        0, 0, 0, 0             // 중요한 색상 수
    };

    f = fopen(filename, "wb");
    fwrite(bmpfileheader, 1, 14, f);
    fwrite(bmpinfoheader, 1, 40, f);

    // 픽셀 데이터를 저장할 때 역순으로 저장
    for (int y = height - 1; y >= 0; y--) {  // 이미지의 각 라인을 아래에서 위로 읽음
        fwrite(rgb_frame->data[0] + y * rgb_frame->linesize[0], 1, 3 * width, f);
    }

    fclose(f);
    av_free(rgb_buffer);
    av_frame_free(&rgb_frame);
    sws_freeContext(img_convert_ctx);
}


static inline int clip(int value, int min, int max)
{
    return (value > max ? max : (value < min ? min : value));
}

static void sigHandler(int signo)
{
    cond = 0;
}



// YUYV 데이터를 BMP로 저장하는 함수 추가
void save_yuyv_as_bmp(const char *filename, unsigned char *yuyv, int width, int height) {
    FILE *f;
    unsigned int filesize = 54 + 3 * width * height;  // 54바이트 헤더 + 픽셀 데이터 크기

    unsigned char bmpfileheader[14] = {
        'B', 'M',              // 매직 넘버
        filesize & 0xFF, (filesize >> 8) & 0xFF, (filesize >> 16) & 0xFF, (filesize >> 24) & 0xFF,
        0, 0, 0, 0,            // 예약된 필드
        54, 0, 0, 0            // 데이터 시작 위치(헤더 크기)
    };
    unsigned char bmpinfoheader[40] = {
        40, 0, 0, 0,           // 정보 헤더 크기
        width & 0xFF, (width >> 8) & 0xFF, (width >> 16) & 0xFF, (width >> 24) & 0xFF,   // 이미지 너비
        height & 0xFF, (height >> 8) & 0xFF, (height >> 16) & 0xFF, (height >> 24) & 0xFF,  // 이미지 높이
        1, 0,                  // 플레인 수
        24, 0,                 // 비트 수(24 비트)
        0, 0, 0, 0,            // 압축 방식(없음)
        0, 0, 0, 0,            // 이미지 크기(비압축)
        0, 0, 0, 0,            // 수평 해상도
        0, 0, 0, 0,            // 수직 해상도
        0, 0, 0, 0,            // 색상 수
        0, 0, 0, 0             // 중요한 색상 수
    };

    f = fopen(filename, "wb");
    fwrite(bmpfileheader, 1, 14, f);
    fwrite(bmpinfoheader, 1, 40, f);

    // YUYV 데이터를 RGB로 변환하여 BMP로 저장
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x += 2) {
            // YUYV 포맷에서 Y0 U Y1 V 추출
            unsigned char y0 = yuyv[2 * x];
            unsigned char u = yuyv[2 * x + 1];
            unsigned char y1 = yuyv[2 * x + 2];
            unsigned char v = yuyv[2 * x + 3];

            // YUV를 RGB로 변환 (Y0 U V)
            int r = clip((298 * y0 + 409 * v + 128) >> 8, 0, 255);
            int g = clip((298 * y0 - 100 * u - 208 * v + 128) >> 8, 0, 255);
            int b = clip((298 * y0 + 516 * u + 128) >> 8, 0, 255);

            // BMP 파일에 픽셀 저장 (BGR 순서)
            fwrite(&b, 1, 1, f);
            fwrite(&g, 1, 1, f);
            fwrite(&r, 1, 1, f);

            // YUV를 RGB로 변환 (Y1 U V)
            r = clip((298 * y1 + 409 * v + 128) >> 8, 0, 255);
            g = clip((298 * y1 - 100 * u - 208 * v + 128) >> 8, 0, 255);
            b = clip((298 * y1 + 516 * u + 128) >> 8, 0, 255);

            // BMP 파일에 두 번째 픽셀 저장 (BGR 순서)
            fwrite(&b, 1, 1, f);
            fwrite(&g, 1, 1, f);
            fwrite(&r, 1, 1, f);
        }
        yuyv += width * 2;  // YUYV는 한 픽셀당 2바이트
    }

    fclose(f);
}

// YUV420P 데이터를 RGB565로 변환하는 함수
void yuv420p_to_rgb565(unsigned char *yuv420p_data[3], unsigned short *rgb565_data, int width, int height, int fb_width) {
    int y, x, y_stride, uv_stride;
    int r, g, b;
    int y_value, u_value, v_value;

    y_stride = width;
    uv_stride = width / 2;
    int start_x = 400;
    int start_y = 250;

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            int y_index = y * y_stride + x;
            int uv_index = (y / 2) * uv_stride + (x / 2);

            y_value = yuv420p_data[0][y_index];
            u_value = yuv420p_data[1][uv_index] - 128;
            v_value = yuv420p_data[2][uv_index] - 128;

            // YUV -> RGB 변환 공식
            r = clip((298 * y_value + 409 * v_value + 128) >> 8, 0, 255);
            g = clip((298 * y_value - 100 * u_value - 208 * v_value + 128) >> 8, 0, 255);
            b = clip((298 * y_value + 516 * u_value + 128) >> 8, 0, 255);

            // RGB888 -> RGB565로 변환
            rgb565_data[(start_y + y) * fb_width + (start_x + x)] = ((r & 0xF8) << 8) | ((g & 0xFC) << 3) | (b >> 3);
        }
    }
}
