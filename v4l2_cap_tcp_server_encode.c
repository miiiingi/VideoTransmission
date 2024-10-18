#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <errno.h>
#include <pthread.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <linux/videodev2.h>
#include <libavutil/frame.h>
#include <libavutil/imgutils.h>
#include <libswscale/swscale.h>
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>  // AVFormatContext, avformat_alloc_output_context2, av_interleaved_write_frame
#include <libavformat/avio.h>      // AVIOContext, avio_open, avio_close
#include <libavutil/avutil.h>      // 기본 유틸리티 함수


#define WIDTH 		    640
#define HEIGHT 		    480  
#define BUFFER_COUNT	4  	// 버퍼의 개수
#define VIDEO_DEV     "/dev/video0"
#define SERVER_PORT 8080
#define SERVER_IP "127.0.0.1"

int buffer_size = WIDTH * HEIGHT * 2;
typedef unsigned char uchar;

struct buffer {
    void   *start;
    size_t length;
};
struct buffer buffers[BUFFER_COUNT];
struct v4l2_buffer buf;                     // V4L2에서 사용할 메모리 버퍼
static int cond = 1;
/*
 * 프레임 버퍼에 대한 다양한 화면 속성 정보를 담고있다.
 * xres, yres, xres_virtual, yres_virtual
 * bits_per_pixel: 한 픽셀을 표현하는 데 사용되는 비트수(e.g. 16, 24, 32 비트)
*/

static void sigHandler(int signo);
static int init_v4l2(int *fd, struct buffer *buffers);
static void* client_handler(void* arg);
static void* server_handler(void* arg);
static int server_setup();
static int convert_yuyv422_to_yuv420p(uint8_t *input_data, uint8_t **output_data, int width, int height, struct SwsContext *sws_ctx, AVFrame *pFrameIn, AVFrame *pFrameOut);
void save_yuv420p_as_bmp(const char *filename, AVFrame *yuv420p_frame, int width, int height);
// H.264 인코딩을 위한 함수 선언
void save_to_h264(int client_fd, AVCodecContext *enc_ctx, AVPacket *pkt, FILE *outfile);

int main(int argc, char** argv) 
{
    int cam_fd;
    signal(SIGINT, sigHandler);

    // V4L2 초기화
    if (init_v4l2(&cam_fd, buffers) < 0) {
        fprintf(stderr, "V4L2 initialization failed\n");
        return -1;
    }

    int server_fd = server_setup();
    if(server_fd < 0){
	perror("server setup failed\n");
	return -1;
    }

    pthread_t server_thread;
    pthread_create(&server_thread, NULL, server_handler, &server_fd);

    int addrlen = sizeof(struct sockaddr_in);
    struct sockaddr_in client_addr;

    // V4L2를 이용한 영상의 캡쳐 및 표시
    while (cond) {
        // 버퍼 초기화
        memset(&buf, 0, sizeof(buf));

        // MMAP 기반으로 영상 캡쳐
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;

	// STREAMON 명령으로 캡쳐한 비디오 프레임을 디큐한다. 
        if (ioctl(cam_fd, VIDIOC_DQBUF, &buf) < 0) {
            perror("Failed to dequeue buffer");
            break;
        }
        
        // 버퍼를 다시 큐에 넣기
        if (ioctl(cam_fd, VIDIOC_QBUF, &buf) < 0) {
            perror("Failed to queue buffer");
            break;
        }
    }

    printf("\nGood Bye!!!\n");

    // 캡쳐 종료
    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    ioctl(cam_fd, VIDIOC_STREAMOFF, &type);

    // 메모리 정리
    for (int i = 0; i < BUFFER_COUNT; i++) {
        munmap(buffers[i].start, buffers[i].length);
    }

    // 파일디스크립터 정리
    close(cam_fd);

    return 0;
}
static void sigHandler(int signo)
{
    cond = 0;
}

// V4L2 설정
static int init_v4l2(int *fd, struct buffer *buffers) 
{
    // 장치의 해상도와 포맷 설정하는 구조체
    struct v4l2_format format;
    // v4l2 장치에서 버퍼를 요청할 때 사용하는 구조체
    // 메모리 매핑 방식으로 비디오 프레임을 처리할 버퍼를 할당하고 설정할 때 사용
    struct v4l2_requestbuffers reqbuf;

    // v4l2에서 비디오 프레임 데이터를 처리하기 위해 사용하는 버퍼에 대한 정보를 담고 있다.
    // ioctl함수에서 VIDIOC_QUERYBUF, VIDIOC_QBUF, VIDIOC_DQBUF와 같은 명령어와 함께 사용된다.
    // 이 버퍼는 비디오 데이터를 큐에 넣고 큐에서 빼는 작업을 위해 사용됨.
    struct v4l2_buffer buf;

    // 카메라 장치 열기
    *fd = open(VIDEO_DEV, O_RDWR);
    if (*fd < 0) {
        perror("Failed to open video device");
        return -1;
    }

    // 포맷 설정 (YUYV)
    memset(&format, 0, sizeof(format));
    format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    format.fmt.pix.width = WIDTH;
    format.fmt.pix.height = HEIGHT;
    format.fmt.pix.pixelformat = V4L2_PIX_FMT_YUYV;
    format.fmt.pix.field = V4L2_FIELD_NONE;

    // 카메라 장치의 포맷을 설정
    if (ioctl(*fd, VIDIOC_S_FMT, &format) < 0) {
        perror("Failed to set format");
        close(*fd);
        return -1;
    }
    printf("영상의 해상도 : %d x %d\n", format.fmt.pix.width, format.fmt.pix.height);

    // 버퍼 요청
    memset(&reqbuf, 0, sizeof(reqbuf));
    reqbuf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    //버퍼의 메모리를 설정함. 아래 처럼 설정하게 되면 MMAP을 통해 커널 메모리를 사용자 공간으로 매핑하는 것
    reqbuf.memory = V4L2_MEMORY_MMAP;
    reqbuf.count = BUFFER_COUNT;

    // VIDIOC_REQBUF는 버퍼를 할당해달라는 시스템 콜
    if (ioctl(*fd, VIDIOC_REQBUFS, &reqbuf) < 0) {
        perror("Failed to request buffers");
        close(*fd);
        return -1;
    }

    // 버퍼 매핑
    for (int i = 0; i < BUFFER_COUNT; i++) {
        memset(&buf, 0, sizeof(buf));
        buf.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
        buf.memory = V4L2_MEMORY_MMAP;
        buf.index = i;

	// 드라이버에서 비디오 장치의 버퍼에 대한 정보를 조회하는 명령 -> VIDIOC_REQBUFS로 버퍼를 요청한 후, 각 버퍼에 대해 VIDIOC_QUERYBUF를 사용하여 해당 버퍼의 상태를 확인하고 메모리를 매핑하는데 필요한 정보를 가져옴
        if (ioctl(*fd, VIDIOC_QUERYBUF, &buf) < 0) {
            perror("Failed to query buffer");
            close(*fd);
            return -1;
        }

	//buffers는 우리가 정의한 구조체 변수로 화면에 보여줄 버퍼를 다루게되고
	//buf는 v4l2에서 다루는 버퍼로 커널 영역에 존재한다. 
	//카메라로부터 비디오 프레임을 캡쳐하면 buf에 저장되는데, 그것을 유저 영역에서 접근할 수 없기 때문에 mmap을 통해서 buffers라는 구조체로 buf에 접근할 수 있도록 설정해주는 것!
        buffers[i].length = buf.length;
	// start 인자에 mmap을 할당한다. 
        buffers[i].start = mmap(NULL, buf.length, PROT_READ | PROT_WRITE, MAP_SHARED, *fd, buf.m.offset);
        if (buffers[i].start == MAP_FAILED) {
            perror("Failed to mmap buffer");
            close(*fd);
            return -1;
        }

        // 큐에 버퍼를 넣음
        if (ioctl(*fd, VIDIOC_QBUF, &buf) < 0) {
            perror("Failed to queue buffer");
            close(*fd);
            return -1;
        }
    }

    enum v4l2_buf_type type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
    // 비디오 스트리밍을 시작하는 시스템 콜 -> 이 시스템 콜을 호출하면 큐에있는 버퍼에 비디오 프레임이 저장되게 되는데 이것을 사용하기 위해 나중에 디큐를 해주는 것!
    if (ioctl(*fd, VIDIOC_STREAMON, &type) < 0) {
        perror("Failed to start capturing");
        close(*fd);
        return -1;
    }

    return 0;
}

void* client_handler(void* arg) {
    int client_fd = *((int *)arg);
    free(arg);

    int frame_size_in = WIDTH * HEIGHT * 2;
    int frame_size_out = WIDTH * HEIGHT * 3 / 2;

    // FFmpeg 컨텍스트 및 프레임 설정
    AVFrame *pFrameIn = av_frame_alloc();
    if(!pFrameIn){
	perror("pFramIn Allocation Failed");
	return NULL;
    }
    pFrameIn->format = AV_PIX_FMT_YUYV422;
    pFrameIn->width = WIDTH;
    pFrameIn->height = HEIGHT;
    if(av_frame_get_buffer(pFrameIn, 1) < 0){
	perror("pFrameIn Buffer Allocation Failed");
	av_frame_free(&pFrameIn);
	return NULL;
    }

    AVFrame *pFrameOut = av_frame_alloc();
    if(!pFrameOut){
	perror("pFrameOut Allocation Failed");
	av_frame_free(&pFrameIn);
	av_frame_free(&pFrameOut);
	return NULL;
    }
    pFrameOut->format = AV_PIX_FMT_YUV420P;
    pFrameOut->width = WIDTH;
    pFrameOut->height = HEIGHT;
    if(av_frame_get_buffer(pFrameOut, 1) < 0){
	perror("pFrameOut Buffer Allocation Failed");
	av_frame_free(&pFrameIn);
	av_frame_free(&pFrameOut);
    }

    /*
     * yuyv 데이터를 yuv420p로 변환하는 context를 선언해줌
     */
    struct SwsContext *sws_ctx = sws_getContext(WIDTH, HEIGHT, AV_PIX_FMT_YUYV422, WIDTH, HEIGHT, AV_PIX_FMT_YUV420P, SWS_LANCZOS, NULL, NULL, NULL);
    if(!sws_ctx){
	av_frame_free(&pFrameIn);
	av_frame_free(&pFrameOut);
	return NULL;
    }

    uint8_t *yuv420p_data[3];
    yuv420p_data[0] = malloc(WIDTH * HEIGHT);                  // Y plane
    yuv420p_data[1] = malloc((WIDTH / 2) * (HEIGHT / 2));      // U plane
    yuv420p_data[2] = malloc((WIDTH / 2) * (HEIGHT / 2));      // V plane
							       //
    /*
     * yuv420p에서 h.264로 변환하기 위한 코덱 정의
     */
    AVCodecContext *enc_ctx;
    const AVCodec *codec = avcodec_find_encoder(AV_CODEC_ID_H264);
    if (!codec) {
        fprintf(stderr, "H.264 codec not found\n");
        return NULL;
    }

    /*
     * 위에서 정한 코덱을 통해서 인코딩 하는 컨텍스트 정의 및 생성
     */
    enc_ctx = avcodec_alloc_context3(codec);
    if (!enc_ctx) {
        fprintf(stderr, "Could not allocate video codec context\n");
        return NULL;
    }
    
    enc_ctx->bit_rate = 4000000;  // 비트레이트 설정
    enc_ctx->width = WIDTH;
    enc_ctx->height = HEIGHT;
    enc_ctx->time_base = (AVRational){1, 120};  // 25 fps
    enc_ctx->framerate = (AVRational){120, 1};
    enc_ctx->gop_size = 4;  // GOP 크기 설정 (10 프레임마다 키프레임)
    enc_ctx->max_b_frames = 0;
    enc_ctx->pix_fmt = AV_PIX_FMT_YUV420P;
    
    /*
     * 위에서 정한 코덱과 인코딩 컨텍스트를 연다.
     */
    if (avcodec_open2(enc_ctx, codec, NULL) < 0) {
        fprintf(stderr, "Could not open H.264 codec\n");
        avcodec_free_context(&enc_ctx);
        return NULL;
    }

    AVPacket *pkt = av_packet_alloc();
    if (!pkt) {
        fprintf(stderr, "Could not allocate packet\n");
        avcodec_free_context(&enc_ctx);
        return NULL;
    }
    
    // H.264 파일을 저장할 파일 포인터
    FILE *outfile = fopen("server_send.h264", "wb");
    // BMP로 저장할 파일명 지정
    const char *bmp_filename = "output_frame.bmp";
    if (!outfile) {
        perror("Failed to open output file");
        return NULL;
    }
    int64_t pts = 0;
    int frame_index = 0;
    AVRational time_base = enc_ctx->time_base;

    while (cond) {
	/*
	 * buffers[buf.index].start: yuyv 데이터가 들어있음.
	 * yuv420p_data: yuyv로부터 변환된 yuv420p을 담을 버퍼.
	 * pFrameIn: yuyv를 담을 버퍼 buffers[buf.index].start에 담긴 데이터를 복사한다.
	 * pFrameOut: yuv420p를 담을 버퍼 yuv_420p_data에 담긴 데이터를 복사한다.
	 */
        if (convert_yuyv422_to_yuv420p(buffers[buf.index].start, yuv420p_data, WIDTH, HEIGHT, sws_ctx, pFrameIn, pFrameOut) == 0) {
	    /*
	     * 위에서 정한 인코딩 컨텍스트를 이용해서 pFrameOut에 담긴 yuv420p 데이터를 h.264로 인코딩한다. 
	     */
	    pFrameOut->pts = pts;
	    pts += av_rescale_q(1, (AVRational){1, 40}, time_base);
	    if(avcodec_send_frame(enc_ctx, pFrameOut) < 0){
		perror("Error Sending Frame To Encoder\n");
		break;
	    }
	    /*
	     * 클라이언트에게 h.264로 인코딩된 패킷을 전송한다.
	     */
	    // save_yuv420p_as_bmp(bmp_filename, pFrameOut, WIDTH, HEIGHT);
	    save_to_h264(client_fd, enc_ctx, pkt, outfile);
        }
    }

    av_packet_free(&pkt);
    free(yuv420p_data[0]);
    free(yuv420p_data[1]);
    free(yuv420p_data[2]);
    av_frame_free(&pFrameIn);
    av_frame_free(&pFrameOut);
    avcodec_free_context(&enc_ctx);
    sws_freeContext(sws_ctx);
    close(client_fd);
    return NULL;
}

void* server_handler(void* arg)
{
    int server_fd = *((int *)arg);
    struct sockaddr_in client_addr;
    socklen_t addrlen = sizeof(client_addr);

    while(cond){
	int *client_fd = (int *)malloc(sizeof(int));
	if ((*client_fd = accept(server_fd, (struct sockaddr *)&client_addr, &addrlen)) < 0) {
            perror("accept failed");
            free(client_fd);
            continue;
        }

	pthread_t tid;
	int thread_status = pthread_create(&tid, NULL, client_handler, client_fd);
	if (thread_status != 0) {
	    perror("pthread_create failed");
	} else {
	    printf("Thread created successfully\n");
	}
	pthread_detach(tid);
    }

    close(server_fd);
    return NULL;
}

// YUYV422 → YUV420P 변환을 위한 FFmpeg 사용 함수
int convert_yuyv422_to_yuv420p(uint8_t *input_data, uint8_t **output_data, int width, int height, struct SwsContext *sws_ctx, AVFrame *pFrameIn, AVFrame *pFrameOut) {
    // 입력 프레임에 YUYV422 데이터 설정
    if (av_image_fill_arrays(pFrameIn->data, pFrameIn->linesize, input_data, AV_PIX_FMT_YUYV422, width, height, 1) < 0) {
        fprintf(stderr, "YUYV422 데이터 설정 실패\n");
        return -1;
    }

    // YUV420P로 변환
    if (sws_scale(
            sws_ctx,
            (const uint8_t *const *)pFrameIn->data,
            pFrameIn->linesize,
            0, height,
            pFrameOut->data,
            pFrameOut->linesize
        ) <= 0) {
        fprintf(stderr, "YUYV422 to YUV420P 변환 실패\n");
        return -1;
    }

    // 변환된 YUV420P 데이터 할당 및 복사
    int y_size = width * height;
    int uv_size = (width / 2) * (height / 2);
    memcpy(output_data[0], pFrameOut->data[0], y_size);              // Y
    memcpy(output_data[1], pFrameOut->data[1], uv_size);             // U
    memcpy(output_data[2], pFrameOut->data[2], uv_size);             // V

    return 0;
}

// YUV420 데이터를 RGB로 변환하고 BMP로 저장하는 함수
void save_yuv420p_as_bmp(const char *filename, AVFrame *yuv420p_frame, int width, int height) {
    struct SwsContext *img_convert_ctx = sws_getContext(
        width, height, AV_PIX_FMT_YUV420P,
        width, height, AV_PIX_FMT_RGB24,
        SWS_BICUBIC, NULL, NULL, NULL
    );

    if (!img_convert_ctx) {
        fprintf(stderr, "Error initializing sws context\n");
        return;
    }

    // RGB24 프레임 할당
    AVFrame *rgb_frame = av_frame_alloc();
    int num_bytes = av_image_get_buffer_size(AV_PIX_FMT_RGB24, width, height, 1);
    uint8_t *rgb_buffer = (uint8_t *)av_malloc(num_bytes);

    av_image_fill_arrays(rgb_frame->data, rgb_frame->linesize, rgb_buffer, AV_PIX_FMT_RGB24, width, height, 1);

    // YUV420P -> RGB24 변환
    sws_scale(img_convert_ctx, (const uint8_t * const *)yuv420p_frame->data, yuv420p_frame->linesize, 0, height,
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

    // 상하 반전된 픽셀 데이터를 저장하면서 RGB -> BGR 변환
    for (int i = height - 1; i >= 0; i--) {
        uint8_t *row_ptr = rgb_frame->data[0] + i * rgb_frame->linesize[0];
        for (int j = 0; j < width; j++) {
            // RGB 값을 BGR로 변환하여 저장
            uint8_t bgr_pixel[3] = {row_ptr[3 * j + 2], row_ptr[3 * j + 1], row_ptr[3 * j]};
            fwrite(bgr_pixel, 1, 3, f);
        }
    }

    fclose(f);

    // 자원 해제
    av_free(rgb_buffer);
    av_frame_free(&rgb_frame);
    sws_freeContext(img_convert_ctx);
}

int server_setup()
{
    int server_fd;
    struct sockaddr_in server_addr;

    if((server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0){
	perror("socket failed\n");
	return -1;
    }

    server_addr.sin_family = AF_INET;
    server_addr.sin_addr.s_addr = inet_addr(SERVER_IP);
    server_addr.sin_port = htons(SERVER_PORT);

    if(bind(server_fd, (struct sockaddr *)&server_addr, sizeof(server_addr))<0){
	perror("bind failed\n");
	close(server_fd);
	return -1;
    }

    if(listen(server_fd, 3) < 0){
	perror("listen failed\n");
	close(server_fd);
	return -1;
    }

    return server_fd;
}

void save_to_h264(int client_fd, AVCodecContext *enc_ctx, AVPacket *pkt, FILE *outfile) {
    while (avcodec_receive_packet(enc_ctx, pkt) == 0) {
        // 클라이언트로 인코딩된 패킷 전송
        int sent_bytes = send(client_fd, pkt->data, pkt->size, 0);
        printf("\rSent H.264 data: %zd bytes", sent_bytes);
	fflush(stdout);

        // 인코딩된 패킷을 H.264 파일로 저장
        fwrite(pkt->data, 1, pkt->size, outfile);

        av_packet_unref(pkt);  // 패킷 해제
    }
}
