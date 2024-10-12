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

#define WIDTH 		    800
#define HEIGHT 		    600  
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
// struct v4l2_buffer buf;                     // V4L2에서 사용할 메모리 버퍼
static int cond = 1;
//프레임 버퍼에 대한 다양한 화면 속성 정보를 담고있다.
//xres, yres, xres_virtual, yres_virtual
//bits_per_pixel: 한 픽셀을 표현하는 데 사용되는 비트수(e.g. 16, 24, 32 비트)

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

void* client_handler(void* arg){
    int client_fd = *((int *)arg);
    free(arg);

    while(cond){
	send(client_fd, buffers[buf.index].start, buffer_size, 0);
	usleep(1000);
    }

    close(client_fd);

    return NULL;
}

void* server_handler(void* arg){
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
	pthread_create(&tid, NULL, client_handler, client_fd);
	pthread_detach(tid);
    }

    close(server_fd);
    return NULL;
}

int server_setup(){
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
        
        // YUYV 데이터를 RGB565로 변환
	printf("buf.index: %d\n", buf.index);

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
