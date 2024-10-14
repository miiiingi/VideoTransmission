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
#include <linux/videodev2.h>
#include <arpa/inet.h>

#define WIDTH 		    800
#define HEIGHT 		    600  

#define VIDEO_DEV     "/dev/video0"
#define FB_DEV        "/dev/fb0"
#define SERVER_PORT 8080
#define SERVER_IP "127.0.0.1"

typedef unsigned char uchar;

struct buffer {
    void   *start;
    size_t length;
};

static int cond = 1;
//프레임 버퍼에 대한 다양한 화면 속성 정보를 담고있다.
//xres, yres, xres_virtual, yres_virtual
//bits_per_pixel: 한 픽셀을 표현하는 데 사용되는 비트수(e.g. 16, 24, 32 비트)
static struct fb_var_screeninfo vinfo;

static void sigHandler(int signo)
{
    cond = 0;
}

static inline int clip(int value, int min, int max)
{
    return (value > max ? max : (value < min ? min : value));
}

// YUYV 데이터를 RGB565로 변환하는 함수
void yuyv2Rgb565(uchar *yuyv, unsigned short *rgb, int width, int height) 
{
    uchar* in = (uchar*)yuyv;
    unsigned short pixel;
    int istride = width*2;     /* 이미지의 폭을 넘어가면 다음 라인으로 내려가도록 설정 */
    int x, y, j;
    int y0, u, y1, v, r, g, b;
    long loc = 0;
    /*
    printf("vinfo.xres: %d\n", vinfo.xres);
    printf("vinfo.yres: %d\n", vinfo.yres);
    printf("width: %d\n", width);
    printf("height %d\n", height);
    */
    for (y = 0; y < height; ++y) {
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
            rgb[loc++] = pixel;

            /* YUV를 RGB로 전환 : Y1 + U + V */
            r = clip((298 * y1 + 409 * v + 128) >> 8, 0, 255);
            g = clip((298 * y1 - 100 * u - 208 * v + 128) >> 8, 0, 255);
            b = clip((298 * y1 + 516 * u + 128) >> 8, 0, 255);
            pixel = ((r>>3)<<11)|((g>>2)<<5)|(b>>3);      /* 16비트 컬러로 전환 */
            rgb[loc++] = pixel;
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
    vinfo.yres_virtual = vinfo.yres * 2;

    if(ioctl(fd, FBIOPUT_VSCREENINFO, &vinfo)){
	perror("Error setting variable screen info");
	close(fd);
	return -1;
    }
    */
    printf("vinfo.yres: %d\n", vinfo.yres);
    printf("vinfo.yres_virtual: %d\n", vinfo.yres_virtual);

    //
    // printf("vinfo.bits_per_pixel: %d\n", vinfo.bits_per_pixel);
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

void draw_framebuffer(int fb_fd, unsigned short *fbPtr, unsigned short *rgbBuffer, int fbSize, int *currentBuffer){
    int location = (*currentBuffer) * vinfo.yres * vinfo.xres;

    memset(fbPtr + location, 0, fbSize / 2);

    memcpy(fbPtr + location, rgbBuffer, fbSize / 2);

    vinfo.yoffset = (*currentBuffer) * vinfo.yres;

    if(ioctl(fb_fd, FBIOPAN_DISPLAY, &vinfo)){
	perror("Failed to panning display");
    }

    *currentBuffer = (*currentBuffer) ? 0: 1;

}

int main(int argc, char** argv) 
{
    signal(SIGINT, sigHandler);
    unsigned short *rgbBuffer, *fbPtr;
    int cam_fd, fb_fd;
    int fbSize, BUFFER_SIZE = WIDTH * HEIGHT * 2; // YUYV는 픽셀당 2바이트 
    int currentBuffer;
    // struct buffer buffers[BUFFER_COUNT];
    struct v4l2_buffer buf;                     // V4L2에서 사용할 메모리 버퍼

    int sock;
    struct sockaddr_in server_addr;
    char buffer[BUFFER_SIZE];
    ssize_t bytes_received;
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

    // V4L2를 이용한 영상의 캡쳐 및 표시
    while (cond) {
	memset(&buffer, 0, sizeof(buffer));
	// memset(&fbPtr, 0, sizeof(fbPtr));
	bytes_received = recv(sock, buffer, BUFFER_SIZE, 0);
        if (bytes_received < 0) {
            perror("recv failed");
            break;
        } else if (bytes_received == 0) {
            printf("Server closed connection\n");
            break;
        }

        printf("Received %zd bytes from server\n", bytes_received);
        yuyv2Rgb565(buffer, rgbBuffer, WIDTH, HEIGHT);

	// draw_framebuffer(fb_fd, fbPtr, rgbBuffer, fbSize, &currentBuffer);

        // 프레임버퍼에 RGB565 데이터 쓰기
        memcpy(fbPtr, rgbBuffer, fbSize); 
    }

    printf("\nGood Bye!!!\n");
    munmap(fbPtr, fbSize);
    free(rgbBuffer);

    // 파일디스크립터 정리
    close(cam_fd);
    close(fb_fd);

    return 0;
}
