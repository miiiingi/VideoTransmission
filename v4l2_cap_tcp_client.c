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
static void save_framebuffer_as_bmp(const char *filename, unsigned short *fbPtr, int width, int height);

static inline int clip(int value, int min, int max);
static void sigHandler(int signo);

int main(int argc, char** argv) 
{
    signal(SIGINT, sigHandler);
    int cam_fd, fb_fd;
    int fbSize, BUFFER_SIZE = WIDTH * HEIGHT * 2; // YUYV는 픽셀당 2바이트 
    unsigned char* frame_buffer = malloc(BUFFER_SIZE);

    int sock;
    struct sockaddr_in server_addr;
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
	memset(frame_buffer, 0, BUFFER_SIZE);
	int total_received = 0;	
	/*
	 * 원래 비디오 프레임으로부터 데이터를 받아올 때, 고정된 크기의 버퍼를 받아왔었는데, 그렇게 하면 한 번에 많은 양의 데이터 들어왔을 때, 처리가 어렵다. 따라서 한 라인씩 받으면서 포인터를 옮겨주고 변환하는 과정을 반복해야하는데 그것을 위한 do-while문 
	 * 그리고 프레임 버퍼를 초기화해줄 때, 동적할당된 프레임 버퍼 변수 자체를 새롭게 초기화해주어야한다. 
	 * */
	do{
	    int bytes_received = recv(sock, frame_buffer + total_received, BUFFER_SIZE - total_received, 0);
	    if (bytes_received < 0) {
		perror("recv failed");
		free(frame_buffer);
		close(sock);
		exit(EXIT_FAILURE);
	    } else if (bytes_received == 0) {
		printf("Client Quit\n");
		free(frame_buffer);
		close(sock);
		exit(EXIT_FAILURE);
	    }
	    total_received += bytes_received;

	    printf("Received %zd bytes from server\n", bytes_received);
	    yuyv2Rgb565(frame_buffer, fbPtr, WIDTH, HEIGHT);

	} while(total_received < BUFFER_SIZE);
    };
    save_framebuffer_as_bmp("framebuffer_output.bmp", fbPtr, WIDTH, HEIGHT);
    printf("\nGood Bye!!!\n");
    munmap(fbPtr, fbSize);
    free(rgbBuffer);

    // 파일디스크립터 정리
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
    printf("vinfo.yres: %d\n", vinfo.yres);
    printf("vinfo.yres_virtual: %d\n", vinfo.yres_virtual);

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

// BMP 파일로 프레임 버퍼 저장 함수
void save_framebuffer_as_bmp(const char *filename, unsigned short *fbPtr, int width, int height) {
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

    unsigned char *img = (unsigned char *)malloc(3 * width * height);
    memset(img, 0, 3 * width * height);

    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            unsigned short pixel = fbPtr[y * width + x];
            unsigned char r = (pixel >> 11) & 0x1F;
            unsigned char g = (pixel >> 5) & 0x3F;
            unsigned char b = pixel & 0x1F;

            img[(x + (height - 1 - y) * width) * 3 + 2] = (r << 3);
            img[(x + (height - 1 - y) * width) * 3 + 1] = (g << 2);
            img[(x + (height - 1 - y) * width) * 3 + 0] = (b << 3);
        }
    }

    f = fopen(filename, "wb");
    fwrite(bmpfileheader, 1, 14, f);
    fwrite(bmpinfoheader, 1, 40, f);
    fwrite(img, 3, width * height, f);
    fclose(f);
    free(img);
}

static inline int clip(int value, int min, int max)
{
    return (value > max ? max : (value < min ? min : value));
}

static void sigHandler(int signo)
{
    cond = 0;
}


