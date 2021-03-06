﻿
// capstoneDlg.cpp: 구현 파일
//

#include "pch.h"
#include "framework.h"
#include "capstone.h"
#include "capstoneDlg.h"
#include "afxdialogex.h"

#ifdef _DEBUG
#define new DEBUG_NEW
#endif


// 응용 프로그램 정보에 사용되는 CAboutDlg 대화 상자입니다.

class CAboutDlg : public CDialogEx
{
public:
	CAboutDlg();

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_ABOUTBOX };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);    // DDX/DDV 지원입니다.

// 구현입니다.
protected:
	DECLARE_MESSAGE_MAP()
};

CAboutDlg::CAboutDlg() : CDialogEx(IDD_ABOUTBOX)
{
}

void CAboutDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CAboutDlg, CDialogEx)
END_MESSAGE_MAP()


// CcapstoneDlg 대화 상자



CcapstoneDlg::CcapstoneDlg(CWnd* pParent /*=nullptr*/)
	: CDialogEx(IDD_CAPSTONE_DIALOG, pParent),
	hwndDesktop(::GetDesktopWindow())
{
	m_hIcon = AfxGetApp()->LoadIcon(IDR_MAINFRAME);
}

void CcapstoneDlg::DoDataExchange(CDataExchange* pDX)
{
	CDialogEx::DoDataExchange(pDX);
}

BEGIN_MESSAGE_MAP(CcapstoneDlg, CDialogEx)
	ON_WM_SYSCOMMAND()
	ON_WM_PAINT()
	ON_WM_QUERYDRAGICON()
	ON_BN_CLICKED(IDC_TEST_BUTTON, &CcapstoneDlg::OnBnClickedTestButton)
	ON_BN_CLICKED(IDC_LOADIMAGE, &CcapstoneDlg::OnBnClickedLoadimage)
END_MESSAGE_MAP()


// CcapstoneDlg 메시지 처리기

BOOL CcapstoneDlg::OnInitDialog()
{
	CDialogEx::OnInitDialog();

	// 시스템 메뉴에 "정보..." 메뉴 항목을 추가합니다.

	// IDM_ABOUTBOX는 시스템 명령 범위에 있어야 합니다.
	ASSERT((IDM_ABOUTBOX & 0xFFF0) == IDM_ABOUTBOX);
	ASSERT(IDM_ABOUTBOX < 0xF000);

	CMenu* pSysMenu = GetSystemMenu(FALSE);
	if (pSysMenu != nullptr)
	{
		BOOL bNameValid;
		CString strAboutMenu;
		bNameValid = strAboutMenu.LoadString(IDS_ABOUTBOX);
		ASSERT(bNameValid);
		if (!strAboutMenu.IsEmpty())
		{
			pSysMenu->AppendMenu(MF_SEPARATOR);
			pSysMenu->AppendMenu(MF_STRING, IDM_ABOUTBOX, strAboutMenu);
		}
	}

	// 이 대화 상자의 아이콘을 설정합니다.  응용 프로그램의 주 창이 대화 상자가 아닐 경우에는
	//  프레임워크가 이 작업을 자동으로 수행합니다.
	SetIcon(m_hIcon, TRUE);			// 큰 아이콘을 설정합니다.
	SetIcon(m_hIcon, FALSE);		// 작은 아이콘을 설정합니다.

	// TODO: 여기에 추가 초기화 작업을 추가합니다.
	recognizer = new FaceRecognition();

	
	//This codes for self train.
	//But not recommend.
	//Use trainer.
	//SelfTrain();

	recognizer->LoadModel();
	return TRUE;  // 포커스를 컨트롤에 설정하지 않으면 TRUE를 반환합니다.
}
void CcapstoneDlg::SelfTrain() {
	LoadKnownImage();
	KnownImagePreprocessing();
}
void CcapstoneDlg::LoadKnownImage() {
	freopen("list.txt", "r", stdin);
	FILE * err = freopen("err.txt", "w", stdout);
	std::string fName;
	while (~scanf("%[^\n]", fName.c_str())) {
		getchar();
		Mat img = imread(fName.c_str(), -1);
		if (img.empty()) continue;
		imgs.push_back(img);
		
		std::string crop = NameCropping(fName);
		
		imgName.push_back(crop);
		printf("Name: %s\n", imgName.back().c_str());
		printf("depth: %d\n", img.depth());
		printf("channel: %d\n", img.channels());
	}
	fclose(err);
}
std::string CcapstoneDlg::NameCropping(std::string & str) {
	std::string res = str;
	auto slash = res.find('/');
	if (slash != string::npos)
		res = res.substr(slash +1);
	
	auto dotPos = res.find('.');
	if (dotPos != std::string::npos)
		res = res.substr(0, dotPos);

	return res;
}
void CcapstoneDlg::KnownImagePreprocessing() {
	
	//recognizer->AddToSVM(imgs);
	//recognizer->SamplingForTraining();
	//recognizer->SetLabels(imgName);
	//recognizer->Train();
	
	recognizer->LoadModel();
}
void CcapstoneDlg::OnSysCommand(UINT nID, LPARAM lParam)
{
	if ((nID & 0xFFF0) == IDM_ABOUTBOX)
	{
		CAboutDlg dlgAbout;
		dlgAbout.DoModal();
	}
	else
	{
		CDialogEx::OnSysCommand(nID, lParam);
	}
}

// 대화 상자에 최소화 단추를 추가할 경우 아이콘을 그리려면
//  아래 코드가 필요합니다.  문서/뷰 모델을 사용하는 MFC 애플리케이션의 경우에는
//  프레임워크에서 이 작업을 자동으로 수행합니다.

void CcapstoneDlg::OnPaint()
{
	if (IsIconic())
	{
		CPaintDC dc(this); // 그리기를 위한 디바이스 컨텍스트입니다.

		SendMessage(WM_ICONERASEBKGND, reinterpret_cast<WPARAM>(dc.GetSafeHdc()), 0);

		// 클라이언트 사각형에서 아이콘을 가운데에 맞춥니다.
		int cxIcon = GetSystemMetrics(SM_CXICON);
		int cyIcon = GetSystemMetrics(SM_CYICON);
		CRect rect;
		GetClientRect(&rect);
		int x = (rect.Width() - cxIcon + 1) / 2;
		int y = (rect.Height() - cyIcon + 1) / 2;

		// 아이콘을 그립니다.
		dc.DrawIcon(x, y, m_hIcon);
	}
	else
	{
		CDialogEx::OnPaint();
	}
}

// 사용자가 최소화된 창을 끄는 동안에 커서가 표시되도록 시스템에서
//  이 함수를 호출합니다.
HCURSOR CcapstoneDlg::OnQueryDragIcon()
{
	return static_cast<HCURSOR>(m_hIcon);
}

Mat CcapstoneDlg::hwnd2mat(HWND hwnd) {
	HDC hwindowDC, hwindowCompatibleDC;

	int height, width, srcheight, srcwidth;
	HBITMAP hbwindow;
	Mat src;
	BITMAPINFOHEADER  bi;

	hwindowDC = ::GetDC(hwnd);
	hwindowCompatibleDC = CreateCompatibleDC(hwindowDC);
	SetStretchBltMode(hwindowCompatibleDC, COLORONCOLOR);

	RECT windowsize;    // get the height and width of the screen
	::GetClientRect(hwnd, &windowsize);

	srcheight = windowsize.bottom;
	srcwidth = windowsize.right;
	height = windowsize.bottom / 1;  //change this to whatever size you want to resize to
	width = windowsize.right / 1;

	src.create(height, width, CV_8UC4);
	
	// create a bitmap
	hbwindow = CreateCompatibleBitmap(hwindowDC, width, height);
	bi.biSize = sizeof(BITMAPINFOHEADER);    //http://msdn.microsoft.com/en-us/library/windows/window/dd183402%28v=vs.85%29.aspx
	bi.biWidth = width;
	bi.biHeight = -height;  //this is the line that makes it draw upside down or not
	bi.biPlanes = 1;
	bi.biBitCount = 32;
	bi.biCompression = BI_RGB;
	bi.biSizeImage = 0;
	bi.biXPelsPerMeter = 0;
	bi.biYPelsPerMeter = 0;
	bi.biClrUsed = 0;
	bi.biClrImportant = 0;

	// use the previously created device context with the bitmap
	SelectObject(hwindowCompatibleDC, hbwindow);
	// copy from the window device context to the bitmap device context
	StretchBlt(hwindowCompatibleDC, 0, 0, width, height, hwindowDC, 0, 0, srcwidth, srcheight, SRCCOPY); //change SRCCOPY to NOTSRCCOPY for wacky colors !
	GetDIBits(hwindowCompatibleDC, hbwindow, 0, height, src.data, (BITMAPINFO*)& bi, DIB_RGB_COLORS);  //copy from hwindowCompatibleDC to hbwindow

	// avoid memory leak
	DeleteObject(hbwindow);
	DeleteDC(hwindowCompatibleDC);
	::ReleaseDC(hwnd, hwindowDC);

	return src;
}

void CcapstoneDlg::OnBnClickedTestButton()
{
	static TCHAR BASED_CODE szFilter[] = _T("이미지 파일(*.BMP, *.GIF, *.JPG, *.PNG) | *.BMP;*.GIF;*.JPG;*.PNG;*.bmp;*.jpg;*.gif;*.png |");

	CFileDialog dlg(TRUE, _T("*.jpg"), _T("image"), OFN_HIDEREADONLY, szFilter);

	if (IDOK == dlg.DoModal()) {
		string pathName = dlg.GetPathName();

		Mat img = imread(pathName, IMREAD_UNCHANGED);
		recognizer->Prediction(img);

	}
	
	
	FaceRecognition* fr = new FaceRecognition();

	return;
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.
	HaarCascadeInit();

	//VideoCapture camera(0);
	//if (!camera.isOpened()) return;
	/*
	while (camera.read(frame)) {
		if (frame.empty()) break;
		DetectAndDisplay(frame);
		waitKey(1);
	}
	*/
	while (1) {
		Mat tmp = hwnd2mat(hwndDesktop);
		//resize(tmp, tmp, Size(tmp.cols / 2, tmp.rows / 2));
		DetectAndDisplay(tmp);
		waitKey(33);
	}
}

bool CcapstoneDlg::HaarCascadeInit() {
	return faceCascade.load(haarcascadePath);
}
void CcapstoneDlg::DetectAndDisplay(Mat frame)
{
	std::vector<Rect> faces;
	Mat frame_gray;
	if (frame.empty()) return;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	faceCascade.detectMultiScale(frame_gray, faces, 1.1, 2, 0 | CASCADE_SCALE_IMAGE, Size(30, 30));

	for (size_t i = 0; i < faces.size(); i++)
	{
		Point center(faces[i].x + faces[i].width / 2, faces[i].y + faces[i].height / 2);
		ellipse(frame, center, Size(faces[i].width / 2, faces[i].height / 2),
			0, 0, 360, Scalar(0, 0, 255), 4, 8, 0);
	}
	resize(frame, frame, Size(frame.cols / 2, frame.rows / 2));
	imshow("Test", frame);
}



void CcapstoneDlg::OnBnClickedLoadimage()
{
	// TODO: 여기에 컨트롤 알림 처리기 코드를 추가합니다.

	static TCHAR BASED_CODE szFilter[] = _T("이미지 파일(*.BMP, *.GIF, *.JPG, *.PNG) | *.BMP;*.GIF;*.JPG;*.PNG;*.bmp;*.jpg;*.gif;*.png |");

	CFileDialog dlg(TRUE, _T("*.jpg"), _T("image"), OFN_HIDEREADONLY, szFilter);

	if (IDOK == dlg.DoModal()) {
		string pathName = dlg.GetPathName();

		Mat img = imread(pathName, IMREAD_UNCHANGED);
		auto res = recognizer->Prediction(img);
		//for (int i = 0; i < res.size(); ++i)
			//MessageBox(res[i].c_str());
		//MessageBox(pathName.c_str());

	}
}
