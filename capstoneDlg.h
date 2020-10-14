
// capstoneDlg.h: 헤더 파일
//

#pragma once

#include "FaceRecognition.h"

// CcapstoneDlg 대화 상자
class CcapstoneDlg : public CDialogEx
{
// 생성입니다.
public:
	CcapstoneDlg(CWnd* pParent = nullptr);	// 표준 생성자입니다.

// 대화 상자 데이터입니다.
#ifdef AFX_DESIGN_TIME
	enum { IDD = IDD_CAPSTONE_DIALOG };
#endif

	protected:
	virtual void DoDataExchange(CDataExchange* pDX);	// DDX/DDV 지원입니다.


// 구현입니다.
protected:
	HICON m_hIcon;

	// 생성된 메시지 맵 함수
	virtual BOOL OnInitDialog();
	afx_msg void OnSysCommand(UINT nID, LPARAM lParam);
	afx_msg void OnPaint();
	afx_msg HCURSOR OnQueryDragIcon();
	DECLARE_MESSAGE_MAP()

	HWND hwndDesktop;		//클라이언트 정보 저장
	string haarcascadePath = "C:\\OpenCV\\build\\etc\\haarcascades\\haarcascade_frontalface_alt.xml";
	CascadeClassifier faceCascade;
	FaceRecognition *recognizer;

	std::vector<Mat> imgs;
	std::vector<std::string> imgName;
	
	std::string NameCropping(string & str);
	void LoadKnownImage();
	void KnownImagePreprocessing();
	bool HaarCascadeInit();
	Mat hwnd2mat(HWND hwnd);
	void DetectAndDisplay(Mat frame);

	void SelfTrain();

public:
	afx_msg void OnBnClickedTestButton();
	afx_msg void OnBnClickedLoadimage();
};
