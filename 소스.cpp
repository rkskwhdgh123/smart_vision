#include <opencv2/opencv.hpp> //헤더파일 포함
#include <iostream> //헤더파일 포함
using namespace cv; //cv 사용
using namespace cv::dnn; //cv밑에 dnn 사용
using namespace std; //std 사용
Mat Remove_Background(Mat img); //배경 지우는 함수 선언
int Emotion_inference(Mat img); //감정을 추론하는 함수 선언
Mat face_chracter(Mat img,int emotion, int chracter_number); //얼굴에 캐릭터 이미지를 씌워주는 함수 선언
const String model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"; //Net에 사용하는 모델의 이름
const String config = "deploy.prototxt"; //Net에 사용하는 모델의 이름
bool key_q = false;  // 키보드 q가 눌렀는지 판단하는 변수 전역으로 선언 
bool key_v = true;  // 키보드 v가 눌렀는지 판단하는 변수 전역으로 선언

int main(void) //메인함수 시작점
{


	VideoCapture cap(0); //VideoCapture객체 생성 0번 노트북 카메라 사용
	if (!cap.isOpened()) { cerr << "Camera open failed!" << endl; return -1; } //카메라가 열리지 않으면 종료
	Net net = readNet(model, config); //Net 생성
	if (net.empty()) { cerr << "Net open failed!" << endl; return -1; } //net이 열리지 않으면 종료

	cout << "****************설명서******************" << endl;   //사용 설명서 메세지 출력
	cout << "esc: 종료" << endl;                                  //사용 설명서 메세지 출력
	cout << "q: 본인 얼굴 보기 on/off" << endl;                   //사용 설명서 메세지 출력
	cout << "c: 캐릭터 변경" << endl;                             //사용 설명서 메세지 출력
	cout << "v: 캐릭터 얼굴에 삽입 on/off" << endl;               //사용 설명서 메세지 출력
	cout << "b: 배경 제거 on/off" << endl;                        //사용 설명서 메세지 출력
	cout << "****************************************" << endl;   //사용 설명서 메세지 출력

	Mat face,frame; // Mat 객체 생성 (face=부분영역추출에 사용,frame= 카메라 로부터 영상을 입력받을때 사용)
	int key,chr_emotion, chracter_number=1;  //키보드입력을 저장할 변수, 감정을 저장할 변수 선언
	bool remove_onoff = false; // 배경제거를 사용할건지 판단하는 bool타입 변수 선언

	while (true) { //무한 반복
		cap >> frame; //카메라로부터 영상을 입력받고
		if (frame.empty()) break; //frame이 비어있으면 종료
		cout << frame.cols << endl;
		cout << frame.rows << endl;
		if(remove_onoff)frame=Remove_Background(frame); //remove_onoff가 true이면 배경제거
	
		Mat blob = blobFromImage(frame, 1, Size(300, 300), Scalar(104, 177, 123)); //blobFromImage 1x1x300x300으로 변환
		net.setInput(blob); //net 객체안에 넣기
	
		Mat res = net.forward(); //추론
		
		Mat detect(res.size[2], res.size[3], CV_32FC1, res.ptr<float>()); //Mat 객체 생성 
		cout << res.ptr<float>();
		for (int i = 0; i < detect.rows; i++) {  // 얼굴검출
			float confidence = detect.at<float>(i, 2);
			if (confidence < 0.5) break; //가중치가 0.5 이하이면 제외
			int x1 = cvRound(detect.at<float>(i, 3) * frame.cols);  //왼쪽위 x좌표
			int y1 = cvRound(detect.at<float>(i, 4) * frame.rows);  //왼쪽위 y좌표
			int x2 = cvRound(detect.at<float>(i, 5) * frame.cols);  //오른쪽 아래 x좌표
			int y2 = cvRound(detect.at<float>(i, 6) * frame.rows);  //오른쪽 아래  y좌표


			rectangle(frame, Rect(Point(x1, y1), Point(x2, y2)), Scalar(0, 255, 0)); //도형그리기(사각형)
			face = frame(Rect(Point(x1, y1), Point(x2, y2))); // 부분영역 추출

	
			chr_emotion = Emotion_inference(face); //감정 추론 함수. 몇번째 감정인지 리턴
			
			
			
			Mat dst=face_chracter(face, chr_emotion, chracter_number); //사이즈조정된 이미지 출력 
			if(key_v) //v 키가 눌려서 true 가되면
			dst.copyTo(face); // 행렬을 복사

			String label = format("Face: %4.3f", confidence); //label 이름 문자열 
			putText(frame, label, Point(x1, y1 - 1), FONT_HERSHEY_SIMPLEX, 0.8, //  이미지에 문자열 출력
				Scalar(0, 255, 0));
		}
		imshow("frame", frame); //이미지 호출
		key = waitKey(1); //1msec 대기하고 key값을 받아오기
		if (key == 27) break;  //esc버튼이 눌리면 종료
		else if (key == 'c') {  //c버튼이 눌리면
			chracter_number++;  //캐릭터 넘버 증가
			if (chracter_number == 5) //5번째 캐릭터는 없기때문에 5번째 캐릭터가 되면
				chracter_number = 1;  //첫번째 캐릭터로
		}else if (key == 'q') {  //q버튼이 눌리면
			if (key_q ==false) key_q = true; //key_q가 false면 true로 바꾸기
			else if (key_q == true) { key_q = false; 	destroyWindow("Emotion"); } //key_q가 true가 되면 false로 바꾸고 윈도우 닫기
		}
		else if (key == 'b') { //b버튼이 눌리면
			if (remove_onoff == false)remove_onoff = true; //remove_onoff 가 false 이면 true 로 바꾸기
			else if (remove_onoff == true) remove_onoff = false;  //remove_onoff가 true이면 false로 바꾸기
		}
		else if (key == 'v') {  //v 버튼이 눌리면
			if (key_v == false)key_v = true; //key_v 가 false면 true 로바꾸기
			else if (key_v == true) key_v = false; //아니면 반대로 바꾸기
		}
	}
	return 0; //메인함수 종료
}

int Emotion_inference(Mat img) { //감정을 추론하는함수

	vector<String> classNames = {"happy","sad", "suprise","angry" };  //vector sting 객체 생성
	Net net = readNet("frozen_model.pb"); //net객체 생성

	
	Mat inputBlob = blobFromImage(img, 1.0 / 127.5, Size(224, 224), Scalar(-1, -1, -1)); //1x1x224x224 로 변환
	net.setInput(inputBlob); //net에 변환된 mat객체 넣기
	Mat prob = net.forward(); //추론

	double maxVal; // 가중치를 저장할 변수
	Point maxLoc; //몇번째 인지를 알아내기위해 선언하는 point 변수 
	minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc); //최대값 추출 알고리즘 
	String str = classNames[maxLoc.x] + format("(%4.2lf%%)", maxVal * 100) ; //string 변수 선언
	putText(img, str, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255)); // 이미지에 문자열 넣기

	if (key_q) {  //key_q값이 true 이면
		imshow("Emotion", img); //이미지 호출
	}

	return maxLoc.x; //몇번째 감정인지 리턴
}

Mat face_chracter(Mat img, int emotion ,int chracter_number) { //얼굴 표정 캐릭터 사이즈 조정후 리턴하는 함수
	
	Mat dst; //결과를 저장할 Mat 객체 생성

	if (emotion == 0) { // 인수로 0 이 들어오면
		string filename = format("기쁜%d.PNG", chracter_number); //format 함수 사용 불러올 이미지 문자열 결정
		Mat ch = imread(filename, IMREAD_COLOR); // Mat 객체 생성 이미지 호출
		resize(ch, dst, Size(img.cols, img.rows)); // 어파인 변환 인수로 들어온 img의 사이즈만큼 크기 변환
	}
	else if (emotion == 1){ // 인수로 1 이 들어오면
		string filename = format("슬픈%d.PNG", chracter_number);//format 함수 사용 불러올 이미지 문자열 결정
		Mat ch = imread(filename, IMREAD_COLOR); // Mat 객체 생성 이미지 호출
		resize(ch, dst, Size(img.cols, img.rows)); // 어파인 변환 인수로 들어온 img의 사이즈만큼 크기 변환
 	}
	else if (emotion == 2) { // 인수로 2 이 들어오면//format 함수 사용 불러올 이미지 문자열 결정
		string filename = format("놀란%d.PNG", chracter_number);
		Mat ch = imread(filename, IMREAD_COLOR);// Mat 객체 생성 이미지 호출
		resize(ch, dst, Size(img.cols, img.rows)); // 어파인 변환 인수로 들어온 img의 사이즈만큼 크기 변환
	}
	else if (emotion == 3) {// 인수로 3 이 들어오면
		string filename = format("화난%d.PNG", chracter_number);//format 함수 사용 불러올 이미지 문자열 결정
		Mat ch = imread(filename, IMREAD_COLOR);// Mat 객체 생성 이미지 호출
		resize(ch, dst, Size(img.cols, img.rows));// 어파인 변환 인수로 들어온 img의 사이즈만큼 크기 변환
	}

	return dst; //결과 Mat 객체 리턴
}

Mat Remove_Background(Mat img) { //배경 제거 함수

	Rect rectangle(100, 100, 500, 500);  //Rect 객체 생성 좌측상단점 x,y   가로세로 w,h  직사각형 밖의 화소는 배경으로 레이블링

	Mat result;//마스크 행렬 Mat 객체 생성
	Mat bgModel, fgModel; // Mat 객체 생성

	grabCut(img,    result,  rectangle,  bgModel, fgModel, 1, GC_INIT_WITH_RECT); // grabcut 함수사용 ,직사각형 사용
	 // cv::CC_INT_WITH_RECT 플래그를 이용한 경계 직사각형 모드를 사용하도록 지정
	
	compare(result, GC_PR_FGD, result, CMP_EQ);// 전경일 가능성이 있는 화소를 마크한 것을 가져오기
	//cv::GC_PR_FGD 전경에 속할 수도 있는 화소
	//CMP_EQ 면 같은 화소가 같은지 비교

	Mat foreground(img.size(), CV_8UC3, Scalar(255, 255, 255)); // 결과 영상 생성
	img.copyTo(foreground, result); //복사본이 저장될 행렬, 마스트 행렬(0)이 아닌 자리에 행렬 복사

	return foreground; //배경이 제거된 이미지 리턴
}