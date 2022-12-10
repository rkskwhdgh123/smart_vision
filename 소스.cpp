#include <opencv2/opencv.hpp> //������� ����
#include <iostream> //������� ����
using namespace cv; //cv ���
using namespace cv::dnn; //cv�ؿ� dnn ���
using namespace std; //std ���
Mat Remove_Background(Mat img); //��� ����� �Լ� ����
int Emotion_inference(Mat img); //������ �߷��ϴ� �Լ� ����
Mat face_chracter(Mat img,int emotion, int chracter_number); //�󱼿� ĳ���� �̹����� �����ִ� �Լ� ����
const String model = "res10_300x300_ssd_iter_140000_fp16.caffemodel"; //Net�� ����ϴ� ���� �̸�
const String config = "deploy.prototxt"; //Net�� ����ϴ� ���� �̸�
bool key_q = false;  // Ű���� q�� �������� �Ǵ��ϴ� ���� �������� ���� 
bool key_v = true;  // Ű���� v�� �������� �Ǵ��ϴ� ���� �������� ����

int main(void) //�����Լ� ������
{


	VideoCapture cap(0); //VideoCapture��ü ���� 0�� ��Ʈ�� ī�޶� ���
	if (!cap.isOpened()) { cerr << "Camera open failed!" << endl; return -1; } //ī�޶� ������ ������ ����
	Net net = readNet(model, config); //Net ����
	if (net.empty()) { cerr << "Net open failed!" << endl; return -1; } //net�� ������ ������ ����

	cout << "****************����******************" << endl;   //��� ���� �޼��� ���
	cout << "esc: ����" << endl;                                  //��� ���� �޼��� ���
	cout << "q: ���� �� ���� on/off" << endl;                   //��� ���� �޼��� ���
	cout << "c: ĳ���� ����" << endl;                             //��� ���� �޼��� ���
	cout << "v: ĳ���� �󱼿� ���� on/off" << endl;               //��� ���� �޼��� ���
	cout << "b: ��� ���� on/off" << endl;                        //��� ���� �޼��� ���
	cout << "****************************************" << endl;   //��� ���� �޼��� ���

	Mat face,frame; // Mat ��ü ���� (face=�κп������⿡ ���,frame= ī�޶� �κ��� ������ �Է¹����� ���)
	int key,chr_emotion, chracter_number=1;  //Ű�����Է��� ������ ����, ������ ������ ���� ����
	bool remove_onoff = false; // ������Ÿ� ����Ұ��� �Ǵ��ϴ� boolŸ�� ���� ����

	while (true) { //���� �ݺ�
		cap >> frame; //ī�޶�κ��� ������ �Է¹ް�
		if (frame.empty()) break; //frame�� ��������� ����
		cout << frame.cols << endl;
		cout << frame.rows << endl;
		if(remove_onoff)frame=Remove_Background(frame); //remove_onoff�� true�̸� �������
	
		Mat blob = blobFromImage(frame, 1, Size(300, 300), Scalar(104, 177, 123)); //blobFromImage 1x1x300x300���� ��ȯ
		net.setInput(blob); //net ��ü�ȿ� �ֱ�
	
		Mat res = net.forward(); //�߷�
		
		Mat detect(res.size[2], res.size[3], CV_32FC1, res.ptr<float>()); //Mat ��ü ���� 
		cout << res.ptr<float>();
		for (int i = 0; i < detect.rows; i++) {  // �󱼰���
			float confidence = detect.at<float>(i, 2);
			if (confidence < 0.5) break; //����ġ�� 0.5 �����̸� ����
			int x1 = cvRound(detect.at<float>(i, 3) * frame.cols);  //������ x��ǥ
			int y1 = cvRound(detect.at<float>(i, 4) * frame.rows);  //������ y��ǥ
			int x2 = cvRound(detect.at<float>(i, 5) * frame.cols);  //������ �Ʒ� x��ǥ
			int y2 = cvRound(detect.at<float>(i, 6) * frame.rows);  //������ �Ʒ�  y��ǥ


			rectangle(frame, Rect(Point(x1, y1), Point(x2, y2)), Scalar(0, 255, 0)); //�����׸���(�簢��)
			face = frame(Rect(Point(x1, y1), Point(x2, y2))); // �κп��� ����

	
			chr_emotion = Emotion_inference(face); //���� �߷� �Լ�. ���° �������� ����
			
			
			
			Mat dst=face_chracter(face, chr_emotion, chracter_number); //������������ �̹��� ��� 
			if(key_v) //v Ű�� ������ true ���Ǹ�
			dst.copyTo(face); // ����� ����

			String label = format("Face: %4.3f", confidence); //label �̸� ���ڿ� 
			putText(frame, label, Point(x1, y1 - 1), FONT_HERSHEY_SIMPLEX, 0.8, //  �̹����� ���ڿ� ���
				Scalar(0, 255, 0));
		}
		imshow("frame", frame); //�̹��� ȣ��
		key = waitKey(1); //1msec ����ϰ� key���� �޾ƿ���
		if (key == 27) break;  //esc��ư�� ������ ����
		else if (key == 'c') {  //c��ư�� ������
			chracter_number++;  //ĳ���� �ѹ� ����
			if (chracter_number == 5) //5��° ĳ���ʹ� ���⶧���� 5��° ĳ���Ͱ� �Ǹ�
				chracter_number = 1;  //ù��° ĳ���ͷ�
		}else if (key == 'q') {  //q��ư�� ������
			if (key_q ==false) key_q = true; //key_q�� false�� true�� �ٲٱ�
			else if (key_q == true) { key_q = false; 	destroyWindow("Emotion"); } //key_q�� true�� �Ǹ� false�� �ٲٰ� ������ �ݱ�
		}
		else if (key == 'b') { //b��ư�� ������
			if (remove_onoff == false)remove_onoff = true; //remove_onoff �� false �̸� true �� �ٲٱ�
			else if (remove_onoff == true) remove_onoff = false;  //remove_onoff�� true�̸� false�� �ٲٱ�
		}
		else if (key == 'v') {  //v ��ư�� ������
			if (key_v == false)key_v = true; //key_v �� false�� true �ιٲٱ�
			else if (key_v == true) key_v = false; //�ƴϸ� �ݴ�� �ٲٱ�
		}
	}
	return 0; //�����Լ� ����
}

int Emotion_inference(Mat img) { //������ �߷��ϴ��Լ�

	vector<String> classNames = {"happy","sad", "suprise","angry" };  //vector sting ��ü ����
	Net net = readNet("frozen_model.pb"); //net��ü ����

	
	Mat inputBlob = blobFromImage(img, 1.0 / 127.5, Size(224, 224), Scalar(-1, -1, -1)); //1x1x224x224 �� ��ȯ
	net.setInput(inputBlob); //net�� ��ȯ�� mat��ü �ֱ�
	Mat prob = net.forward(); //�߷�

	double maxVal; // ����ġ�� ������ ����
	Point maxLoc; //���° ������ �˾Ƴ������� �����ϴ� point ���� 
	minMaxLoc(prob, NULL, &maxVal, NULL, &maxLoc); //�ִ밪 ���� �˰��� 
	String str = classNames[maxLoc.x] + format("(%4.2lf%%)", maxVal * 100) ; //string ���� ����
	putText(img, str, Point(10, 30), FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255)); // �̹����� ���ڿ� �ֱ�

	if (key_q) {  //key_q���� true �̸�
		imshow("Emotion", img); //�̹��� ȣ��
	}

	return maxLoc.x; //���° �������� ����
}

Mat face_chracter(Mat img, int emotion ,int chracter_number) { //�� ǥ�� ĳ���� ������ ������ �����ϴ� �Լ�
	
	Mat dst; //����� ������ Mat ��ü ����

	if (emotion == 0) { // �μ��� 0 �� ������
		string filename = format("���%d.PNG", chracter_number); //format �Լ� ��� �ҷ��� �̹��� ���ڿ� ����
		Mat ch = imread(filename, IMREAD_COLOR); // Mat ��ü ���� �̹��� ȣ��
		resize(ch, dst, Size(img.cols, img.rows)); // ������ ��ȯ �μ��� ���� img�� �����ŭ ũ�� ��ȯ
	}
	else if (emotion == 1){ // �μ��� 1 �� ������
		string filename = format("����%d.PNG", chracter_number);//format �Լ� ��� �ҷ��� �̹��� ���ڿ� ����
		Mat ch = imread(filename, IMREAD_COLOR); // Mat ��ü ���� �̹��� ȣ��
		resize(ch, dst, Size(img.cols, img.rows)); // ������ ��ȯ �μ��� ���� img�� �����ŭ ũ�� ��ȯ
 	}
	else if (emotion == 2) { // �μ��� 2 �� ������//format �Լ� ��� �ҷ��� �̹��� ���ڿ� ����
		string filename = format("���%d.PNG", chracter_number);
		Mat ch = imread(filename, IMREAD_COLOR);// Mat ��ü ���� �̹��� ȣ��
		resize(ch, dst, Size(img.cols, img.rows)); // ������ ��ȯ �μ��� ���� img�� �����ŭ ũ�� ��ȯ
	}
	else if (emotion == 3) {// �μ��� 3 �� ������
		string filename = format("ȭ��%d.PNG", chracter_number);//format �Լ� ��� �ҷ��� �̹��� ���ڿ� ����
		Mat ch = imread(filename, IMREAD_COLOR);// Mat ��ü ���� �̹��� ȣ��
		resize(ch, dst, Size(img.cols, img.rows));// ������ ��ȯ �μ��� ���� img�� �����ŭ ũ�� ��ȯ
	}

	return dst; //��� Mat ��ü ����
}

Mat Remove_Background(Mat img) { //��� ���� �Լ�

	Rect rectangle(100, 100, 500, 500);  //Rect ��ü ���� ��������� x,y   ���μ��� w,h  ���簢�� ���� ȭ�Ҵ� ������� ���̺�

	Mat result;//����ũ ��� Mat ��ü ����
	Mat bgModel, fgModel; // Mat ��ü ����

	grabCut(img,    result,  rectangle,  bgModel, fgModel, 1, GC_INIT_WITH_RECT); // grabcut �Լ���� ,���簢�� ���
	 // cv::CC_INT_WITH_RECT �÷��׸� �̿��� ��� ���簢�� ��带 ����ϵ��� ����
	
	compare(result, GC_PR_FGD, result, CMP_EQ);// ������ ���ɼ��� �ִ� ȭ�Ҹ� ��ũ�� ���� ��������
	//cv::GC_PR_FGD ���濡 ���� ���� �ִ� ȭ��
	//CMP_EQ �� ���� ȭ�Ұ� ������ ��

	Mat foreground(img.size(), CV_8UC3, Scalar(255, 255, 255)); // ��� ���� ����
	img.copyTo(foreground, result); //���纻�� ����� ���, ����Ʈ ���(0)�� �ƴ� �ڸ��� ��� ����

	return foreground; //����� ���ŵ� �̹��� ����
}