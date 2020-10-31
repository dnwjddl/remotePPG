# rPPG
🌱피부색의 미세한 변화를 통해 사람의 심박수 측정
* Webcam에서도 실행가능
---------------------------------------
**baseline코드**
1. 얼굴 검출(ROI)
haarcascade 사용
2. RGB -> YCbCr 변경 후 신호가 가장 강한 green 컬러 추출 후 평균값 내기
3. detrend 를 통하여 trend를 많이 받는 신호 제거
4. band-pass filtering 을 통한 노이즈 제거
5. DFT를 이용하여 심박수 측정을 위한 주파수 변환
6. 가장 높은 값에 대한 심박수 측정

#### 결과 그래프 및 값
![최종결과](https://user-images.githubusercontent.com/72767245/97080996-6bf40300-163a-11eb-8cdc-9e86072a9c4d.png)
