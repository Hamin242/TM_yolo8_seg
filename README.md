# TM_yolo8_seg
TM 누유 검사를 위한 seg 모델 

데이터는 해당 디렉토리에 다음과 같이 구성 
data
  images
    train
    test
    val
  json
    train
    test
    val
  labels
    train
    test
    val
  video
결과는 다음과 같이 구성 
runs
  segment>weights>train_seg(val,test포함)
