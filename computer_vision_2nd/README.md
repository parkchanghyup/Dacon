# 제 2회 컴퓨터 비전 학습 경진대회 
[링크](https://dacon.io/competitions/official/235697/overview/description/)

- 228팀 중 9등을 차지하여 수상.
- 이 폴더는 위 링크의 컴퓨터 비전 경진대회 코드 입니다.
- date 폴더내 있는 파일 또한 위 링크에서 다운 받으실 수 있습니다.
- 모델은 `effnet`,`mobileNet`,`ResNet`총 3개의 모델을 이용 해보았고, 성는이 가장 좋은 Effcient Net을 최종 모델로 사용하였습니다.

## 전처리

- 데이터 증강 기법으로 수평 회전, 수직 회전, 랜덤 회전 총 3개의 기법 적용
  
## 모델링
- optimizer : Adam
- lr_scheduler : StepLR(optimizer,step_size = 5,gamma = 0.9)
- loss function : BCELoss
- epochs = 30
- batch_size = 16
- 교차 검증(k = 3)
- Efficient net - b7 사용.                  



## reference
---
- [Efficient net](https://everyday-deeplearning.tistory.com/entry/%EC%B4%88%EA%B0%84%EB%8B%A8-%EB%85%BC%EB%AC%B8%EB%A6%AC%EB%B7%B0-Efficient-NetworkGoogle-Research-Brain-Team)
- [mobilenet](https://ariz1623.tistory.com/300)
- [ResNet](https://github.com/signatrix/regnet)
