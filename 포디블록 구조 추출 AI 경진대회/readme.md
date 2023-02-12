# 포디블록 구조 추출 AI 경진대회
- [대회링크](https://dacon.io/competitions/official/236046/overview/description)
- 2D 이미지 내 포디블록의 10가지의 블록 패턴들의 존재 여부 분류
- 대회 기간 : 23.01.02 ~ 23.01.30
- 35등 / 461팀(상위 7%), 최종 모델 정확도 : 0.94236


## 전처리 
- random augment
- rembg라이브러리를 통한 배경제거
- 배경 이미지 합성
- SAM optimzer(모델 성능이 악화되어 최종 모델에 적용 x)

## 모델링

- optimizer : Adam, AdamW
- LR : 0.0005
- lr_scheduler : StepLR(optimizer,step_size = 5,gamma = 0.9)
- epochs : 200
- 교차 검증 k = 10
- `Efficient net V2 - M`, `Efficient net V2 - L`, `dense net 201` 

## 실험 스코어
|모델|voting|전처리|교차 검증|정확도|
|---|---|---|---|---|
|Efficient b7|x|random augment|x|0.8650|
|Efficient b7|x|random augment, 배경 제거|x|0.8668|
|Efficient b7|x|random augment, 배경 제거, SAM optimizer|x|0.8333|
|Efficient V2 - M|x|random augment, 배경 제거|x|0.9196|
|Efficient V2 - M|x|random augment, 배경 제거|x|0.9196|
|Efficient V2 - M|x|random augment, 배경 합성|x|0.9360|
|Densenet|X|random augment, 배경 합성|x|0.9342|
|Efficient V2 - M,L|hard voting|random augment, 배경 합성|x|0.9363|
|Efficient V2 - M,L|soft voting|random augment, 배경 합성|x|0.9365|
|Efficient V2 - M,L, densenet|hard voting|random augment, 배경 합성|k = 10|0.9465|
|Efficient V2 - M,L, densenet|soft voting|random augment, 배경 합성|k = 10|0.9479|



