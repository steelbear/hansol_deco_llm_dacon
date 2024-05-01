# 도배 하자 질의 응답 처리 : 한솔데코 시즌2 AI 경진대회
- 대회 링크: [link](https://dacon.io/competitions/official/236216/overview/description)
- 주제: 도배 하자 질의 응답 AI 모델 개발
- 설명: 도배 하자 도메인에 대한 질의를 바탕으로 지능적인 응답을 생성하는 AI 모델 개발
- 기간: 2024.01.29 ~ 2024.03.11

# 평가 방식
- 평가 산식: Cosine Similarity
    - 'distiluse-base-multilingual-cased-v1'을 통해 생성된 답변을 벡터로 임베딩 후 정답과 유사도 계산
- 평가 기준
    - Public score: 전체 테스트 데이터 중 사전 샘플링된 40%
    - Private score: 전체 테스트 데이터 100%

# 결과
|사용 모델|적용한 방법|Public score|Private score|
|--------|----------|------------|-------------|
|beomi/OPEN-SOLAR-KO-10.7B|Agumentation + RAG|0.768472|0.752784|
|beomi/OPEN-SOLAR-KO-10.7B|Agumentation + GK|0.760639|0.743935|

# 사용 모델
- [beomi/OPEN-SOLAR-KO-10.7B](https://huggingface.co/beomi/OPEN-SOLAR-KO-10.7B)