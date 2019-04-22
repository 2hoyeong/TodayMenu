""" csv 데이터를 원하는 형태로 변환하기 위한 프로그램 """
# [한식, 일식, 중식, 양식, 패스트푸드, 피자, 치킨, 분식, 술집, 카페]
# [한식/육류, 한식/생선, 한식/면류]

import pandas
import numpy as np
import gensim

model = gensim.models.Word2Vec.load('/data/ko/ko.bin')

한식          = 0
일식          = 1
중식          = 2
양식          = 3
패스트푸드    = 4
피자          = 5
치킨          = 6
분식          = 7
술집          = 8
카페          = 9


df = pandas.read_csv('/data/restaurant.csv', engine='python')

matrix = df.values # dataframe to numpy
result = [[np.nan, np.nan, np.nan]] # 작업의 편의를 위해 배열의 모양을 잡아준다.

# 조건에 맞는 데이터를 찾아서 result에 추가하는 작업
for line in matrix:
    if line[0] is np.NaN or line[1] is np.NaN:
        continue
    #if line[3] is np.nan and line[4] is np.nan:
    if line[3] is np.nan:
        continue

    #line[3] = str(line[3])
    if line[3] == "양식&외국식" or line[3] == "레스토랑":
        line[3] = "양식"
    elif line[3] == "술한잔하기 좋은 집" or line[3] == "정종/대포집/소주방":
        line[3] = "술집"
    elif (
         line[3] == "별식" or line[3] == "차" or line[3] == "기타" or line[3] == "기타 휴게음식점" or 
         line[3] == "경양식" or line[3] == "탕류(보신용)" or line[3] == "복어취급" or line[3] == "복어" or # 기타,경양식 보류, 탕류(보신용)은 분류가 제대로 되어있지 않음
         line[3] == "출장조리" or line[3] == "능이버섯"
         ) :
        continue
    elif (  # 회집
         line[3] == "동인동 찜갈비" or line[3] == "뷔페식" or line[3] == "감자탕" or line[3] == "냉면" or 
         line[3] == "회집" or line[3] == "횟집" or line[3] == "식육(숯불구이)" or line[3] == "식육취급" or 
         line[3] == "생선회" or line[3] == "재첩" or line[3] == "고기구이" or line[3] == "매운탕" or 
         line[3] == "묵밥.태평초" or line[3] == "산채음식" or line[3] == "영주삼계탕" or line[3] == "영주순대" or 
         line[3] == "오리" or line[3] == "오삼불고기" or line[3] == "영주청국장" or line[3] == "영주한우숯불" or 
         line[3] == "한정식" or line[3] == "약선전문" or line[3] == "연근정식" or line[3] == "한방오리불고기" or 
         line[3] == "생오리구이" or line[3] == "곤드레밥" or line[3] == "탕류" or line[3] == "소고기" or 
         line[3] == "닭/오리" or line[3] == "해물" or line[3] == "돼지고기" or line[3] == "한우/돼지고기" or 
         line[3] == "버섯찌개" or line[3] == "올갱이국 및 해장국" or line[3] == "두부류" or line[3] == "매운탕 및 해물류" or
         line[3] == "냉면/막국수/칼국수" or line[3] == "영양탕" or line[3] == "차밭골정식" or line[3] == "뷔페" or
         line[3] == "인삼갈비탕" or line[3] == "오리바베큐"
         ) :
        line[3] = "한식"
    elif line[3] == "중국식":
        line[3] = "중식"
    elif line[3] == "김밥(도시락)" or line[3] == "쫄면":
        line[3] = "분식"
    elif line[3] == "통닭(치킨)" or line[3] == "호프/통닭" or line[3] == "모듬똥집":
        line[3] = "치킨"
    
    if line[3] == "한식":
        line[4] = 한식
    elif line[3] == "일식":
        line[4] = 일식
    elif line[3] == "중식":
        line[4] = 중식
    elif line[3] == "양식":
        line[4] = 양식
    elif line[3] == "패스트푸드":
        line[4] = 패스트푸드
    elif line[3] == "피자":
        line[4] = 피자
    elif line[3] == "치킨":
        line[4] = 치킨
    elif line[3] == "분식":
        line[4] = 분식
    elif line[3] == "술집":
        line[4] = 술집
    elif line[3] == "카페":
        line[4] = 카페

    result = np.vstack((result, [line[0], line[3], line[4]]))

result = result[1:] # 배열의 모양을 잡기위한 [NaN, NaN, NaN]을 제거 한다.
df = pandas.DataFrame(result)
df.columns = ['NAME', 'C1', 'C2']
u = df.C1.unique()
print(df)


#print(df.C1.unique())
#print(df.C2.unique())

