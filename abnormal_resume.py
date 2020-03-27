import re
import time

from nltk.tokenize import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from collections import defaultdict
from tqdm import tqdm

def Abnormal_Resume_Main(RESUME):
    
    try:
        MIN_LENGTH = 50 # 하이퍼 파라미터 : 길이가 30자 미만인 자소서 검출
        RATIO_LIMIT = 0.05 # 하이퍼 파라미터 : 전체 글자수에서 해당 단어의 비율 [RATIO_LIMIT] 이상이면 검출

        abnormal_result = dict()
        corpus_len = len(RESUME) # 자기소개서 길이

        # 0. null값 체크
        if (RESUME == None) or (RESUME == '.') or (RESUME == ' ') or (RESUME == ''):
            abnormal_result['0'] = "정상적인 자기소개서가 입력되지 않았습니다."

        # 1. 길이 체크
        if len(RESUME) < MIN_LENGTH:
            abnormal_result['1'] = "자기소개서의 길이가 짧습니다."

        abnormal_word_li = []

        # 2. 한글 : 자음/모음만 있는 단어 / 영어 및 숫자 : 전체에서 너무 많은 부분을 차지한다면 검출 ex) asdfafas
        han = re.compile('[ㄱ-ㅎ]+')  # 자음이 1회 이상 반복되는 걸 찾는다. ex) ㅇㅇ
        abnormal_word_han = han.findall(RESUME)

        han_mo = re.compile('[ㅏ-ㅣ]+')  # 모음이 1회 이상 반복되는 걸 찾는다. ex) ㅇㅇ
        abnormal_word_han_mo = han_mo.findall(RESUME)

        eng = re.compile('[a-zA-Z]+') # 영어 단어 모두 찾는다.
        abnormal_word_eng = eng.findall(RESUME)

        num = re.compile('[0-9]+') # 숫자 단어 모두 찾는다.
        abnormal_word_num = num.findall(RESUME)

        spe_char = re.compile('[^ㄱ-ㅣ가-힣|a-zA-Z|0-9|.]+') # 특수문자 모두 찾는다.
        abnormal_word_spe_char = spe_char.findall(RESUME)


        if (abnormal_word_han != []) or (abnormal_word_eng != []) or (abnormal_word_num != []) or (abnormal_word_spe_char != []) or (abnormal_word_han_mo != []):

            abnormal_word = set(abnormal_word_han + abnormal_word_eng + abnormal_word_num + abnormal_word_spe_char + abnormal_word_han_mo) # list 합치고 중복제거

            for word in abnormal_word:

                abnormal_ratio = round(len(word) / corpus_len, 3)  # abnormal_ratio = 이상 글자수 / 전체 글자수

                if (abnormal_ratio >= RATIO_LIMIT): # 일부 쓰는 경우 정상 자소서로 판단 ex) ㅇㅈ은 저의 무기
                    abnormal_word_li.append([word, abnormal_ratio])

        if len(abnormal_word_li) > 0:
            abnormal_result['2'] = abnormal_word_li

        # 3-1. TF-IDF 학습 전 전처리
        tfidf_word_li = []
        
        RESUME = re.sub("(&)[&|a-zA-Z]+(;)", " ", RESUME)
        RESUME = re.sub("&#[0-9]+;", " ", RESUME)
        RESUME = re.sub("[^ㄱ-ㅣ가-힣|a-zA-Z|0-9|]+", " ", RESUME)
        corpus = RESUME

        RESUME_SENT_TOK = sent_tokenize(RESUME)

        # 3-2. TF-IDF Fitting
        tfidf_criteria_result = Abnormal_Resume_Tfidf(RESUME_SENT_TOK)

        if (tfidf_criteria_result != []): # TF-IDF 기준 점수를 초과하는 단어가 있는지 확인

            for word, abnormal_ratio in tfidf_criteria_result:

                # tf-idf로 추출된 반복 의심 단어가 해당 자소서에서 얼마나 차지하는지 계산한다.
                tfidf_word = re.compile(word)
                tfidf_word_all = tfidf_word.findall(corpus)
                tfidf_word_leng = sum([len(word) for word in tfidf_word_all])
                repeated_ratio = round(tfidf_word_leng / corpus_len, 3)

                if (repeated_ratio >= RATIO_LIMIT): # 의심단어가 자소서 전체에서 비율이 얼마 없다면 정상으로 판단
                    tfidf_word_li.append([word, abnormal_ratio])
        
        if len(tfidf_word_li) > 0:
            abnormal_result['3'] = tfidf_word_li
                    
    except Exception as e:
        pass
    
    return abnormal_result


def Abnormal_Resume_Tfidf(RESUME):
       
    # TF-IDF
    try:
        corpus = RESUME
        # 하이퍼 파라미터로 아래 점수 이상인 단어만 반환한다.
        tfidf_criteria_score = 0.2
        
        vectorizer = TfidfVectorizer()
        sp_matrix = vectorizer.fit_transform(corpus)

        word2id = defaultdict(lambda : 0) # idx 값 

        for idx, feature in enumerate(vectorizer.get_feature_names()):
            word2id[feature] = idx

        for i, sent in enumerate(corpus):
             tfidf_result = [(token, sp_matrix[i, word2id[token]]) for token in sent.split()]

        tfidf_result = set(tfidf_result) # 중복제거

        tfidf_criteria_result = list() # 기준점수 이상인 단어와 tf-idf score를 담는다.

        for word, tfidf_score in tfidf_result:

            if tfidf_score > tfidf_criteria_score:
                tfidf_criteria_result.append([word, tfidf_score])

    except ValueError:
        
        tfidf_criteria_result = []
        return tfidf_criteria_result
    
    return tfidf_criteria_result
