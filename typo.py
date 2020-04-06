import sys
import pymysql
import re
import pandas as pd
from pandas import DataFrame
import gensim
import sqlite3 as sq
import pandas.io.sql as pd_sql
import re, collections
from pandas import DataFrame
from gensim.models import Word2Vec
import matplotlib.pyplot as plt
from gensim.test.utils import datapath
from gensim.test.utils import common_texts
from gensim.models import Phrases
from gensim.models import KeyedVectors
from datetime import datetime


kor_begin = 44032
kor_end = 55203
chosung_base = 588
jungsung_base = 28
jaum_begin = 12593
jaum_end = 12622
moum_begin = 12623
moum_end = 12643




chosung_list = [ 'ㄱ', 'ㄲ', 'ㄴ', 'ㄷ', 'ㄸ', 'ㄹ', 'ㅁ', 'ㅂ', 'ㅃ',
        'ㅅ', 'ㅆ', 'ㅇ' , 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jungsung_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ',
        'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ', 'ㅙ', 'ㅚ',
        'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ',
        'ㅡ', 'ㅢ', 'ㅣ']


jongsung_list = [
    ' ', 'ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ',
        'ㄹ', 'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ',
        'ㅁ', 'ㅂ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅊ',
        'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

jaum_list = ['ㄱ', 'ㄲ', 'ㄳ', 'ㄴ', 'ㄵ', 'ㄶ', 'ㄷ', 'ㄸ', 'ㄹ',
              'ㄺ', 'ㄻ', 'ㄼ', 'ㄽ', 'ㄾ', 'ㄿ', 'ㅀ', 'ㅁ', 'ㅂ',
              'ㅃ', 'ㅄ', 'ㅅ', 'ㅆ', 'ㅇ', 'ㅈ', 'ㅉ', 'ㅊ', 'ㅋ', 'ㅌ', 'ㅍ', 'ㅎ']

moum_list = ['ㅏ', 'ㅐ', 'ㅑ', 'ㅒ', 'ㅓ', 'ㅔ', 'ㅕ', 'ㅖ', 'ㅗ', 'ㅘ',
              'ㅙ', 'ㅚ', 'ㅛ', 'ㅜ', 'ㅝ', 'ㅞ', 'ㅟ', 'ㅠ', 'ㅡ', 'ㅢ', 'ㅣ']


def compose(chosung, jungsung, jongsung):
    char = chr(
        kor_begin +
        chosung_base * chosung_list.index(chosung) +
        jungsung_base * jungsung_list.index(jungsung) +
        jongsung_list.index(jongsung)
    )
    return char

def decompose(c):
    if not character_is_korean(c):
        return None
    i = ord(c)
    if (jaum_begin <= i <= jaum_end):
        return (c, ' ', ' ')
    if (moum_begin <= i <= moum_end):
        return (' ', c, ' ')

    # decomposition rule
    i -= kor_begin
    cho  = i // chosung_base
    jung = ( i - cho * chosung_base ) // jungsung_base
    jong = ( i - cho * chosung_base - jung * jungsung_base )
    return (chosung_list[cho], jungsung_list[jung], jongsung_list[jong])


def character_is_korean(c):
    i = ord(c)
    return ((kor_begin <= i <= kor_end) or
            (jaum_begin <= i <= jaum_end) or
            (moum_begin <= i <= moum_end))




'''
def words(text) : return re.findall('[a-z]+',text.lower())

def train(features):
    model = collections.defaultdict(lambda: 1)
    for f in features:
        model[f] += 1
    return model

NWORDS = train(words(file('big.txt').read()))

alphabet = ' abcdefghijklmnopqrstuvwxyz'

def edits1(word):
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = [a + b[1:] for a,b in splits if b]
    transposes = [a + b[1] + b[0] + b[2:] for a,b in splits if len(b)>1]
    replaces = [a + c + b[1:] for a,b in splits for c in alphabet if b]
    inserts = [a + c + b  for a, b in splits for c in alphabet]
    return set(deletes + transposes + replaces + inserts)

def known_edits2(word):
    return set(e2 for e1 in edits1(word) for e2 in edits1(e1) if e2 in NWORDS)

def known(words): return set(w for w in words if w in NWORDS)

def correct(word):
    candidates = known([word]) or known(edits1(word)) or known_edits2(word) or [word]
    return max(candidates, key=NWORDS.get)
'''


def levenshtein(s1, s2, debug=False):
    if len(s1) <len(s2):
        return levenshtein(s2, s1, debug)

    if(len(s2) == 0):
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i+1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j+1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug :
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]



def levenshtein1(s1, s2, cost=None, debug=False):
    if len(s1) <len(s2):
        return levenshtein(s2, s1, debug=debug)

    if(len(s2) == 0):
        return len(s1)

    if cost is None:
        cost = {}

    #changed
    def substitution_cost(c1, c2):
        if c1 == c2:
            return 0
        return cost.get((c1, c2), 1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i+1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j+1] + 1
            deletions = current_row[j] + 1
            # changed
            substitutions = previous_row[j] + substitution_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))

        if debug :
            print(current_row[1:])

        previous_row = current_row

    return previous_row[-1]


def jamo_levenshtein(s1, s2, debug=False):   ##편집거리 계산
    if len(s1) < len(s2) :
        return jamo_levenshtein(s2, s1, debug)

    if len(s2) == 0:
        return len(s1)

    def substitution_cost(c1,c2):
        if c1 == c2:
            return 0
        return levenshtein(decompose(c1), decompose(c2))/3

    previous_row = range(len(s2) +1)
    for i,c1 in enumerate(s1):
        current_row = [i+1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            # Changed
            substitutions = previous_row[j] + substitution_cost(c1, c2)
            current_row.append(min(insertions, deletions, substitutions))

        ##  편집거리 행렬 출력
        # if debug:
        #     print(['%.3f' % v for v in current_row[1:]])

        previous_row = current_row

    return previous_row[-1]




def morph():
    now = datetime.now()
    nowdatetime = now.strftime('%Y-%m-%d %H:%M:%S')

    print("---morph start---")
    conn = pymysql.connect(host='10.231.238.16', port=3306, user='signal_admin', password='signal12#$',
                           db='vchtbot')
    model = gensim.models.Word2Vec.load("w2v_model_128_window10_mincount10_word_v5")
    wv_from_text = KeyedVectors.load(datapath('w2v_model_128_window10_mincount10_word_v5'))
    print("text : ",wv_from_text)
    w2c = dict()
    words = []
    for item in model.wv.vocab:
        words.append(item)
    print("num : ", len(words))


    words.sort()  ##sorting


    bigram_Transformer = Phrases(common_texts)
    model = Word2Vec(bigram_Transformer[common_texts], min_count=1)
    curs = conn.cursor()
    FROM = "STOPWORD_SYNONYM_PAIR (STOPWORD, STOPWORD_NUM, STOPWORD_PAIR, INSERTED_TIME)" #STOPWORD_DICT_MORPH (SEQ, MORPH, USE_YN, INSERTED_TIME)"


    ## stopword_dict_morph table 생성
    # i = 1
    # for word in words:
    #     sql = "INSERT INTO "+FROM+" VALUES ('"+ str(i) +"', '"+str(word)+"', 'Y', '"+str(nowdatetime)+"');"  ##; 필수
    #     print(sql)
    #     curs.execute(sql)
    #     row=curs.fetchall()
    #     i += 1

    ## stopword_synonym_pair table 생성
    # sql3 = "INSERT INTO " + FROM + " VALUES ('롯데정보통신', '4', 'lotte data communication company', '" +str(nowdatetime)+"');"  #str(i) + "', '" + str(word) + "', 'N', '" + str(nowdatetime) + "');"  ##; 필수
    # print(sql3)
    # curs.execute(sql3)
    # row = curs.fetchall()


    # view stopword_dict_morph
    # sql2 = "SELECT * FROM "+FROM
    # pd_sql.execute(sql2, conn)
    # df = pd_sql.read_sql(sql2,conn, index_col=None)
    # print(df)
    #
    # DataFrame.to_csv(df,'test.txt', header=None, index=None, mode='a')
    conn.commit()
    conn.close()
    # curs.execute(sql2)
    # print(curs.fetchall())



if __name__ == "__main__":

    print("---gensim check---")
    conn = pymysql.connect(host='10.231.238.16', port=3306, user='signal_admin', password='signal12#$',
                           db='vchtbot')
    curs = conn.cursor()
    ## gensim_list 가져오기
    sql = "SELECT MORPH FROM STOPWORD_DICT_MORPH;"
    print(sql)
    curs.execute(sql)
    row = curs.fetchall()
    gensim_list = []
    l = len(row)
    print("len ",l)
    for i in range(l):
        temp = list((row[i]))
        gensim_list.extend(temp)   ##gensim 단어 리스트 생성



    company_list = []
    sql1 = "SELECT STOPWORD FROM STOPWORD_DICT;"
    print(sql1)
    curs.execute(sql1)
    row = curs.fetchall()
    for j in row:
        j = list(j)
        j = j[0].replace("\(\)", "")
        company_list.append(j)    ##회사명 리스트 생성

    conn.commit()
    conn.close()

    gensim_set = set(gensim_list)
    company_set = set(company_list)
    # print("gensim list : ",gensim_list)
    words = []
    print("---start---")
    f = open("sample.txt", 'r', encoding='UTF8')  ##자기소개서 읽어오기
    for i  in f.readlines():
        words.append(i)
    words = words[0].split()   ##sample words
    word_list = []
    for i in words:
        print("i  : ",i)
        i = re.sub(u"\,",'',i)
        i = re.sub(u"\.", '', i)
        i = re.sub(u"\'", '', i)
        i = re.sub(u"[a-zA-Z]", '', i)
        # i=i.replace('\(\)\,','')
        # i=i.replace('\,','')
        word_list.append(i)
    print("words",word_list)


    for keyword in word_list:
        check = []
        if (keyword in gensim_set) or (keyword in company_set): ##표준어 여부 확인
            print("표준어 O : ", keyword)
            continue

        else:
            print("표준어 X ", keyword)
            for com in company_list:   ##회사명 리스트와 편집거리 계산
                temp = []
                temp.append(keyword)
                temp.append(com)
                temp.append(jamo_levenshtein(keyword, com, debug=True))
                check.append(temp)
            # print("check : ",check)
            for gen in gensim_list:   ##gensim 단어 편집거리 계산

                gen = re.sub(u"\,", '', gen)
                gen = re.sub(u"\.", '', gen)
                gen = re.sub(u"\|", '', gen)
                gen = re.sub(u"\'", '', gen)
                gen = re.sub(u"\;", '', gen)

                gen = re.sub(u"[0-9a-zA-Z]", '', gen)

                if(gen == ''):
                    continue
                else:
                    # print("gen : ",gen)
                    temp = []
                    temp.append(keyword)
                    temp.append(gen)
                    temp.append(jamo_levenshtein(keyword, gen, debug=True))
                    check.append(temp)
        # print("check : ",check)
        my_df = pd.DataFrame(check)
        name = str(keyword)+"_ouput.csv"
        my_df.to_csv(name, encoding ='ms949', index=False, header= False)

        sorted_check = sorted(check, key=lambda jamo:jamo[2])
        print("Close : "+str(sorted_check[0])+" "+str(sorted_check[1])+"  "+str(sorted_check[2]))

    f.close()