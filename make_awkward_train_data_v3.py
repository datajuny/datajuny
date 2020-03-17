import re

from tqdm import tqdm
import pymysql
import pandas as pd

from nltk.tokenize import sent_tokenize

'''
답변별 -> 문장분리 csv 파일 5개 생성
v2 는 언어모델을 위해 만든 데이터이다.
v3 첨삭대상 문장을 위한 데이터 구성(문장1, 문장2, LABEL) -> data 구분자는 탭( \t )
'''

vocabulary_path1 = './data_in/first_resume_v3.csv'
vocabulary_path2 = './data_in/second_resume_v3.csv'
vocabulary_path3 = './data_in/third_resume_v3.csv'
vocabulary_path4 = './data_in/forth_resume_v3.csv'
vocabulary_path5 = './data_in/fifth_resume_v3.csv'

vocabulary_path6 = './data_in/all_resume_v3.csv'
vocabulary_path7 = './data_in/all_abnormal_resume_v3.csv'

DATABASE_CONFIG = {
     'host': '10.131.8.144',
     'dbname': 'recruit',
     'user': 'root',
     'password': 'p@ssw0rd!@',
     'port': 3306,
     'charset':'utf8'
}

def make_resume_sentence_train_data():

    conn = pymysql.connect(host=DATABASE_CONFIG['host'], port=DATABASE_CONFIG['port'],
                           user=DATABASE_CONFIG['user'], passwd=DATABASE_CONFIG['password'],
                           db=DATABASE_CONFIG['dbname'], charset=DATABASE_CONFIG['charset'])
    curs = conn.cursor()

    read_resume1 = ("SELECT RESUME1 "
                   "FROM APPLY_DATA_RAW "
                   "WHERE Title LIKE '%신입채용 일반전형%' "
                   "AND length(RESUME1) > 10 ")

    read_resume2 = ("SELECT RESUME2 "
                   "FROM APPLY_DATA_RAW "
                   "WHERE Title LIKE '%신입채용 일반전형%' "
                   "AND length(RESUME2) > 10 ")

    read_resume3 = ("SELECT RESUME3 "
                   "FROM APPLY_DATA_RAW "
                   "WHERE Title LIKE '%신입채용 일반전형%' "
                   "AND length(RESUME3) > 10 ")

    read_resume4 = ("SELECT RESUME4 "
                   "FROM APPLY_DATA_RAW "
                   "WHERE Title LIKE '%신입채용 일반전형%' "
                   "AND length(RESUME4) > 10 ")

    read_resume5 = ("SELECT RESUME5 "
                   "FROM APPLY_DATA_RAW "
                   "WHERE Title LIKE '%신입채용 일반전형%' "
                   "AND length(RESUME5) > 10")

    read_resume6 = ("SELECT RESUME1, RESUME2, RESUME3, RESUME4, RESUME5 "
                   "FROM APPLY_DATA_RAW "
                   "WHERE Title LIKE '%신입채용 일반전형%' "
                   "AND length(RESUME1) > 10 AND length(RESUME2) > 10 AND length(RESUME3) > 10 AND length(RESUME4) > 10 AND length(RESUME5) > 10")

    curs.execute(read_resume1)
    rows1 = curs.fetchall()

    curs.execute(read_resume2)
    rows2 = curs.fetchall()

    curs.execute(read_resume3)
    rows3 = curs.fetchall()

    curs.execute(read_resume4)
    rows4 = curs.fetchall()

    curs.execute(read_resume5)
    rows5 = curs.fetchall()

    curs.execute(read_resume6)
    rows6 = curs.fetchall()

    # 답변별 데이터 만드는 소스
    # 지원자별 자소서를 반환한다.
    # for person in tqdm(rows1):
    #
    #     # 답변 순서대로 자소서를 반환한다.
    #     for resume in person:
    #
    #         if len(resume) < 30:
    #             continue
    #
    #         resume = preprocessing(resume)
    #
    #         sent_resume = sent_tokenize(resume)
    #
    #         len_sent_resume = len(sent_resume)
    #
    #         with open(vocabulary_path1, 'a', encoding = 'utf-8') as vocabulary_file:
    #
    #             for i in range(0, len_sent_resume - 1):
    #
    #                 if len(sent_resume[i]) < 5 or len(sent_resume[i+1]) < 5:
    #                     continue
    #                 vocabulary_file.write(sent_resume[i] + '\t' + sent_resume[i+1] + '\t' + str(1) + '\n')
    #

    # write head
    with open(vocabulary_path6, 'a', encoding = 'utf-8') as vocabulary_file:
        vocabulary_file.write('answer_num' + '\t' + 'sen1' + '\t' + 'sen2' + '\t' + 'label' + '\n')

    # 66
    #지원자별 자소서를 반환한다.
    for person in tqdm(rows6):

        # 답변 순서대로 자소서를 반환한다.
        for idx, resume in enumerate(person):
            # idx : 답변번호

            if len(resume) < 30:
                continue

            resume = preprocessing(resume)

            # 문장분리
            sent_resume = sent_tokenize(resume)

            # 문장갯수
            len_sent_resume = len(sent_resume)

            with open(vocabulary_path6, 'a', encoding='utf-8') as vocabulary_file:
                for i in range(0, len_sent_resume - 1):
                    # 길이가 5이하인 문장은 스킵한다.
                    if (len(sent_resume[i]) < 5) or (len(sent_resume[i+1]) < 5):
                        continue
                    vocabulary_file.write(str(idx + 1) + '\t' + sent_resume[i] + '\t' + sent_resume[i + 1] + '\t' + str(1) + '\n')


def make_resume_abnormal_sentence_train_data():

    normal_resume = pd.read_csv("/data/araVom/Language_model/data_in/all_resume_v3.csv", encoding='utf-8', sep='\t')
    chatbot = pd.read_csv('/data/araVom/Language_model/data_in/ChatBotData.csv', encoding='utf-8')

    with open(vocabulary_path7, 'a', encoding = 'utf-8') as vocabulary_file:
        vocabulary_file.write('answer_num' + '\t' + 'sen1' + '\t' + 'sen2' + '\t' + 'label' + '\n')

    for stop, values in tqdm(enumerate(normal_resume[['answer_num', 'sen1']].values)):

        if len(values) != 2:
            continue

        resume_no = values[0]
        sentence = values[1]

        with open(vocabulary_path7, 'a', encoding='utf-8') as vocabulary_file:

            if stop % 3 == 0:
                vocabulary_file.write(sentence + '\t' + normal_resume[normal_resume['answer_num'] != resume_no]['sen2'].sample(n=1).values[0] + '\t' + str(0) + '\n')

            elif stop % 3 == 1:
                vocabulary_file.write(sentence + '\t' + chatbot['Q'].sample(n=1).values[0] + '\t' + str(0) + '\n')

            elif stop % 3 == 2:
                vocabulary_file.write(sentence + '\t' + chatbot['A'].sample(n=1).values[0] + '\t' + str(0) + '\n')


def preprocessing(resume):
#def preprocessing(resume, company_list):

    resume = re.sub("\&[a-zA-Z]+\;", " ", resume)  # 태그제거(ex: &lt;)
    resume = re.sub('[^a-zA-Z|가-힣|a-zA-Z|0-9|.]+', " ", resume)
    resume = resume.lower() # 소문자로 변경
    resume = re.sub("[.]", ". ", resume)
    resume = re.sub("[,]", " ", resume)
    resume = re.sub("[\"]", " ", resume)

    # for c in company_list:
    #     resume = re.sub(c, ' [회사명]', resume)
    #
    # resume = re.sub("(\[회사명\] \[회사명\])", "[회사명]", resume)
    # resume = re.sub("(\[회사명\]  \[회사명\])", "[회사명]", resume)
    # resume = re.sub("(\[회사명\] \[회사명\] \[회사명\])", "[회사명]", resume)

    return resume


if __name__ == "__main__":

    # 정상 케이스 만드는 함수
    # make_resume_sentence_train_data()

    # 반대 케이스 만드는 함수
    # (1) 정상 문장 + 다른 답변(33%)
    # (2) 정상 문장 + 챗봇 질문(33%)
    # (3) 정상 문장 + 챗본 답변(33%)
    make_resume_abnormal_sentence_train_data()
