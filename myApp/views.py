from django.shortcuts import render
import pandas as pd

##
def index(request):
    return render(request , 'index.html')



def main(request):
    return render(request , 'main.html')


def result(request):
    text = request.GET['search']
    text = text.split(sep=', ')
    context=find_info(text[0],text[1])
    return render(request, 'result.html', context)

def find_info(lec, prof):
    print(lec)
    print(prof)
    review_df = pd.read_csv('myApp/static/src/review_datal_all.csv')
    review_a = review_df[review_df['professor_name'] == prof]
    review_b = review_a[review_a['lecture_name'] == lec]
    review_b.reset_index(inplace=True)
    review_tmp = review_b['text'].to_list()
    review_data = '<hr>'.join(review_tmp)
    print(type(review_data))
    df = pd.read_csv('myApp/static/src/testset_groupby.csv')
    homework = ['과제']
    lecture = ['수업']
    professor = ['교수']
    exam = ['시험']
    grade = ['학점']

    a = df[df['professor_name'] == prof]
    b = a[a['lecture_name'] == lec]
    b.reset_index(inplace=True)
    for i in range(len(b)):
        if b['aspect'][i] == homework[0]:
            homework.append(b['score'][i])
        elif b['aspect'][i] == lecture[0]:
            lecture.append(b['score'][i])
        elif b['aspect'][i] == professor[0]:
            professor.append(b['score'][i])
        elif b['aspect'][i] == exam[0]:
            exam.append(b['score'][i])
        elif b['aspect'][i] == grade[0]:
            grade.append(b['score'][i])
        else:
            break
    if len(homework) == 1:
        print("HOMEWORK NULL")
        homework.append(0.0)
    if len(lecture) == 1:
        print("LECJTURE NULL")
        lecture.append(0.0)
    if len(professor) == 1:
        print("PROFESSOR NULL")
        professor.append(0.0)
    if len(exam) == 1:
        print("EXAM NULL")
        exam.append(0.0)
    if len(grade) == 1:
        print("GRADE NULL")
        grade.append(0.0)


    summary = get_summary(prof,lec)
    get_wc(prof, lec)
    summary = '<br>'.join(summary)

    preprocessed = use_multiprocess(data_text_preprocessing, review_df["text"], 3)
    review_df["token"] = preprocessed
    sim = similarity(review_df, lec)
    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
    print(type(sim))
    sim_lec = sim['lecture_name'].tolist()
    sim_score = sim['score'].tolist()
    sim_lec_result = ''
    sim_score_result = ''
    for i in range(len(sim_lec)):
        sim_lec_result = sim_lec_result + str(sim_lec[i]) +'<br>'
        sim_score_result = sim_score_result + str(round(sim_score[i],2)) +'<br>'


    context={
        'class_score' : round(lecture[1],1)*10,
        'homework_score' : round(homework[1],1)*10,
        'professor_score' : round(professor[1],1)*10,
        'exam_score' : round(exam[1],1)*10,
        'grade_score' : round(grade[1],1)*10,
        'lecture_name' : lec,
        'professor_name' : prof,
        'summary' : summary,
        'review' : review_data,
        'sim_lec' : sim_lec_result,
        'sim_score' : sim_score_result
    }

    return context;

def get_wc(prof, lec) :
    from wordcloud import WordCloud
    import matplotlib.pyplot as plt
    import pandas as pd
    from konlpy.tag import Mecab
    from collections import Counter

    df = pd.read_csv('myApp/static/src/review_By_lec_prof.csv')


    a = df[df['professor_name']==prof]
    a = a[a['lecture_name']==lec]
    mecab = Mecab()

    noun = mecab.nouns(a['clean_txt'].tolist()[0])
    for i,v in enumerate(noun) :
        if len(v)<2 :
            noun.pop(i)

    print(noun)
    count = Counter(noun)
    print(count)
    noun_list = count.most_common(100)
    print('!!!!!!')
    print(noun_list)
    print('~~~~~~~')
    wc = WordCloud(background_color = 'white', font_path = 'myApp/static/src/NanumGothic.ttf',  width = 300, height = 300)
    print(wc)
    wc.generate_from_frequencies(dict(noun_list)) # font_path='font/NanumGothic.ttf' ,stopwords=stopwords
    plt.figure()
    plt.imshow(wc)
    plt.axis('off')
    wc.to_file('myApp/static/wordcloud/test.png')

def get_summary(professor, lecture):
    from typing import List
    from konlpy.tag import Mecab
    import numpy as np
    import pandas as pd
    import re
    import matplotlib.pyplot as plt
    from lexrankr import LexRank
    from konlpy.tag import Mecab

    df_idx = pd.read_csv('myApp/static/src/review_By_lec_prof.csv')
    mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')

    class MecabTokenizer:
        mecab: Mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')
        def __call__(self, text: str) -> List[str]:
            tokens: List[str] = self.mecab.morphs(text)
            return tokens

    mecab: Mecab = MecabTokenizer()
    lexrank: LexRank = LexRank(mecab)

    tmp_df = df_idx[df_idx['professor_name'] == professor]
    tmp_df = tmp_df[tmp_df['lecture_name']==lecture]
    tmp_df.reset_index(inplace=True)
    test = tmp_df['clean_txt'].tolist()
    summary = lexrank.summarize(tmp_df['clean_txt'][0])

    summaries = lexrank.probe()  # probe(10)
    # for i, summary in enumerate(summaries[:10]):
    #     print(professor, "교수님의", lecture, '강의 요약평 : ')
    #     print(str(i), " : ", summary)

    return summaries[:7]


def data_text_preprocessing(data):
    # 한글
    import re
    korean = re.sub("[^ㄱ-ㅎㅏ-ㅣ가-힣]", " ", data)
    stopwords = '이 있 하 것 들 그 되 수 이 보 않 없 나 사람 주 아니 등 같 우리 때 고 년 가 한 지 대하 오 말 일 그렇 위하 은 는 함 음 심 습니다 아요 세요 에요 었 였 에 을 를'.split()

    # 토큰화와 불용어
    from konlpy.tag import Mecab
    mecab = Mecab()
    tokenization = mecab.morphs(korean)
    no_stopwords = [token for token in tokenization if token not in stopwords]

    return ' '.join(no_stopwords)

from multiprocessing import Pool

def use_multiprocess(func, iter, workers):
    pool = Pool(processes=workers)
    result = pool.map(func, iter)
    pool.close()
    return result

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel


def similarity(review, lec):
    # parameter : review csv파일

    # {title:idx} 데이터 구조 만들기
    title = set(list(review['lecture_name']))
    title = sorted(list(title))
    size = len(title)
    indices = dict(zip(title, range(size)))

    # lecture review만들기
    lecture_review = {}
    for index, row in review.iterrows():
        lecture_name = row['lecture_name']
        corpus = row['token']
        if type(corpus) == str:
            if lecture_name in lecture_review.keys():
                lecture_review[lecture_name] += corpus
            else:
                lecture_review[lecture_name] = corpus
        else:  # 국어의미론 NAN값 존재
            if lecture_name in lecture_review.keys():
                lecture_review[lecture_name] += ''
            else:
                lecture_review[lecture_name] = ''
    lecture_review = sorted(lecture_review.items())
    data = [text[1] for text in lecture_review]

    # 유사도를 구하기 위한 수치벡터화(tfidf)
    from sklearn.feature_extraction.text import TfidfVectorizer
    tfidf = TfidfVectorizer()
    # overview에 대해서 tf-idf 수행
    tfidf_matrix = tfidf.fit_transform(data)
    tfidf_matrix.shape

    # 코사인 유사도
    from sklearn.metrics.pairwise import linear_kernel
    cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

    lecture_review = pd.DataFrame(lecture_review, columns=['lecture_name', 'corpus'])

    def recommendations_by_cosine(title, method=cosine_sim):
        idx = indices[title]
        scores = list(enumerate(method[idx]))  # 모든 강의와 유사도 구하기
        scores = sorted(scores, key=lambda x: x[1], reverse=True)  # 유사도 정렬
        scores = scores[1:6]  # 가장 유사한 강의 5개
        movie_indices = [i[0] for i in scores]  # 강의의 인덱스받기

        result_df = lecture_review.iloc[movie_indices].copy()
        result_df['score'] = [i[1] for i in scores]  # score추가

        del result_df['corpus']

        return result_df

    result = recommendations_by_cosine(lec)
    return result