import firebase_admin
from firebase_admin import credentials
from firebase_admin import db
import os
import pandas as pd
import numpy as np
import gensim
import re
from tqdm import trange
from sklearn.metrics.pairwise import cosine_similarity
from nltk.corpus import stopwords
from Exceptions import NonExistentUsername
from Exceptions import CallError
from sys import argv

class Recomendations():

    def __init__(self, path, mode, wv):
        """
        Режим инициализации: получение векторных представлений слов (эмбеддингов),
        их сохранение для последующего использования
        Режим использования: загрузка готовых эмбеддингов и исользование
        для подбора рекомендаций.
        """
        self.products = None
        self.interactions = None
        self.wv = None
        self.index_words_set = None
        self.get_data(path) #получение данных в датафрейм

        stop_words = stopwords.words('russian') + ['ассортимент', 'ассортименте', 'ассортим', 'см', 'м', 'шт', 'л', 'г', 'против']
        values = np.zeros(len(stop_words))
        self.stop_words_dict = dict(zip(stop_words, values))
        if mode == 'init':
            self.get_products_embeddings()
            self.model.wv.save("word2vec_recsys.wordvectors")
        elif mode == "using":
            self.wv = wv
            self.index_words_set = set(self.wv.index_to_key)
            self.variable_initialization(self.wv)

    def get_data(self, path):
        """
        Получение данных об актуальных товарах в pandas DataFrame
        """
        df = pd.read_csv(path)
        self.products = df.copy()

    def get_interactions(self, uname):
        """
        Получение из базы данных информации о товарах, 
        с которыми взаимодействовал пользователь
        """
        #os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = path/to/serviceAccountKey.json
        cred = credentials.Certificate(os.environ['GOOGLE_APPLICATION_CREDENTIALS'])
        firebase_admin.initialize_app(cred, {
        'databaseURL': 'https://gift-ideas-2-default-rtdb.europe-west1.firebasedatabase.app'
        })
        ref = db.reference("History")
        structure = ref.get()
        if uname not in structure.keys():
            raise NonExistentUsername("User with such name does not exist!")
        self.interactions = [int(k) for k in structure[uname]['history_of_search'].keys()]
        self.interactions.extend([int(k) for k in structure[uname]['wishlist'].keys()])

    def get_avg_vector(self, words, wv, num_features, index_words_set):
        """
        Получение векторного представления предложения путём усреднения 
        векторов входящих в него слов
        """
        feature_vec = np.zeros((num_features, ), dtype='float32')
        n_words = 0
        for word in words:
            if word in index_words_set:
                n_words += 1
                feature_vec = np.add(feature_vec, wv[word])
        if (n_words > 0):
            feature_vec = feature_vec / n_words
        return feature_vec    
    
    def preprocess(self, text, stop_words_dict):
        """
        Предобработка текста (очистка от ненужных символов,
        приведение слов к нижнему регистру, удаление стоп слов)
        """
        word_list = []
        for word in re.findall("[а-яА-Яa-zA-ZёЁ]+", text.lower()):
            if word not in stop_words_dict:
                word_list.append(word)
        return word_list

    def variable_initialization(self, wv, products_descriptions = None):
        """
        Инициализация необходимых переменных.
        """
        if products_descriptions is None:
            products_descriptions = []
            for i in trange(0, self.products['name'].shape[0]):
                products_descriptions.append(self.preprocess(self.products['name'].loc[i], self.stop_words_dict))
        self.wv = wv
        self.index_words_set = set(self.wv.index_to_key)
        self.embeddings = np.zeros((len(products_descriptions), 200))
        for i in range(self.embeddings.shape[0]):
            self.embeddings[i] = self.get_avg_vector(products_descriptions[i], self.wv, 200, self.index_words_set)

    def get_products_embeddings(self):
        """
        Обучение модели word2vec на текстовых названиях товаров. 
        Формирование векторов для всех слов, использованных в названиях.
        Получение векторного представления для текстовых названий товаров
        (внутри метода self.variable_initialization())
        """
        products_descriptions = []
        for i in trange(0, self.products['name'].shape[0]):
            products_descriptions.append(self.preprocess(self.products['name'].loc[i], self.stop_words_dict))
            
        self.model = gensim.models.Word2Vec(
                sentences=products_descriptions,
                vector_size=200,                  #размер эмбеддингов
                window=10,
                min_count=3)
        self.variable_initialization(self.model.wv, products_descriptions)
            
    def get_rec_U2I(self):
        """
        Получение рекомендаций для конкретного пользователя:
        1) объединение текстовых названий товаров, с которыми взаимодействовал
        пользователь, в одну строку и её предобработка;
        2) получение усредненного вектора для этой строки
        3) нахождение косинусной близости для вектора товаров 
        конкретного пользователя с каждым из остальных товаров
        4) топ 10 товаров по полученной метрике - рекомендации
        для данного пользователя
        """
        user_vector = " ".join(self.products[self.products['article']==_id]['name'].to_string() for _id in self.interactions)
        user_processed = self.preprocess(user_vector, self.stop_words_dict)
        user_emb = self.get_avg_vector(user_processed, self.wv, 200, self.index_words_set)
        metrics = cosine_similarity([user_emb], self.embeddings)
        self.products['metrics']=metrics.reshape(-1)
        print(u"Для пользователя, который взаимодействовал с товарами\n")
        for id_ in self.interactions:
            print(self.products[self.products['article']==id_]['name'].to_string()+" {}".format(id_))
        print(u"\nТакие рекомендации")
        recomendations = pd.DataFrame(self.products.sort_values(by='metrics', ascending=False).loc[:,['name', 'metrics', 'article']])
        for product in self.interactions:
            recomendations.drop(recomendations[recomendations.article==product].index, inplace = True)
        print(recomendations[:10])
        recomendations[:10]['article'].to_json("Recomendations.json")

if __name__ == '__main__':
    if len(argv) > 5 or len(argv) < 3:
        raise CallError("Uncorrect number of comand line parameters")

    if len(argv) == 5 :
        wv = gensim.models.KeyedVectors.load(argv[4], mmap='r')
        mode = argv[3]
    else:
        wv = None
        mode = argv[3]
    
    recsys = Recomendations(argv[2], mode, wv)
    uname = argv[1]
    recsys.get_interactions(uname) #чтение из бд полей необходимого ползователя
    recsys.get_rec_U2I()

#USAGE EXAMPLE: python Recomendations_with_db.py <uname> <path_to_products.csv> <mode> <path_to_word2vec_recsys.wordvectors>(optional)
#in catalog with these file you also should have new 
#python Recomendations_with_db.py w@w_com products.csv using word2vec_recsys.wordvectors
#python Recomendations_with_db.py w@w_com products.csv init