import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import cosine_similarity
from utils import *
from feature_extraction import *


class TwoStageModel(object):
    """Двух-уровневая модель рекомендательной системы."""

    def __init__(self, model1_params: dict, model2_params: dict):
        """
        Создание объекта класса.

        Args:
            model1_params:
                Гиперпараметры модели LightFM.

            model2_params:
                Гиперпараметры модели CatBoostClassifier.
        """
        self.model1_params = model1_params
        self.model2_params = model2_params
        self.model_is_fit = False


    def fit(self,
            df1: pd.DataFrame,
            df2: pd.DataFrame,
            movies: pd.DataFrame,
            num_candidates: int = 200):
        """
        Обучение модели.

        Args:
            df1:
                Датафрейм с тренировочными данными для модели LightFM.

            df2:
                Датафрейм с данными, из которых будут извлекаться таргеты для
                модели CatBoostClassifier.

            movies:
                Датафрейм с информацией по каждому фильму.

            num_candidates:
                Число кандидатов для рекомендации, из которых будут удалены
                повторы и отобраны финальные рекомендации.
        """
        # инициализация базовых моделей
        model1 = LightFM(**self.model1_params)
        self.model2 = CatBoostClassifier(**self.model2_params)

        # создание копий входных наборов данных и сохранение временной метки
        # верхней границы подвыборки
        df_train1 = df1.copy()
        train_targets = df2.copy()
        split_border = df_train1['datetime'].max()

        # все предыдущие запуски каждого пользователя
        prev_starts = df_train1[['user_id', 'movie_id']].drop_duplicates()
        prev_starts['prev_start'] = 1

        # предобработка логов
        df_train1 = preprocess(df_train1)
        df_train1 = df_train1[df_train1['personal_score'] == 1]

        # создание таргетов для выборки для обучения CatBoostClassifier
        train_targets = train_targets[['user_id', 'movie_id']].drop_duplicates()
        train_targets['personal_score'] = 1

        # конвертация датасета в удобный вид для обучения модели LightFM
        dataset = Dataset()
        dataset.fit(
            df_train1['user_id'].unique(),
            df_train1['movie_id'].unique()
            )

        # создание матрицы взаимодействий
        interactions = dataset.build_interactions(
            df_train1[['user_id', 'movie_id', 'personal_score']].values
            )[0]

        # создание мэппингов
        user_dict, movie_dict = dataset.mapping()[0], dataset.mapping()[2]
        inv_user_dict = {v: k for k, v in user_dict.items()}
        inv_movie_dict = {v: k for k, v in movie_dict.items()}

        # обучение модели LightFM
        model1.fit(interactions=interactions, epochs=10)
        del interactions

        # сохранение признаков пользователей и фильмов
        last_starts_df = get_last_start(df1)
        user_warmness_df = get_warmness_feature(df_train1)
        genre_percentage_df = get_genre_percentage(df_train1, movies)
        decade_percentage_df = get_decade_percentage(df_train1, movies)
        combined_percentage_df = get_combined_percentage(df_train1, movies)

        del df_train1

        # матрица косинусного сходства по фильмам
        cos_sim_matrix = cosine_similarity(
            model1.item_embeddings, model1.item_embeddings
            )

        # перевод матрицы в датафрейм
        cos_sim_df = pd.DataFrame(
            index=list(movie_dict.values()),
            columns=list(movie_dict.values()),
            data=cos_sim_matrix,
            )

        del cos_sim_matrix

        cos_sim_df = cos_sim_df.stack().reset_index()
        cos_sim_df.rename(
            columns={
                'level_0': 'movie_id',
                'level_1': 'last_started_movie_id',
                0: 'cos_sim_to_last_start'
                },
            inplace=True,
            )

        # возвращение фильмам оригинальных id
        for col in ['movie_id', 'last_started_movie_id']:
            cos_sim_df[col] = cos_sim_df[col].map(inv_movie_dict)

        # список id пользователей и фильмов внутри датасета LightFM
        user_ids = list(user_dict.values())
        movie_ids = list(movie_dict.values())

        # создание тренировочного датафрейма для CatBoostClassifier
        df_train2 = pd.DataFrame({
            'user_id': user_ids,
            'rank': [np.arange(num_candidates).tolist()] * len(user_ids)
            })

        # добавление списка id рекомендаций от модели LightFM
        df_train2['movie_id'] = df_train2['user_id'].apply(
            lambda x: np.argsort(
                model1.predict(x, movie_ids)
                )[::-1][:num_candidates],
            )

        # перевод всех рекомендаций в вид отдельного элемента выборки и
        # замена всех id на оригинальные
        df_train2 = df_train2.explode(['movie_id', 'rank'], ignore_index=True)
        df_train2['user_id'] = df_train2['user_id'].map(inv_user_dict)
        df_train2['movie_id'] = df_train2['movie_id'].map(inv_movie_dict)

        # удаление из выборки всех пар пользователь-фильм, которые ранее
        # встречались
        df_train2 = df_train2.merge(
            prev_starts,
            on=['user_id', 'movie_id'],
            how='left',
            )

        df_train2 = df_train2[df_train2['prev_start'].isna()]
        df_train2.drop(columns=['prev_start'], inplace=True)

        del prev_starts

        # добавление таргета к каждому наблюдению
        df_train2 = df_train2.merge(
            train_targets,
            on=['user_id', 'movie_id'],
            how='left',
            )

        df_train2['personal_score'].fillna(0, inplace=True)
        df_train2['personal_score'] = df_train2['personal_score'].astype(int)

        del train_targets

        # кол-во таргетов в тренировочной выборке
        subsample_count = df_train2['personal_score'].value_counts()

        min_class = subsample_count.idxmin()
        max_class = subsample_count.idxmax()

        # аргументы функции сэмплирования наблюдений для тренировочной выборки
        min_sample_args = {
            'n': subsample_count.loc[min_class],
            'random_state': 42,
            'ignore_index': True
            }

        max_sample_args = {
            'n': 3 * subsample_count.loc[min_class],
            'random_state': 42,
            'ignore_index': True
            }

        df_train2 = pd.concat(
            [
                df_train2[
                    df_train2['personal_score'] == min_class
                    ].sample(**min_sample_args),
                df_train2[
                    df_train2['personal_score'] == max_class
                    ].sample(**max_sample_args),
                ],
            ignore_index=True,
            )

        # добавление признака прошедшего времени от последнего просмотра фильма
        df_train2 = df_train2.merge(last_starts_df, on='user_id')
        df_train2['time_since_last_start'] = split_border - df_train2['last_start_datetime']
        df_train2['time_since_last_start'] = df_train2['time_since_last_start'].apply(
            lambda x: x.total_seconds()
            )
        df_train2.drop(columns=['last_start_datetime'], inplace=True)

        del last_starts_df

        # добавление признака косинусного сходства между фильмом-кандидатом и
        # последним просмотренным фильмом
        df_train2 = df_train2.merge(
            cos_sim_df,
            on=['movie_id', 'last_started_movie_id'],
            )

        del cos_sim_df

        df_train2.drop(columns='last_started_movie_id', inplace=True)

        # добавление признака "тёплый пользователь"
        df_train2 = df_train2.merge(user_warmness_df, on='user_id')
        del user_warmness_df

        # добавление инфо о жанре фильма и декаде его выхода
        df_train2 = df_train2.merge(
            movies[['movie_id', 'genres', 'decade']],
            on='movie_id',
            )

        # добавление признака доля просмотров фильмов в жанре фильма-кандидата
        df_train2 = df_train2.merge(
            genre_percentage_df,
            on=['user_id', 'genres'],
            how='left',
            ).fillna(0)

        del genre_percentage_df

        # добавление признака доля просмотров фильмов с декадой выхода
        # фильма-кандидата
        df_train2 = df_train2.merge(
            decade_percentage_df,
            on=['user_id', 'decade'],
            how='left',
            ).fillna(0)

        del decade_percentage_df

        # добавление признака доля просмотров фильмов с декадой выхода и жанром
        # фильма-кандидата
        df_train2 = df_train2.merge(
            combined_percentage_df,
            on=['user_id', 'genres', 'decade'],
            how='left',
            ).fillna(0)

        del combined_percentage_df

        df_train2.drop(
            columns=['genres', 'decade', 'movie_id', 'user_id'],
            inplace=True,
            )

        # формирование списков признаков
        self.features = [
            'rank', 'warm_user', 'time_since_last_start',
            'cos_sim_to_last_start', 'genre_percent', 'decade_percent',
            'pair_percent'
            ]
        cat_features = ['warm_user']

        # разбиение тренировочного набора на тренировочную и валидационную
        # подвыборки
        x_train, x_val, y_train, y_val = train_test_split(
            df_train2[self.features],
            df_train2['personal_score'],
            test_size=0.1,
            stratify=df_train2['personal_score'],
            random_state=42,
            )

        del df_train2

        # обучение модели CatBoostClassifier
        self.model2.fit(
            X=x_train,
            y=y_train,
            cat_features=cat_features,
            eval_set=(x_val, y_val),
            verbose=0,
            )

        del x_train, y_train, x_val, y_val

        # добавление флага, что модель обучена
        self.model_is_fit = True


    def evaluate(self,
                 train_logs: pd.DataFrame,
                 val_logs: pd.DataFrame,
                 movies: pd.DataFrame,
                 num_candidates: int = 200):
        """
        Оценка предсказаний модели по метрике MAP@20.

        Args:
            train_logs:
                Датафрейм с тренировочными логами для модели LightFM.

            val_logs:
                Датафрейм с валидационными логами.

            movies:
                Датафрейм с информацией по каждому фильму.

            num_candidates:
                Число кандидатов для рекомендации, из которых будут удалены
                повторы и отобраны финальные рекомендации.
        """
        # проверка, обучена ли модель
        assert self.model_is_fit == True, 'You should fit model before evaluating it!'

        # инициализация модели LightFM
        model1 = LightFM(**self.model1_params)

        # все запуски фильмов всех пользователей за тренировочный период
        prev_starts = train_logs[['user_id', 'movie_id']].drop_duplicates()
        prev_starts['prev_start'] = 1

        # сохранение временной метки верхней границы подвыборки
        split_border = train_logs['datetime'].max()

        # id пользователей, по которым нет логов в валидационной подвыборке
        nan_users = np.setdiff1d(
            train_logs['user_id'].unique(),
            val_logs['user_id'].unique(),
            assume_unique=True,
            )

        nan_users_df = pd.DataFrame({'user_id': nan_users})
        nan_users_df['movie_id'] = np.nan

        # выделение таргетов подвыборки
        targets = val_logs[['user_id', 'movie_id']].copy()
        targets.drop_duplicates(inplace=True)
        targets = pd.concat([targets, nan_users_df])
        targets = targets.groupby('user_id').agg(list)
        targets = targets['movie_id']

        # предобработка тренировочных логов
        df = preprocess(train_logs)

        # топ популярных фильмов по числу посмотревших пользователей
        top20 = df['movie_id'].value_counts().index.tolist()[:20]
        top100 = df['movie_id'].value_counts().index.tolist()[:100]

        # сохранение только положительных взаимодействий для дальнейших операций
        df = df[df['personal_score'] == 1]

        # конвертация датасета в удобный вид для обучения модели LightFM
        dataset = Dataset()
        dataset.fit(df['user_id'].unique(), df['movie_id'].unique())

        # создание матрицы взаимодействий
        interactions = dataset.build_interactions(
            df[['user_id', 'movie_id', 'personal_score']].values
            )[0]

        # создание мэппингов
        user_dict, movie_dict = dataset.mapping()[0], dataset.mapping()[2]
        inv_user_dict = {v: k for k, v in user_dict.items()}
        inv_movie_dict = {v: k for k, v in movie_dict.items()}

        # обучение модели LightFM
        model1.fit(interactions=interactions, epochs=10)
        del interactions

        # сохранение признаков пользователей и фильмов
        last_starts_df = get_last_start(train_logs)
        user_warmness_df = get_warmness_feature(df)
        genre_percentage_df = get_genre_percentage(df, movies)
        decade_percentage_df = get_decade_percentage(df, movies)
        combined_percentage_df = get_combined_percentage(df, movies)

        del df

        # матрица косинусного сходства по фильмам
        cos_sim_matrix = cosine_similarity(
            model1.item_embeddings, model1.item_embeddings
            )

        # перевод матрицы в датафрейм
        cos_sim_df = pd.DataFrame(
            index=list(movie_dict.values()),
            columns=list(movie_dict.values()),
            data=cos_sim_matrix,
            )

        del cos_sim_matrix

        cos_sim_df = cos_sim_df.stack().reset_index()
        cos_sim_df.rename(
            columns={
                'level_0': 'movie_id',
                'level_1': 'last_started_movie_id',
                0: 'cos_sim_to_last_start'
                },
            inplace=True,
            )

        # возвращение фильмам оригинальных id
        for col in ['movie_id', 'last_started_movie_id']:
            cos_sim_df[col] = cos_sim_df[col].map(inv_movie_dict)

        # список id пользователей и фильмов внутри датасета LightFM
        user_ids = list(user_dict.values())
        movie_ids = list(movie_dict.values())

        # создание датафрейма с фильмами-кандидатами
        candidates = pd.DataFrame({
            'user_id': user_ids,
            'rank': [np.arange(num_candidates).tolist()] * len(user_ids)
            })

        # добавление списка id рекомендаций от модели LightFM
        candidates['movie_id'] = candidates['user_id'].apply(
            lambda x: np.argsort(
                model1.predict(x, movie_ids)
                )[::-1][:num_candidates],
            )

        # кол-во партий, на которые разбивается датафрейм для экономии памяти
        num_batches = np.ceil(len(user_ids) / 50000)

        # последовательность операций с каждой партией данных
        for inds_chunk in np.array_split(candidates.index.values, num_batches):

            temp_set = candidates.loc[inds_chunk].copy()
            candidates.drop(index=inds_chunk, inplace=True)

            # перевод всех рекомендаций в вид отдельного элемента выборки и
            # замена всех id на оригинальные
            temp_set = temp_set.explode(['movie_id', 'rank'], ignore_index=True)
            temp_set['user_id'] = temp_set['user_id'].map(inv_user_dict)
            temp_set['movie_id'] = temp_set['movie_id'].map(inv_movie_dict)

            # удаление из выборки всех пар пользователь-фильм, которые ранее
            # встречались
            temp_set = temp_set.merge(
                prev_starts,
                on=['user_id', 'movie_id'],
                how='left',
                )

            temp_set = temp_set[temp_set['prev_start'].isna()]
            temp_set.drop(columns='prev_start', inplace=True)

            # добавление признака прошедшего времени от последнего просмотра
            # фильма
            temp_set = temp_set.merge(last_starts_df, on='user_id')
            temp_set['time_since_last_start'] = split_border - temp_set['last_start_datetime']
            temp_set['time_since_last_start'] = temp_set['time_since_last_start'].apply(
                lambda x: x.total_seconds()
                )
            temp_set.drop(columns=['last_start_datetime'], inplace=True)

            # добавление признака косинусного сходства между фильмом-кандидатом
            # и последним просмотренным фильмом
            temp_set = temp_set.merge(
                cos_sim_df,
                on=['movie_id', 'last_started_movie_id'],
                )

            temp_set.drop(columns=['last_started_movie_id'], inplace=True)

            # добавление признака "тёплый пользователь"
            temp_set = temp_set.merge(user_warmness_df, on='user_id')

            # добавление инфо о жанре фильма и декаде его выхода
            temp_set = temp_set.merge(
                movies[['movie_id', 'genres', 'decade']],
                on='movie_id',
                )

            # добавление признака доля просмотров фильмов в жанре
            # фильма-кандидата
            temp_set = temp_set.merge(
                genre_percentage_df,
                on=['user_id', 'genres'],
                how='left',
                ).fillna(0)

            # добавление признака доля просмотров фильмов с декадой выхода
            # фильма-кандидата
            temp_set = temp_set.merge(
                decade_percentage_df,
                on=['user_id', 'decade'],
                how='left',
                ).fillna(0)

            # добавление признака доля просмотров фильмов с декадой выхода и
            # жанром фильма-кандидата
            temp_set = temp_set.merge(
                combined_percentage_df,
                on=['user_id', 'genres', 'decade'],
                how='left',
                ).fillna(0)

            temp_set.drop(columns=['genres', 'decade'], inplace=True)

            # предсказание вероятности запуска фильма от CatBoostClassifier
            temp_set['proba'] = self.model2.predict_proba(
                temp_set[self.features]
                )[:,1]
            temp_set = temp_set[['user_id', 'movie_id', 'proba']]

            # сохранение топ-20 наиболее вероятных кандидатов в виде списка
            temp_set = temp_set.sort_values(
                by=['user_id', 'proba'],
                ascending=[True, False],
                )

            temp_set.drop(columns=['proba'], inplace=True)
            temp_set = temp_set.groupby('user_id').head(20)
            temp_set = temp_set.groupby('user_id', as_index=False).agg(list)

            # соединение датафреймов из разных партий
            if 'recs' in locals():
                recs = pd.concat([recs, temp_set])
            else:
                recs = temp_set.copy()

            del temp_set

        recs.set_index('user_id', inplace=True)

        # освобождение памяти
        del candidates, last_starts_df, cos_sim_df, user_warmness_df
        del genre_percentage_df, decade_percentage_df, combined_percentage_df

        # группа "холодных пользователей" #1
        # они есть в валидационной выборке, но их нет в тренировочной
        # для них стандартная рекомендация - топ-20
        cold_users1 = np.setdiff1d(
            val_logs['user_id'].unique(),
            train_logs['user_id'].unique(),
            assume_unique=True,
            )

        cold_df1 = pd.DataFrame({
            'user_id': cold_users1,
            'movies': [top20] * len(cold_users1)
            })

        cold_df1.set_index('user_id', inplace=True)

        # группа "холодных пользователей" #2
        # они исчезли из тренировочной выборки после предобработки, потому что
        # у них нет ни одного просмотренного фильма
        # для них стандартная рекомендация - топ-100, из которых удаляются все
        # запускавшиеся ранее фильмы и отбираются топ-20
        cold_users2 = np.setdiff1d(
            train_logs['user_id'].unique(),
            list(user_dict.keys()),
            assume_unique=True,
            )

        cold_df2 = pd.DataFrame({
            'user_id': cold_users2,
            'movie_id': [top100] * len(cold_users2)
            })

        cold_df2 = cold_df2.explode('movie_id', ignore_index=True)

        cold_df2 = cold_df2.merge(
            prev_starts,
            on=['user_id', 'movie_id'],
            how='left',
            )

        cold_df2 = cold_df2[cold_df2['prev_start'].isna()]
        cold_df2 = cold_df2.groupby('user_id').head(20)[['user_id', 'movie_id']]
        cold_df2 = cold_df2.groupby('user_id', as_index=False).agg(list)
        cold_df2.set_index('user_id', inplace=True)

        # соединение датафреймов с рекомендациями для "тёплых" и "холодных"
        # пользователей
        recs = pd.concat(
            [recs['movie_id'], cold_df1['movies'], cold_df2['movie_id']]
            )
        recs.name = 'movies'

        # вывод значения метрики MAP@20
        map_at_20(recs, targets)


    def predict(self,
                logs: pd.DataFrame,
                movies: pd.DataFrame,
                num_candidates: int = 200):
        """
        Создание списка рекомендаций для каждого пользователя.

        Args:
            logs:
                Датафрейм с логами для модели LightFM.

            movies:
                Датафрейм с информацией по каждому фильму.

            num_candidates:
                Число кандидатов для рекомендации, из которых будут удалены
                повторы и отобраны финальные рекомендации.

        Returns:
            recs:
                Датафрейм, где каждому пользователю соответствует список из 20
                фильмов.
        """
        # проверка, обучена ли модель
        assert self.model_is_fit == True, 'You should fit model before making predictions!'

        # инициализация модели LightFM
        model1 = LightFM(**self.model1_params)

        # все запуски фильмов всех пользователей за тренировочный период
        prev_starts = logs[['user_id', 'movie_id']].drop_duplicates()
        prev_starts['prev_start'] = 1

        # сохранение временной метки верхней границы подвыборки
        split_border = logs['datetime'].max()

        # предобработка логов
        df = preprocess(logs)

        # топ-100 популярных фильмов по числу посмотревших пользователей
        top100 = df['movie_id'].value_counts().index.tolist()[:100]

        # сохранение только положительных взаимодействий для дальнейших операций
        df = df[df['personal_score'] == 1]

        # конвертация датасета в удобный вид для обучения модели LightFM
        dataset = Dataset()
        dataset.fit(df['user_id'].unique(), df['movie_id'].unique())

        # создание матрицы взаимодействий
        interactions = dataset.build_interactions(
            df[['user_id', 'movie_id', 'personal_score']].values
            )[0]

        # создание мэппингов
        user_dict, movie_dict = dataset.mapping()[0], dataset.mapping()[2]
        inv_user_dict = {v: k for k, v in user_dict.items()}
        inv_movie_dict = {v: k for k, v in movie_dict.items()}

        # обучение модели LightFM
        model1.fit(interactions=interactions, epochs=10)
        del interactions

        # сохранение признаков пользователей и фильмов
        last_starts_df = get_last_start(logs)
        user_warmness_df = get_warmness_feature(df)
        genre_percentage_df = get_genre_percentage(df, movies)
        decade_percentage_df = get_decade_percentage(df, movies)
        combined_percentage_df = get_combined_percentage(df, movies)

        del df

        # матрица косинусного сходства по фильмам
        cos_sim_matrix = cosine_similarity(
            model1.item_embeddings, model1.item_embeddings
            )

        # перевод матрицы в датафрейм
        cos_sim_df = pd.DataFrame(
            index=list(movie_dict.values()),
            columns=list(movie_dict.values()),
            data=cos_sim_matrix,
            )

        del cos_sim_matrix

        cos_sim_df = cos_sim_df.stack().reset_index()
        cos_sim_df.rename(
            columns={
                'level_0': 'movie_id',
                'level_1': 'last_started_movie_id',
                0: 'cos_sim_to_last_start'
                },
            inplace=True,
            )

        # возвращение фильмам оригинальных id
        for col in ['movie_id', 'last_started_movie_id']:
            cos_sim_df[col] = cos_sim_df[col].map(inv_movie_dict)

        # список id пользователей и фильмов внутри датасета LightFM
        user_ids = list(user_dict.values())
        movie_ids = list(movie_dict.values())

        # создание датафрейма с фильмами-кандидатами
        candidates = pd.DataFrame({
            'user_id': user_ids,
            'rank': [np.arange(num_candidates).tolist()] * len(user_ids)
            })

        # добавление списка id рекомендаций от модели LightFM
        candidates['movie_id'] = candidates['user_id'].apply(
            lambda x: np.argsort(
                model1.predict(x, movie_ids)
                )[::-1][:num_candidates],
            )

        # кол-во партий, на которые разбивается датафрейм для экономии памяти
        num_batches = np.ceil(len(user_ids) / 50000)

        # последовательность операций с каждой партией данных
        for inds_chunk in np.array_split(candidates.index.values, num_batches):

            temp_set = candidates.loc[inds_chunk].copy()
            candidates.drop(index=inds_chunk, inplace=True)

            # перевод всех рекомендаций в вид отдельного элемента выборки и
            # замена всех id на оригинальные
            temp_set = temp_set.explode(['movie_id', 'rank'], ignore_index=True)
            temp_set['user_id'] = temp_set['user_id'].map(inv_user_dict)
            temp_set['movie_id'] = temp_set['movie_id'].map(inv_movie_dict)

            # удаление из выборки всех пар пользователь-фильм, которые ранее
            # встречались
            temp_set = temp_set.merge(
                prev_starts,
                on=['user_id', 'movie_id'],
                how='left',
                )

            temp_set = temp_set[temp_set['prev_start'].isna()]
            temp_set.drop(columns='prev_start', inplace=True)

            # добавление признака прошедшего времени от последнего просмотра
            # фильма
            temp_set = temp_set.merge(last_starts_df, on='user_id')
            temp_set['time_since_last_start'] = split_border - temp_set['last_start_datetime']
            temp_set['time_since_last_start'] = temp_set['time_since_last_start'].apply(
                lambda x: x.total_seconds()
                )
            temp_set.drop(columns=['last_start_datetime'], inplace=True)

            # добавление признака косинусного сходства между фильмом-кандидатом
            # и последним просмотренным фильмом
            temp_set = temp_set.merge(
                cos_sim_df,
                on=['movie_id', 'last_started_movie_id'],
                )

            temp_set.drop(columns='last_started_movie_id', inplace=True)

            # добавление признака "тёплый пользователь"
            temp_set = temp_set.merge(user_warmness_df, on='user_id')

            # добавление инфо о жанре фильма и декаде его выхода
            temp_set = temp_set.merge(
                movies[['movie_id', 'genres', 'decade']],
                on='movie_id',
                )

            # добавление признака доля просмотров фильмов в жанре
            # фильма-кандидата
            temp_set = temp_set.merge(
                genre_percentage_df,
                on=['user_id', 'genres'],
                how='left',
                ).fillna(0)

            # добавление признака доля просмотров фильмов с декадой выхода
            # фильма-кандидата
            temp_set = temp_set.merge(
                decade_percentage_df,
                on=['user_id', 'decade'],
                how='left',
                ).fillna(0)

            # добавление признака доля просмотров фильмов с декадой выхода и
            # жанром фильма-кандидата
            temp_set = temp_set.merge(
                combined_percentage_df,
                on=['user_id', 'genres', 'decade'],
                how='left',
                ).fillna(0)

            temp_set.drop(columns=['genres', 'decade'], inplace=True)

            # предсказание вероятности запуска фильма от CatBoostClassifier
            temp_set['proba'] = self.model2.predict_proba(
                temp_set[self.features]
                )[:,1]
            temp_set = temp_set[['user_id', 'movie_id', 'proba']]

            # сохранение топ-20 наиболее вероятных кандидатов в виде списка
            temp_set = temp_set.sort_values(
                by=['user_id', 'proba'],
                ascending=[True, False],
                )

            temp_set.drop(columns='proba', inplace=True)
            temp_set = temp_set.groupby('user_id').head(20)[['user_id', 'movie_id']]
            temp_set = temp_set.groupby('user_id', as_index=False).agg(list)

            # соединение датафреймов из разных партий
            if 'recs' in locals():
                recs = pd.concat([recs, temp_set], ignore_index=True)
            else:
                recs = temp_set.copy()

            del temp_set

        # освобождение памяти
        del last_starts_df, cos_sim_df, user_warmness_df, genre_percentage_df
        del decade_percentage_df, combined_percentage_df

        # холодные пользователи
        # они исчезли из тренировочной выборки после предобработки, потому что
        # у них нет ни одного просмотренного фильма
        # для них стандартная рекомендация - топ-100, из которых удаляются все
        # запускавшиеся ранее фильмы и отбираются топ-20
        cold_users = np.setdiff1d(
            logs['user_id'].unique(),
            list(user_dict.keys()),
            assume_unique=True,
            )

        cold_df = pd.DataFrame({
            'user_id': cold_users,
            'movie_id': [top100] * len(cold_users)
            })

        cold_df = cold_df.explode('movie_id', ignore_index=True)
        cold_df = cold_df.merge(
            prev_starts,
            on=['user_id', 'movie_id'],
            how='left',
            )

        cold_df = cold_df[cold_df['prev_start'].isna()]
        cold_df = cold_df.groupby('user_id').head(20)[['user_id', 'movie_id']]
        cold_df = cold_df.groupby('user_id', as_index=False).agg(list)

        # соединение датафреймов с рекомендациями для "тёплых" и "холодных"
        # пользователей
        recs = pd.concat([recs, cold_df], ignore_index=True)
        recs.rename(columns={'movie_id': 'movies'}, inplace=True)

        return recs

