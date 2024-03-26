import numpy as np
import pandas as pd


def map_at_20(preds: pd.Series, labels: pd.Series):
    """
    Расчёт метрики MAP@20.

    Args:
        preds:
            Ранжированный список рекомендаций.
        labels:
            Список просмотренных фильмов.
    """
    df = pd.concat([preds.to_frame(), labels.to_frame()], axis=1)
    df['true_recs'] = df.apply(
        lambda x: np.isin(x['movies'], x['movie_id'], assume_unique=True),
        axis=1,
        )
    df['cum_true_recs'] = df['true_recs'].apply(lambda x: x.cumsum())
    df['cum_true_recs'] *= df['true_recs']
    df['cum_true_recs'] = df['cum_true_recs'].apply(
        lambda x: x / np.arange(1, 21)
        )
    df['ap_to_k'] = df['cum_true_recs'].apply(np.mean)
    print(f'MAP@20: {df["ap_to_k"].mean() - 0:.5f}')


def split_by_time(data: pd.DataFrame, periods: list):
    """
    Делит выборку по заданным периодам.

    Args:
        data:
            Датафрейм с логами.

        periods:
            Список длительности последовательных периодов, которые включают логи.

    Returns:
        indices:
            Списки индексов по числу заданных периодов.
    """
    # время от первого лога до окончания каждого из заданных периодов
    periods_cumsum = np.cumsum([pd.Timedelta(period) for period in periods])

    # граница каждого периода - временная отметка завершения периода
    periods_borders = [data['datetime'].min() + t for t in periods_cumsum]
    time_borders = {'border_0': data['datetime'].min()}
    time_borders |= {f'border_{n+1}': v for n, v in enumerate(periods_borders)}

    # список индексов, соответствующих каждому из заданных периодов
    indices = [data[
        (data['datetime'] >= time_borders[f'border_{border_num}']) &
        (data['datetime'] < time_borders[f'border_{border_num + 1}'])
        ].index.tolist() for border_num in range(len(time_borders) - 1)]

    return indices


def get_approx_duration(data: pd.DataFrame):
    """
    Приближенная оценка длительности фильма по логам пользователей.

    Args:
        data:
            Датафрейм с логами.

    Returns:
        duration:
            Данные в виде пары id-длительность.
    """
    df = data.copy()
    df['duration'] = df['duration'] // 60 * 60
    df = df.groupby('movie_id', as_index=False)['duration'].apply(
        lambda x: x[x >= x.quantile(q=0.75)].mode().mean()
        )

    return df


def preprocess(data: pd.DataFrame, max_allowed_gap: str = '1 day'):
    """
    Предобработка данных.

    Args:
        data:
            Датафрейм с логами.

        max_allowed_gap:
            Максимально допустимая разница во времени между концом и началом
            последовательно идущих логов, относящихся к одной паре пользователь-
            фильм, при котором эти логи относятся к одному сеансу просмотра.

    Returns:
        df:
            Датафрейм в виде уникальных пар пользователь-фильм с отметкой о том,
            является ли это взаимодействие отрицательным (0) или положительным
            (1).
    """
    gap = pd.Timedelta(max_allowed_gap).total_seconds()

    df = data.copy()
    std_cols = df.columns

    # Набор данных делится на два под-набора:
    #   набор, где паре пользователь-фильм соответствует один лог;
    #   набор, где паре пользователь-фильм соответствует больше одного лога.
    # "Схлопывание" логов касается только второго под-набора
    df['is_duplicated'] = df.duplicated(['user_id', 'movie_id'], keep=False)
    df_p1 = df[~df['is_duplicated']].copy()
    df_p2 = df[df['is_duplicated']].copy()
    del df

    # Разница во времени в секундах между этим и предыдущим логами,
    # относящимися к одной паре пользователь-фильм
    temp_df = df_p2[['datetime', 'user_id', 'movie_id']]
    temp_df = temp_df.assign(prev_event_dt=temp_df.datetime)

    df_p2 = pd.merge_asof(
        df_p2,
        temp_df,
        on='datetime',
        by=['user_id', 'movie_id'],
        allow_exact_matches=False,
        direction='backward',
        )

    del temp_df

    df_p2['time_diff'] = df_p2['datetime'] - df_p2['prev_event_dt']
    df_p2['time_diff'] = df_p2['time_diff'].apply(lambda x: x.total_seconds())

    # Указывает на то, относятся ли этот и предыдущий лог из одной пары
    # пользователь-фильм к одному сеансу просмотра видео-контента
    df_p2['is_not_break'] = df_p2['duration'] < (df_p2['time_diff'] - gap)

    # Индекс сеанса просмотра фильма пользователем, к которому относится лог
    session_num = df_p2.groupby(
        ['user_id', 'movie_id']
        )['is_not_break'].cumsum()
    session_num.name = 'session_num'
    df_p2 = pd.concat([df_p2, session_num], axis=1)

    # Длительность сеанса просмотра видео-контента пользователем
    cum_duration = df_p2.groupby(
        ['user_id', 'movie_id', 'session_num'],
        as_index=False,
        )['duration'].sum()
    cum_duration.rename(columns={'duration': 'cum_duration'}, inplace=True)
    df_p2 = df_p2.merge(cum_duration, on=['user_id', 'movie_id', 'session_num'])

    # Для каждого сеанса просмотра сохраняется только последний лог
    df_p2.drop_duplicates(
        ['user_id', 'movie_id', 'session_num'],
        keep='last',
        inplace=True,
        )

    # Длительность просмотра по логу заменяется длительностью сеанса просмотра
    df_p2['duration'] = df_p2['cum_duration'].copy()

    # Два под-набора данных обратно конкатенируются
    df = pd.concat([df_p1[std_cols], df_p2[std_cols]], ignore_index=True)
    del df_p1, df_p2

    # вычисление приблизительной длительности фильма и добавление этой
    # информации в датафрейм
    duration_df = get_approx_duration(df)
    df = df.merge(duration_df, on='movie_id', suffixes=('_watching', '_movie'))

    # для каждой уникальной пары пользователь-фильм сохраняется лишь один лог
    # с максимальной длительностью сеанса
    df.sort_values(['user_id', 'movie_id', 'duration_watching'], inplace=True)
    df = df.groupby(['user_id', 'movie_id']).tail(1)

    # присвоение логу оценки просмотра:
    #     0 - фильм не понравился или его запуск - случайность/ошибка
    #         если длительность просмотра < 10% от длительности фильма
    #     1 - фильм понравился или запуск фильма неслучайный
    #         если длительность просмотра >= 10% от длительности фильма
    df['personal_score'] = df['duration_watching'] >= df['duration_movie'] * 0.1
    df['personal_score'] = df['personal_score'].map({False: 0, True: 1})

    return df[['user_id', 'movie_id', 'personal_score']]


def get_last_start(data: pd.DataFrame):
    """
    Возвращает последний лог каждого пользователя.

    Args:
        data:
            Датафрейм с логами.
    """
    df = data.copy()

    # отбор последнего по хронологии лога для каждого пользователя
    df.sort_values(by=['user_id', 'datetime'], inplace=True)
    df = df.groupby('user_id').tail(1)
    df.rename(
        columns={
            'movie_id': 'last_started_movie_id',
            'datetime': 'last_start_datetime'
            },
        inplace=True,
        )

    return df[['user_id', 'last_started_movie_id', 'last_start_datetime']]


class Pipeline(object):
    """Пайплайн."""

    def __init__(self, model):
        """Создание объекта класса."""
        self.model = model

    def run(self, logs: pd.DataFrame, movies: pd.DataFrame, mode: str):
        """
        Запуск пайплайна.

        Args:
            logs:
                Пользовательские логи запуска фильма.

            movies:
                Датафрейм с информацией по каждому фильму.

            mode:
                Режим запуска пайплайна:
                - 'eval': разбиение выборки на тренировочную и валидационную
                части, обучение модели на тренировочной выборке и оценка
                качества ранжирования на валидационной выборке.
                - 'rec': обучение модели на всей выборке и сохранение
                предсказаний.
        """
        if mode not in ['eval', 'rec']:
            raise ValueError("mode param must be 'eval' or 'rec'")

        # пайплайн в режиме оценки модели
        if mode == 'eval':

            # тренировочная выборка - логи за первые 50 дней
            # валидационная выборка - логи за следующие 20 дней
            train_inds, val_inds = split_by_time(
                data=logs,
                periods=['50 days', '20 days'],
                )

            # тренировочная выборка делится на две под-выборки:
            # - логи за первые 30 дней - для обучения LightFM;
            # - логи за следующие 20 дней - для извлечения таргетов для
            #   CatBoostClassifier
            train1_inds, train2_inds = split_by_time(
                data=logs.loc[train_inds],
                periods=['30 days', '20 days'],
                )

            # обучение модели
            self.model.fit(
                df1=logs.loc[train1_inds],
                df2=logs.loc[train2_inds],
                movies=movies,
                )

            # оценка MAP@20
            self.model.evaluate(
                train_logs=logs.loc[train_inds],
                val_logs=logs.loc[val_inds],
                movies=movies,
                )

        # пайплайн в режиме создания рекомендаций
        elif mode == 'rec':

            # вся выборка делится на две под-выборки:
            # - логи за первые 40 дней - для обучения LightFM;
            # - логи за следующие 30 дней - для извлечения таргетов для
            #   CatBoostClassifier
            train1_inds, train2_inds = split_by_time(
                data=logs,
                periods=['40 days', '30 days'],
                )

            # обучение модели
            self.model.fit(
                df1=logs.loc[train1_inds],
                df2=logs.loc[train2_inds],
                movies=movies,
                )

            # вывод предсказаний в виде датафрейма
            recs = self.model.predict(logs, movies)

            return recs

