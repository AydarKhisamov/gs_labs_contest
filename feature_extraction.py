import pandas as pd


def get_warmness_feature(data: pd.DataFrame):
    """
    Каждому пользователю присваивает признак "тёплый пользователь".

    Args:
        data:
            Предобработанный датасет.
    """
    df = data.copy()

    # вычисление кол-ва положительных взаимодействий для пользователя
    df = df.groupby('user_id', as_index=False)['personal_score'].sum()

    # в зависимости от кол-ва положительных взаимодействий присваивается признак
    df['warm_user'] = df['personal_score'] >= 5
    df['warm_user'] = df['warm_user'].map({False: 0, True: 1})

    return df[['user_id', 'warm_user']]


def get_genre_percentage(data: pd.DataFrame, movies: pd.DataFrame):
    """
    Доля всех (в том числе сочетаний) жанров в истории положительных
    взаимодействий всех пользователей.

    Args:
        data:
            Предобработанный датасет.

        movies:
            Датафрейм с информацией по каждому фильму.
    """
    df = data[['user_id', 'movie_id']].copy()

    # присоединение к каждому фильму информации о жанре
    df = df.merge(movies[['movie_id', 'genres']], on='movie_id')

    # кол-во фильмов по каждому (сочетанию) жанру для каждого пользователя
    df1 = df.groupby(['user_id', 'genres'], as_index=False).count()
    df1.rename(columns={'movie_id': 'movies_per_genre'}, inplace=True)

    # кол-во фильмов у каждого пользователя
    df2 = df.groupby('user_id', as_index=False)['movie_id'].count()
    df2.rename(columns={'movie_id': 'movies_per_user'}, inplace=True)

    # вычисление доли каждого (сочетания) жанра от всех просмотров
    df1 = df1.merge(df2, on='user_id')
    df1['genre_percent'] = df1['movies_per_genre'] / df1['movies_per_user']

    return df1[['user_id', 'genres', 'genre_percent']]


def get_decade_percentage(data: pd.DataFrame, movies: pd.DataFrame):
    """
    Доля фильмов, вышедших в каждую декаду в истории положительных
    взаимодействий всех пользователей.

    Args:
        data:
            Предобработанный датасет.

        movies:
            Датафрейм с информацией по каждому фильму.
    """
    df = data[['user_id', 'movie_id']].copy()

    # присоединение к каждому фильму информации о декаде
    df = df.merge(movies[['movie_id', 'decade']], on='movie_id')

    # кол-во фильмов, вышедших в каждую декаду для каждого пользователя
    df1 = df.groupby(['user_id', 'decade'], as_index=False).count()
    df1.rename(columns={'movie_id': 'movies_per_decade'}, inplace=True)

    # кол-во фильмов у каждого пользователя
    df2 = df.groupby('user_id', as_index=False)['movie_id'].count()
    df2.rename(columns={'movie_id': 'movies_per_user'}, inplace=True)

    # вычисление доли фильмов, вышедших в каждую декаду, от всех просмотров для
    # каждого пользователя
    df1 = df1.merge(df2, on='user_id')
    df1['decade_percent'] = df1['movies_per_decade'] / df1['movies_per_user']

    return df1[['user_id', 'decade', 'decade_percent']]


def get_combined_percentage(data: pd.DataFrame, movies: pd.DataFrame):
    """
    Доля фильмов по каждому возможному сочетанию жанра и декады выхода фильма в
    истории положительных взаимодействий всех пользователей.

    Args:
        data:
            Предобработанный датасет.

        movies:
            Датафрейм с информацией по каждому фильму.
    """
    df = data[['user_id', 'movie_id']].copy()

    # присоединение к каждому фильму информации о жанре и декаде выхода
    df = df.merge(movies[['movie_id', 'genres', 'decade']], on='movie_id')

    # кол-во фильмов с любым возможным сочетанием жанра и декады выхода для
    # каждого пользователя
    df1 = df.groupby(['user_id', 'genres', 'decade'], as_index=False).count()
    df1.rename(columns={'movie_id': 'movies_per_pair'}, inplace=True)

    # кол-во фильмов у каждого пользователя
    df2 = df.groupby('user_id', as_index=False)['movie_id'].count()
    df2.rename(columns={'movie_id': 'movies_per_user'}, inplace=True)

    # вычисление доли фильмов по всем сочетаниям жанра и декады выхода от всех
    # просмотров для каждого пользователя
    df1 = df1.merge(df2, on='user_id')
    df1['pair_percent'] = df1['movies_per_pair'] / df1['movies_per_user']

    return df1[['user_id', 'genres', 'decade', 'pair_percent']]

