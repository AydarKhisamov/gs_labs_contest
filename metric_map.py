"""
Скрипт для локальной проверки данных, такой же алгоритм работает на ci cd,
при тегировании ваших проектов.

Заполните PREDICT_FILE,TEST_LOG_FILE,TRAIN_LOG_FILE путями до нужных файлов.
LIMIT_VALUE - нужен только для того, чтобы проверить на малом количестве строк

TEST_LOG_FILE и TRAIN_LOG_FILE - должны быть в том же формате, что и файл выданный ранее logs.csv.


Запустить можно следующими вариантами:
1. Сделать исполняемым через Dockerfile
2. Сделать так, чтобы контейнер ушел в sleep, зайти на него и запустить скрипт
3. Локально поставить необходимые компоненты на вашу систему
"""
import pandas as pd
import json

PREDICT_FILE: str = "output/result.csv" # файл с предсказаниями рекомендаций
TEST_LOG_FILE: str = "train/split_train_2.csv" # файл с тестовой выборкой
TRAIN_LOG_FILE: str = "train/split_train_1.csv" # файл с обучающей выборкой
LIMIT_ROW: int = 0 # лимит проверенных строчек

def check_user_predict_data(row: pd.Series) -> float:
    number_predict: int = 1
    tp: int = 1
    user_result: int = 0

    train_log_movies: list = list(
        train_logs_df[
            train_logs_df['user_id'] == row['user_id']
            ]['movie_id'].values
    )
    test_logs_movies: list = list(
        test_logs_df[
            test_logs_df['user_id'] == row['user_id']
            ]['movie_id'].values
    )
    predict_movies: list = json.loads(row['predict'])
    for predict_movie in predict_movies:
        if predict_movie in test_logs_movies:
            if predict_movie not in train_log_movies:
                user_result += tp / number_predict
                tp += 1
        number_predict += 1
    return user_result / len(predict_movies)

if __name__ == '__main__':
    print('START calculate')
    predict_df: pd.DataFrame = pd.read_csv(
        PREDICT_FILE,
        names=['user_id', 'predict']
    )
    train_logs_df: pd.DataFrame = pd.read_csv(TRAIN_LOG_FILE)
    test_logs_df: pd.DataFrame = pd.read_csv(TEST_LOG_FILE)

    user_num: int = 0
    list_result: list = []
    index: int
    predict_row: pd.Series
    for index, predict_row in predict_df.iterrows():
        user_num += 1
        if user_num > LIMIT_ROW and LIMIT_ROW:
            break
        list_result.append(check_user_predict_data(predict_row))
        print(f'{user_num}/{predict_df.shape[0]}')
    result_mectic_value: float = sum(list_result) / len(train_logs_df['user_id'].value_counts())
    print('FINISH calculate')
    print(f'RESULT: {round(result_mectic_value, 8)}')
