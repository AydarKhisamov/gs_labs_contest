import os
import pandas as pd
import shutil
from models import TwoStageModel
from utils import Pipeline

if __name__ == '__main__':

    # чтение данных
    logs = pd.read_csv('train/logs.csv')
    movies = pd.read_csv('train/movies.csv')

    # конвертация типов данных
    logs['datetime'] = pd.to_datetime(logs['datetime'])
    logs['movie_id'] = logs['movie_id'].astype(int)

    movies['year'] = movies['year'].str[:4]
    movies['decade'] = movies['year'].apply(lambda x: x[:3] + '0s')
    movies.rename(columns={'id': 'movie_id'}, inplace=True)

    # гиперпараметры модели LightFM
    lfm_params = {'no_components': 64, 'loss': 'warp', 'random_state': 42}

    # гиперпараметры модели CatBoostClassifier
    gb_params = {
        'iterations': 5000, 'use_best_model': True,
        'auto_class_weights': 'Balanced', 'task_type': 'GPU', 'random_state': 42,
        'early_stopping_rounds': 500,
        }

    # создание папки для сохранения результатов выполнения этого скрипта
    os.mkdir('output')

    # запуск пайплайна в режиме создания рекомендаций
    pipe = Pipeline(model=TwoStageModel(lfm_params, gb_params))
    predictions = pipe.run(logs, movies, mode='rec')

    # сохранение предсказаний
    predictions.to_csv('output/result.csv', index=False, header=False)

