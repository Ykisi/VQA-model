from train import train
from pathlib import Path
from data_preprocess import question_dict, image_process


data_path = Path('/datashare/')
cache_path = Path('./')


def main():

    cache_path.mkdir(exist_ok=True)
    question_dict(cache_path)
    image_process(data_path, cache_path, 'train')
    image_process(data_path, cache_path, 'val')
    train(cache_path)


if __name__ == '__main__':
    main()