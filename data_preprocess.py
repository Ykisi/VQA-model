import pickle
import json
import numpy
import h5py
from pathlib import Path
from PIL import Image
from Dataset import Dictionary


def question_dict(cache_path):
    q_dict = Dictionary()

    questions_file = '/datashare/v2_openended_mscoco_train2014_questions.json'
    with open(questions_file) as f:
        questions = json.load(f)['questions']

    for question in questions:
        q_dict.tokenize(question['question'], True)

    questions_file = '/datashare/v2_openended_mscoco_val2014_questions.json'
    with open(questions_file) as f:
        questions = json.load(f)['questions']

    for question in questions:
        q_dict.tokenize(question['question'], True)

    q_dict.save(cache_path / 'dict.pkl')


def image_process(data_path, cache_path, set_type):

    print(f'cache {set_type}')
    images = data_path / f'{set_type}2014'
    n_images = len(list(images.glob('*')))
    img_dict = {}
    with h5py.File(cache_path / f'{set_type}_img.hdf5', 'w') as h5:
        img_data = h5.create_dataset('images', shape=(n_images, 3, 64, 64), dtype='i')
        for i, image in enumerate(images.glob('*')):
            img_id = int(image.name.replace('.jpg', '')[-12:])
            img_dict[img_id] = i
            img = numpy.array(Image.open(image).resize((64, 64)).convert('RGB'))
            img_data[i, :] = img.reshape((3, 64, 64))
    pickle.dump(img_dict, Path(cache_path / f'{set_type}_img_dict.pkl').open('wb'))