import json
import torch
import pickle
import numpy as np
import torch.utils.data as data
import h5py


class Dictionary(object):
    def __init__(self):
        self.words = [None]
        self.tokens = {None: 0}

    def __len__(self):
        return len(self.tokens)

    @property
    def padding_idx(self):
        return 0

    def tokenize(self, sentence, add_word=False):
        tokens = []
        words = sentence.lower().replace(',', '').replace('?', '').replace('\'s', '').split()
        for w in words:
            if w not in self.tokens:
                if not add_word:
                    return None
                self.words.append(w)
                self.tokens[w] = len(self)
            tokens.append(self.tokens[w])
        return tokens

    def save(self, path):
        torch.save((self.tokens, self.words), path)

    def load(self, path):
        self.tokens, self.words = torch.load(path)


class VQA_Dataset(data.Dataset):
    def __init__(self, cache_path, set_type):
        super(VQA_Dataset, self).__init__()
        self.entries = []
        self.q_dict = None
        self.ans2label = None
        self.process_data(cache_path, set_type)

    def process_data(self, cache_path, set_type):
        self.q_dict = Dictionary()
        self.q_dict.load(cache_path / 'dict.pkl')
        self.img_dict = pickle.load((cache_path / f'{set_type}_img_dict.pkl').open('rb'))
        self.img_data = h5py.File(cache_path / f'{set_type}_img.hdf5', 'r')['images']
        self.ans2label = pickle.load((cache_path / f'trainval_ans2label.pkl').open('rb'))

        questions_file = '/datashare/' f'v2_OpenEnded_mscoco_{set_type}2014_questions.json'
        with open(questions_file) as f:
            questions = json.load(f)['questions']

        questions = sorted(questions, key=lambda q: q['question_id'])

        answers = pickle.load((cache_path / f'{set_type}_target.pkl').open('rb'))
        answers = sorted(answers, key=lambda x: x['question_id'])

        for question, answer in zip(questions, answers):
            img_id = question['image_id']
            question = {'question_id': question['question_id'], 'question': question['question']}

            q_tokens = self.q_dict.tokenize(question['question'], False)
            q_tokens = q_tokens[:14]
            if len(q_tokens) < 14:
                padding = [self.q_dict.padding_idx] * (14 - len(q_tokens))
                q_tokens = padding + q_tokens

            q_tokens = torch.from_numpy(np.array(q_tokens))
            labels = np.array(answer['labels'])
            scores = np.array(answer['scores'], dtype=np.float32)

            if len(labels):
                labels = torch.from_numpy(labels)
                scores = torch.from_numpy(scores)
                answer['labels'] = labels
                answer['scores'] = scores
            else:
                answer['labels'] = None
                answer['scores'] = None

            self.entries.append((self.img_dict[img_id], q_tokens, answer['labels'], answer['scores']))

    def __getitem__(self, item):
        img_idx, question, answers, scores = self.entries[item]
        img = torch.tensor(self.img_data[img_idx], dtype=torch.float)
        annotation = torch.zeros(len(self.ans2label))
        if answers is not None:
            annotation.scatter_(0, answers, scores)
        return img, question, annotation

    def __len__(self):
        return len(self.entries)