import numpy as np
import torch.nn as nn
import torch.nn.functional as F


class Vocabulary(object):
    """Класс для обработки текста и извлечения словаря для сопоставления"""

    def __init__(self, token_to_idx=None, add_unk=True, unk_token="<UNK>"):
        """
        Args:
            token_to_idx (dict): уже существующий словарь токенов с индексами
            add_unk (bool): флаг, указывающий, следует ли добавлять токен UNK
            unk_token (str): токен UNK для добавления в словарь
        """

        if token_to_idx is None:
            token_to_idx = {}
        self._token_to_idx = token_to_idx

        self._idx_to_token = {idx: token 
                              for token, idx in self._token_to_idx.items()}
        
        self._add_unk = add_unk
        self._unk_token = unk_token
        
        self.unk_index = -1
        if add_unk:
            self.unk_index = self.add_token(unk_token) 
        
        
    def to_serializable(self):
        """ возвращает словарь, который можно сериализовать """
        return {'token_to_idx': self._token_to_idx, 
                'add_unk': self._add_unk, 
                'unk_token': self._unk_token}

    @classmethod
    def from_serializable(cls, contents):
        """ создает экземпляр словаря из сериализованного словаря """
        return cls(**contents)

    def add_token(self, token):
        """Обновить правила сопоставления на основе токена.

        Args:
            token (str): элемент, который нужно добавить в словарь
        Returns:
            index (int): целое число, соответствующее токену
        """
        try:
            index = self._token_to_idx[token]
        except KeyError:
            index = len(self._token_to_idx)
            self._token_to_idx[token] = index
            self._idx_to_token[index] = token
        return index
    
    def add_many(self, tokens):
        """Добавления списка токенов в словарь
        
        Args:
            tokens (list): список строковых токенов
        Returns:
            indices (list): список индексов, соответствующих токенам
        """
        return [self.add_token(token) for token in tokens]

    def lookup_token(self, token):
        """Получить индекс, связанный с токеном
           или индекс UNK, если токен отсутствует.
        
        Args:
            token (str): токен для поиска 
        Returns:
            index (int): индекс, соответствующий токену
        Notes:
            `unk_index` должен быть >=0 (добавлен в словарь)
                для функциональности UNK
        """
        if self.unk_index >= 0:
            return self._token_to_idx.get(token, self.unk_index)
        else:
            return self._token_to_idx[token]

    def lookup_index(self, index):
        """Вернуть токен, связанный с индексом
        
        Args: 
            index (int): индекс для поиска
        Returns:
            token (str): токен, соответствующий индексу
        Raises:
            KeyError: if the index is not in the Vocabulary
        """
        if index not in self._idx_to_token:
            raise KeyError("the index (%d) is not in the Vocabulary" % index)
        return self._idx_to_token[index]

    def __str__(self):
        return "<Vocabulary(size=%d)>" % len(self)

    def __len__(self):
        return len(self._token_to_idx)
    

class SurnameVectorizer(object):
    """ Векторизатор, который координирует словари и использует их """
    def __init__(self, surname_vocab, nationality_vocab):
        """
        Args:
            surname_vocab (Vocabulary): отображает символы в целые числа
            nationality_vocab (Vocabulary): отображает национальности в целые числа
        """
        self.surname_vocab = surname_vocab
        self.nationality_vocab = nationality_vocab

    def vectorize(self, surname):
        """
        Args:
            surname (str): фамилия

        Returns:
            one_hot (np.ndarray): свёрнутая one-hot кодировка 
        """
        vocab = self.surname_vocab
        one_hot = np.zeros(len(vocab), dtype=np.float32)
        for token in surname:
            one_hot[vocab.lookup_token(token)] = 1

        return one_hot

    @classmethod
    def from_dataframe(cls, surname_df):
        """Создание экземпляра векторизатора из датафрейма набора данных.
        
        Args:
            surname_df (pandas.DataFrame): датасет фамилий
        Returns:
            экземпляр SurnameVectorizer
        """
        surname_vocab = Vocabulary(unk_token="@")
        nationality_vocab = Vocabulary(add_unk=False)

        for index, row in surname_df.iterrows():
            for letter in row.surname:
                surname_vocab.add_token(letter)
            nationality_vocab.add_token(row.nationality)

        return cls(surname_vocab, nationality_vocab)

    @classmethod
    def from_serializable(cls, contents):
        surname_vocab = Vocabulary.from_serializable(contents['surname_vocab'])
        nationality_vocab =  Vocabulary.from_serializable(contents['nationality_vocab'])
        return cls(surname_vocab=surname_vocab, nationality_vocab=nationality_vocab)

    def to_serializable(self):
        return {'surname_vocab': self.surname_vocab.to_serializable(),
                'nationality_vocab': self.nationality_vocab.to_serializable()}
    

class SurnameClassifier(nn.Module):
    """ Двухслойный многослойный персептрон для классификации фамилий """
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        Args:
            input_dim (int): размер входных векторов
            hidden_dim (int): выходной размер первого линейного слоя
            output_dim (int): выходной размер второго линейного слоя
        """
        super(SurnameClassifier, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        """Прямой проход классификатора
        
        Args:
            x_in (torch.Tensor): тензор входных данных. 
                x_in.shape должен быть (batch, input_dim)
            apply_softmax (bool): флаг активации softmax должен быть False, 
            если используется с функцией потерь в виде перекрестной энтропии
        Returns:
            возвращает tensor. tensor.shape должен быть (batch, output_dim)
        """
        intermediate_vector = F.relu(self.fc1(x_in))
        prediction_vector = self.fc2(intermediate_vector)

        if apply_softmax:
            prediction_vector = F.softmax(prediction_vector, dim=1)

        return prediction_vector