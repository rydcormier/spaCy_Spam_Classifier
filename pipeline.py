"""A text processing pipeline class."""

import re
import string
from tqdm import tqdm
import nltk.corpus
from bs4 import BeautifulSoup
from spellchecker import SpellChecker
import spacy
from spacy.lookups import Lookups
from spacy.lemmatizer import Lemmatizer


class Pipeline(object):
    """TODO: doc string"""
    builtin_pipes = ('to_lower', 'html_cleaner', 'punct_remover',
                      'spell_check', 'stopword_remover', 'lemmatizer')
    stopwords = nltk.corpus.stopwords.words('english')
    spell = SpellChecker()

    def __init__(self, include=None, exclude=None, **kwargs):
        """
        Create a Pipeline.

        Arguments:
            include -- A list of pipes to include. If None, then exclude
                       must be defined.
            exclude -- A list of pipes to exclude.
            lookups -- (Optional) If adding a lemmatizer,
                       a spaCy Lookups object is needed.
        """
        self.pipeline = []
        if include is not None:
            names = [name for name in self.__class__.builtin_pipes
                        if name in include]
        elif exclude is not None:
            names = [name for name in self.__class__.builtin_pipes
                        if name not in exclude]
        else:
            raise ArgumentError('Both `include` and `exclude` cannot be None.')

        for name in names:
            self.pipeline.append((name, self.create_pipe(name, **kwargs)))


    def __call__(self, text):
        words = None
        for name, proc in self.pipeline:
            if name in ('html_cleaner', 'to_lower', 'punct_remover'):
                # pass as str
                if words is not None:
                    text = self._join_words(text, words)
                    words = None
                text = proc(text)
            else:
                # pass as words
                if words is None:
                    words = self._split_words(text)
                words = proc(words)
        if words is not None:
            text = self._join_words(text, words)
        return text

    @property
    def pipe_names(self):
        return [name for name, _ in self.pipeline]

    def add_pipe(self, proc, name=None, last=True, first=False, before=None):
        """Add a custom pipe."""
        name = name if name is not None else f'{proc}'
        if last:
            self.pipeline.append((name, proc))
        elif first:
            self.pipeline.insert(0, (name, proc))
        else:
            if before not in self.pipe_names:
                raise ArgumentError('No valid position given.')
            for i, name in self.pipe_names:
                if name == before:
                    self.pipeline.insert(i, (name, proc))
                    break

    def create_pipe(self, name, **kwargs):
        """Create a builtin pipe."""
        if name not in self.__class__.builtin_pipes:
            raise ValueError('Invalid pipe name: {}'.format(pipe_name))
        if name == 'html_cleaner':
            return lambda text: BeautifulSoup(text).get_text()
        elif name == 'to_lower':
            return lambda text: text.lower()
        elif name == 'spell_check':
            return lambda words: [(self.__class__.spell.correction(w), s)
                                    for w, s in words]
        elif name == 'stopword_remover':
            filter = lambda word: word if word not in self.__class__.stopwords else ''
            return lambda words: [(filter(w), s) for w, s in words]
        elif name == 'punct_remover':
            trans_table = str.maketrans({c:None for c in string.punctuation})
            return lambda text: text.translate(trans_table)
        elif name == 'lemmatizer':
            assert 'lookups' in kwargs
            lemmatizer = Lemmatizer(kwargs['lookups'])
            return lambda words: [(lemmatizer.lookup(w), s) for w, s in words]


    def pipe(self, *texts, batch_size=1000, as_tuples=True, **kwargs):
        """Process multiple texts in batches."""
        batch = []
        for text in texts:
            if as_tuples:
                text, context = text
            if len(batch) >= batch_size:
                for obj in batch:
                    yield obj
                batch = []
            text = self(text)
            if as_tuples:
                text = text, context
            batch.append(text)
        for obj in batch:
            yield obj

    def _split_words(self, text):
        """
        Split the text into a list of tuples-(str, (int, int))-containing
        the word as a string and the starting and ending positions of the word
        in text.
        """
        res = []
        for m in re.finditer(r'(\w+)', text):
            a, b = m.span()
            res.append((text[a:b], (a, b)))
        return res

    def _join_words(self, text, words):
        """
        For each word and span given, replace the contents of the text span
        with the word.
        """
        res = ''
        pos = 0
        for word, span in words:
            res += text[pos:span[0]] + word
            pos = span[1]
        return res
