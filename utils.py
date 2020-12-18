"""Utility functions for a spam classifier using spaCy."""

import json
import os
import random
import tarfile
import sys
import time

from email.parser import BytesParser, Parser
from urllib.request import urlopen
from tqdm import tqdm

from spacy.util import compounding, minibatch

if not sys.warnoptions:
    import warnings
    warnings.simplefilter('ignore')


###########################################################################
# Corpus Utils
###########################################################################
URL         = 'https://spamassassin.apache.org/old/publiccorpus/'
FILES       = [ '20030228_easy_ham.tar.bz2',
                '20030228_easy_ham_2.tar.bz2',
                '20030228_hard_ham.tar.bz2',
                '20030228_spam.tar.bz2',
                '20050311_spam_2.tar.bz2' ]
CATEGORIES = [ 'easy_ham',
               'easy_ham_2',
               'hard_ham',
               'spam',
               'spam_2' ]

def download_and_extract_corpus(path='./corpus', silent=False):
    """
    Download the corpus archives and extract to disk.

    Arguments:
        path   -- (Optional) Target directory to extract to. Defualts to a new
                  directory `corpus` in the current working directory.
        silent -- (Optional) Supress all output.

    Returns:
        list -  Paths of all extracted message files.
    """
    files = []
    for file in FILES:
        if not silent:
            print('\nDownloading {}...'.format(file))

        with urlopen(f'{URL}{file}') as stream:

            with tarfile.open(fileobj=stream, mode='r|bz2') as tfile:
                it = tfile
                if not silent:
                    it = tqdm(it, desc='Extracting', unit=' file')
                for member in it:
                    try:
                        tfile.extract(member, path=path)
                        if member.name not in CATEGORIES and \
                                'cmds' not in member.name:
                            files.append(f'{path}/{member.name}')
                    except Exception as e:
                        print('Error: {}'.format(e))
    return files


###########################################################################
# Email Message Utils
###########################################################################
def extract_message_content(msg):
    """
    Extract email headers and message body.

    Arguments:
        msg -- An email.message.Message object.

    Returns:
        tuple - The message headers and body.
    """
    headers = {str(k).lower():str(v) for k, v in msg.items()}

    if not msg.is_multipart():
        body = msg.get_payload()
    else:
        plain, html, other = None, None, None
        for part in [p for p in msg.walk() if not p.is_multipart()]:
            ctype = part.get_content_type()

            if 'text/plain' in ctype:
                plain = part.get_payload()
            elif ctype == 'text/html' or ctype == 'mixed/alternative':
                html = part.get_payload()
            else:
                other = part.get_payload()
        if plain:
            body = plain
        elif html:
            body = html
        else:
            body = other
    return headers, body


def format_message_content(headers, body, header_fields=None):
    """
    Format the email message content into a single string.

    Arguments:
        headers -- A dict object mapping the message headers.
        body -- The message body as a string.
        header_fields -- (Optional) Only use the header fields listed, or
                         exclude headers if an empty container is passed. The
                         default is to include all headers.

    Returns:
        str -- The formatted message.
    """
    if header_fields is None:
        header_fields = headers.keys()

    res = ''

    for h in header_fields:
        v = headers.get(h, '')
        if v:
            res += f'{h}: {v}\n'

    res += body
    return res


def parse_message_files(files,
                        batch_size=None,
                        formatter=format_message_content,
                        **kwargs):
    """
    Parse the email message files and return the and content and label.

    Arguments:
        files      -- The message files to parse.
        batch_size -- (Optional) Parse files in batches.
        formatter  -- (Optional) Callable object to format message content.

    Returns:
        list -- Message content and label tuples for the given files.
    """
    res = []
    parser = Parser()
    bytes_parser = BytesParser()

    # generate email.message.Message objects
    def _gen_messages():
        batch = []
        for file in files:
            try:
                with open(file) as fp:
                    msg = parser.parse(fp)
            except UnicodeDecodeError:
                with open(file, 'rb') as fp:
                    msg = bytes_parser.parse(fp)
            if batch_size:
                if len(batch) >= batch_size:
                    for obj in batch:
                        yield obj
                    batch = []
                batch.append(msg)
            else:
                yield msg
        for obj in batch:
            yield obj

    i = 0
    for msg in _gen_messages():
        label = 'spam' in files[i]  # It's spam if it's in a spam directory
        i += 1
        headers, body = extract_message_content(msg)
        content = formatter(headers, body)
        res.append((content, label))
    return res


def preprocess(files,
               pipeline=None,
               labeler=None,
               silent=False,
               **kwargs):
    """
    Preprocess the message text.

    Arguments:
        files     -- File paths to the text files to preprocess.
        pipeline  -- (Optional) Callable object to process the message content.
        labeler   -- (Optional) Callable to format the context.
        silent    -- (Optional) Supress all output.
        **kwargs  -- Any additional keyword arguments are passed to the parser.

    Returns:
        list -- The message text and labels.
    """
    res = []

    if silent:
        it = parse_message_files(files, **kwargs)
    else:
        it = tqdm(parse_message_files(files, **kwargs),
                  desc='Processing', unit='file')

    for content, context in it:
        if pipeline:
            content = pipeline(content)
        if labeler:
            context = labeler(context)
        res.append((content, context))
    return res


###########################################################################
# spaCy Utils
###########################################################################

def train_textcat(nlp,
                  train_data,
                  epochs,
                  drop,
                  dev_data=None,
                  silent=False):
    """
    Train a spaCy text classifier.

    Arguments:
        nlp           -- A spaCy Language object with a `textcat` pipe.
        train_data    -- The texts and annotations to train with.
        epochs        -- Number of training iterations.
        drop          -- The dropout rate.
        dev_data      -- (Optional) Additional texts and annotations for
                         tracking extended metrics.
        silent        -- (Optional) Supress all output.

    Returns:
        tuple -- The model, the training metrics (a dict storing `f_score`,
                 `accuracy`, `loss`, and `time`), and the learned parameters.
    """
    metrics = {'f_score': [], 'accuracy': [], 'loss': [], 'time': 0.0}
    if dev_data is not None:
        dev_texts, dev_golds = zip(*dev_data)

    other_pipes = [p for p in nlp.pipe_names if p != 'textcat']
    textcat = nlp.get_pipe('textcat')

    with nlp.disable_pipes(*other_pipes):
        opt = nlp.begin_training()
        batch_sizes = compounding(4.0, 32.0, 1.001)
        it = range(1, epochs + 1)
        if not silent:
            it = tqdm(it, desc='Training', unit=' epoch')

        for i in it:
            start = time.time()
            losses = {}
            random.shuffle(train_data)

            for batch in minibatch(train_data, size=batch_sizes):
                texts, golds = zip(*batch)
                nlp.update(texts, golds, sgd=opt, drop=drop, losses=losses)

            metrics['loss'].append(losses['textcat'])
            metrics['time'] += time.time() - start

            if dev_data is not None:
                with textcat.model.use_params(opt.averages):
                    scores = evaluate(nlp.tokenizer, textcat, dev_texts, dev_golds)
                metrics['f_score'].append(scores['f_score'])
                metrics['accuracy'].append(scores['accuracy'])

    return nlp, metrics, opt.averages




def evaluate(tokenizer, textcat, texts, cats):
    """
    Evaluate the performance of a text classifier.

    Arguments:
        tokenizer -- A callable object to tokenize texts.
        textcat   -- The text categorizer.
        texts     -- The texts to categorize.
        cats      -- The labels for the texts.

    Returns:
        A dict mapping the evaluation metrics `precision`, `recall`, `f_score`,
        and `accuracy`.
    """
    docs = (tokenizer(text) for text in texts)
    tp = 0.0  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 0.0  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if label == "NOT_SPAM":
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.0
            elif score >= 0.5 and gold[label] < 0.5:
                fp += 1.0
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 and gold[label] >= 0.5:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    accuracy = (tp + tn) / len(texts)
    if (precision + recall) == 0:
        f_score = 0.0
    else:
        f_score = 2 * (precision * recall) / (precision + recall)
    return {"precision": precision, "recall": recall, "f_score": f_score,
            "accuracy": accuracy}


def update_metrics_axes(axes, data, **kwargs):
    assert len(axes) == len(data)
    n = len(axes)
    fontdict = kwargs.get('fontdict', {'fontsize': 14, 'fontstyle': 'italic'})
    xticks = kwargs.get('xticks', [None] * n)
    yticks = kwargs.get('yticks', [None] * n)
    yscale = kwargs.get('yscale', [None] * n)
    colors = kwargs.get('colors', [None] * n)


    for i in range(n):
        a = axes[i]
        a.clear()
        name, y = data[i]
        val = y[-1]
        name = ' '.join([w.title() for w in name.split('_')])
        a.set_title(f'{name}: {val:.6f}', fontdict=fontdict)
        if xticks[i] is not None:
            a.set_xticks(xticks[i])
        if yticks[i] is not None:
            a.set_yticks(yticks[i])
        if yscale[i] is not None:
            a.set_yscale(yscale[i])
        if colors[i] is not None:
            config = {'color': colors[i]}
        else:
            config = {}
        x = list(range(1, len(y) + 1))
        a.plot(x, y, **config)
