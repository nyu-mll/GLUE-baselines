'''Preprocessing functions and pipeline'''
import os
import logging as log
from collections import defaultdict
import _pickle as pkl
import numpy as np
import torch

from allennlp.data import Instance, Vocabulary, Token
from allennlp.data.fields import TextField, LabelField
from allennlp.data.token_indexers import SingleIdTokenIndexer, ELMoTokenCharactersIndexer
from allennlp_mods.numeric_field import NumericField

from tasks import CoLATask, MRPCTask, MultiNLITask, QQPTask, RTETask, \
                  QNLITask, QNLIv2Task, SNLITask, SSTTask, STSBTask, WNLITask
import serialize
import utils

if "cs.nyu.edu" in os.uname()[1] or "dgx" in os.uname()[1]:
    PATH_PREFIX = '/misc/vlgscratch4/BowmanGroup/awang/'
else:
    PATH_PREFIX = '/beegfs/aw3272/'
PATH_PREFIX = PATH_PREFIX + 'processed_data/mtl-sentence-representations/'

#ALL_TASKS = ['mnli', 'mrpc', 'qqp', 'rte', 'qnli', 'snli', 'sst', 'sts-b', 'wnli', 'cola']
ALL_TASKS = ['mnli', 'mrpc', 'qqp', 'rte', 'qnliv2', 'snli', 'sst', 'sts-b', 'wnli', 'cola']
NAME2INFO = {'sst': (SSTTask, 'SST-2/'),
             'cola': (CoLATask, 'CoLA/'),
             'mrpc': (MRPCTask, 'MRPC/'),
             'qqp': (QQPTask, 'QQP/'),
             'sts-b': (STSBTask, 'STS-B/'),
             'mnli': (MultiNLITask, 'MNLI/'),
             'qnli': (QNLITask, 'QNLI/'),
             'qnliv2': (QNLIv2Task, 'QNLIv2/'),
             'rte': (RTETask, 'RTE/'),
             'snli': (SNLITask, 'SNLI/'),
             'wnli': (WNLITask, 'WNLI/')
            }
for k, v in NAME2INFO.items():
    NAME2INFO[k] = (v[0], PATH_PREFIX + v[1])

ALL_SPLITS = ["train", "dev", "test"]

def _get_serialized_record_path(task_name, split, preproc_dir):
    """Get the canonical path for a serialized task split."""
    serialized_record_path = os.path.join(preproc_dir,
                                          "{:s}__{:s}_data".format(task_name, split))
    return serialized_record_path


def _get_instance_generator(task_name, split, preproc_dir, fraction=None):
    """Get a lazy generator for the given task and split.

    Args:
        task_name: (string), task name
        split: (string), split name ('train', 'val', or 'test')
        preproc_dir: (string) path to preprocessing dir
        fraction: if set to a float between 0 and 1, load only the specified percentage
          of examples. Hashing is used to ensure that the same examples are loaded each
          epoch.

    Returns:
        serialize.RepeatableIterator yielding Instance objects
    """
    filename = _get_serialized_record_path(task_name, split, preproc_dir)
    assert os.path.isfile(filename), ("Record file '%s' not found!" % filename)
    return serialize.read_records(filename, repeatable=True, fraction=fraction)


def _indexed_instance_generator(instance_iter, vocab):
    """Yield indexed instances. Instances are modified in-place.

    Args:
        instance_iter: iterable(Instance) of examples
        vocab: Vocabulary for use in indexing

    Yields:
        Instance with indexed fields.
    """
    for instance in instance_iter:
        instance.index_fields(vocab)
        # Strip token fields to save memory and disk.
        del_field_tokens(instance)
        yield instance

def del_field_tokens(instance):
    ''' Save memory by deleting the tokens that will no longer be used '''
    #all_instances = task.train_data.instances + task.val_data.instances + task.test_data.instances
    if 'input1' in instance.fields:
        field = instance.fields['input1']
        del field.tokens
    if 'input2' in instance.fields:
        field = instance.fields['input2']
        del field.tokens

def build_tasks(args):
    '''Prepare tasks'''

    def parse_tasks(task_list):
        '''parse string of tasks'''
        if task_list == 'all':
            tasks = ALL_TASKS
        elif task_list == 'none':
            tasks = []
        else:
            tasks = task_list.split(',')
        return tasks

    train_task_names = parse_tasks(args.train_tasks)
    eval_task_names = parse_tasks(args.eval_tasks)
    all_task_names = list(set(train_task_names + eval_task_names))
    tasks = get_tasks(all_task_names, args.max_seq_len, args.reload_tasks)

    # 2) Vocab and indexers
    token_indexer = {}
    if args.elmo:
        token_indexer["elmo"] = ELMoTokenCharactersIndexer("elmo")
        if not args.elmo_no_glove:
            token_indexer["words"] = SingleIdTokenIndexer()
    else:
        token_indexer["words"] = SingleIdTokenIndexer()

    vocab_path = os.path.join(args.exp_dir, 'vocab')
    if args.reload_vocab or not os.path.exists(vocab_path):
        _build_vocab(args, tasks, vocab_path)
    vocab = Vocabulary.from_files(vocab_path)
    args.max_word_v_size = vocab.get_vocab_size('tokens')
    args.max_char_v_size = vocab.get_vocab_size('chars')
    log.info("\tLoaded vocab from %s" % vocab_path)

    # 3) Word embeddings
    word_embs = None
    if args.word_embs != 'none':
        emb_file = os.path.join(args.exp_dir, 'embs.pkl')
        if args.reload_vocab or not os.path.exists(emb_file):
            word_embs = get_embeddings(vocab, args.word_embs_file, args.d_word)
        else:
            word_embs = pkl.load(open(emb_file, 'rb'))
        log.info("\tLoaded %d word embeddings from %s" % (word_embs.size(0), emb_file))

    # 4) Index tasks using vocab
    preproc_dir = os.path.join(args.exp_dir, "preproc")
    utils.maybe_make_dir(preproc_dir)
    force_reindex = args.reload_indexing or args.reload_vocab
    for task in tasks:
        for split in ALL_SPLITS:
            split_file = _get_serialized_record_path(task.name, split, preproc_dir)
            if not os.path.exists(split_file) or force_reindex:
                _index_split(task, split, token_indexer, vocab, split_file)
        task.train_data = None
        task.val_data = None
        task.test_data = None
        log.info("\tFinished indexing tasks")

    # 5) Initialize tasks with data iterators
    for task in tasks:
        task.train_data = _get_instance_generator(task.name, "train", preproc_dir)
        task.val_data = _get_instance_generator(task.name, "dev", preproc_dir)
        task.test_data = _get_instance_generator(task.name, "test", preproc_dir)

    train_tasks = [task for task in tasks if task.name in train_task_names]
    eval_tasks = [task for task in tasks if task.name in eval_task_names]
    log.info('\t  Training on %s', ', '.join([task.name for task in train_tasks]))
    log.info('\t  Evaluating on %s', ', '.join([task.name for task in eval_tasks]))
    return train_tasks, eval_tasks, vocab, word_embs

def get_tasks(task_names, max_seq_len, reload):
    '''
    Load tasks
    '''
    tasks = []
    for name in task_names:
        assert name in NAME2INFO, 'Task not found!'
        pkl_path = NAME2INFO[name][1] + "%s_task.pkl" % name
        if os.path.isfile(pkl_path) and not reload:
            task = pkl.load(open(pkl_path, 'rb'))
            log.info('\tLoaded existing task %s', name)
        else:
            task = NAME2INFO[name][0](NAME2INFO[name][1], max_seq_len, name)
            pkl.dump(task, open(pkl_path, 'wb'))
        if not hasattr(task, 'example_counts'):
            task.count_examples()
        log.info("\tTask '%s': %s", task.name,
                 " ".join(("%s=%d" % kv for kv in task.example_counts.items())))
        tasks.append(task)
    log.info("\tFinished loading tasks: %s.", ' '.join([task.name for task in tasks]))
    return tasks

def _build_vocab(args, tasks, vocab_path):
    def get_words(tasks):
        '''
        Get all words for all tasks for all splits for all sentences
        Return dictionary mapping words to frequencies.
        '''
        word2freq = defaultdict(int)

        def count_sentence(sentence):
            '''Update counts for words in the sentence'''
            for word in sentence:
                word2freq[word] += 1
            return

        for task in tasks:
            for sentence in task.get_sentences():
                count_sentence(sentence)
        log.info("\tFinished counting words")
        return word2freq

    def get_vocab(word2freq, max_v_sizes):
        '''Build vocabulary'''
        vocab = Vocabulary(counter=None, max_vocab_size=max_v_sizes['word'])
        words_by_freq = [(word, freq) for word, freq in word2freq.items()]
        words_by_freq.sort(key=lambda x: x[1], reverse=True)
        for word, _ in words_by_freq[:max_v_sizes['word']]:
            vocab.add_token_to_namespace(word, 'tokens')
        return vocab

    max_v_sizes = {"word": args.max_word_v_size, "char": args.max_char_v_size}
    word2freq = get_words(tasks)
    vocab = get_vocab(word2freq, max_v_sizes)
    vocab.save_to_files(vocab_path)
    log.info("\tFinished building vocab. Using %d words", vocab.get_vocab_size('tokens'))

def get_embeddings(vocab, vec_file, d_word):
    '''Get embeddings for the words in vocab'''
    word_v_size, unk_idx = vocab.get_vocab_size('tokens'), vocab.get_token_index(vocab._oov_token)
    embeddings = np.random.randn(word_v_size, d_word) #np.zeros((word_v_size, d_word))
    with open(vec_file) as vec_fh:
        for line in vec_fh:
            word, vec = line.split(' ', 1)
            idx = vocab.get_token_index(word)
            if idx != unk_idx:
                idx = vocab.get_token_index(word)
                embeddings[idx] = np.array(list(map(float, vec.split())))
    embeddings[vocab.get_token_index('@@PADDING@@')] = 0.
    embeddings = torch.FloatTensor(embeddings)
    log.info("\tFinished loading embeddings")
    return embeddings

def _index_split(task, split, token_indexer, vocab, split_file):
    '''
    Convert a task's splits into AllenNLP fields then
    Index the splits using the given vocab (experiment dependent)
    '''
    split = task.get_split_text(split)
    proc_split = process_split(split, token_indexer, task.pair_input, task.categorical)
    serialize.write_records(
        _indexed_instance_generator(proc_split, vocab), split_file)

def process_split(split, indexers, pair_input, categorical):
    '''
    Convert a dataset of sentences into padded sequences of indices.

    Args:
        - split (list[list[str]]): list of inputs (possibly pair) and outputs
        - pair_input (int)
        - tok2idx (dict)

    Returns:
    '''
    def _make_instance(instance, pair_input, categorical):
        sent1, sent2, trg, idx = instance
        input1 = TextField(list(map(Token, sent1)), token_indexers=indexers)
        if pair_input:
            input2 = TextField(list(map(Token, sent2)), token_indexers=indexers)
        if categorical:
            label = LabelField(trg, label_namespace="labels", skip_indexing=True)
        else:
            label = NumericField(trg)

        if idx is not None:
            idx = LabelField(idx, label_namespace="idxs", skip_indexing=True)
            if pair_input:
                instance = Instance({"input1": input1, "input2": input2, "label": label, "idx": idx})
            else:
                instance = Instance({"input1": input1, "label": label, "idx": idx})
        else:
            if pair_input:
                instance = Instance({"input1": input1, "input2": input2, "label": label})
            else:
                instance = Instance({"input1": input1, "label": label})
        return instance

    #for sent1, sent2, trg, idx in split:
    for instance in split:
        yield _make_instance(instance, pair_input, categorical)
