import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import pickle, pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.data import DataLoader

def get_IEMOCAP_bert_loaders(path=None, batch_size=32, num_workers=0, pin_memory=False, valid_rate=0.1):
    trainset = IEMOCAPRobertaCometDataset(path=path, split='train')
    # train_sampler, valid_sampler = get_train_valid_sampler(trainset, valid_rate)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              # sampler=train_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)
    validset = IEMOCAPRobertaCometDataset(path=path, split='valid')
    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              # sampler=valid_sampler,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    testset = IEMOCAPRobertaCometDataset(path=path, split='test')
    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def get_EmoryNLP_bert_loaders(path, batch_size=8, classify='emotion', num_workers=0, pin_memory=False):
    trainset = EmoryNLPRobertaCometDataset(path, 'train', classify)
    validset = EmoryNLPRobertaCometDataset(path, 'valid', classify)
    testset = EmoryNLPRobertaCometDataset(path, 'test', classify)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True  # shuffle for training data
                              )

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def get_MELD_bert_loaders(path, batch_size=8, classify='emotion', num_workers=0, pin_memory=False):
    trainset = MELDRobertaCometDataset(path, 'train', classify)
    validset = MELDRobertaCometDataset(path, 'valid', classify)
    testset = MELDRobertaCometDataset(path, 'test', classify)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True  # shuffle for training data
                              )

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader

def get_DailyDialog_bert_loaders(path, batch_size=32, classify='emotion', num_workers=0, pin_memory=False):
    trainset = DailyDialogRobertaCometDataset(path, 'train', classify)
    validset = DailyDialogRobertaCometDataset(path, 'valid', classify)
    testset = DailyDialogRobertaCometDataset(path, 'test', classify)

    train_loader = DataLoader(trainset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory,
                              shuffle=True  # shuffle for training data
                              )

    valid_loader = DataLoader(validset,
                              batch_size=batch_size,
                              collate_fn=trainset.collate_fn,
                              num_workers=num_workers,
                              pin_memory=pin_memory)

    test_loader = DataLoader(testset,
                             batch_size=batch_size,
                             collate_fn=testset.collate_fn,
                             num_workers=num_workers,
                             pin_memory=pin_memory)

    return train_loader, valid_loader, test_loader


class IEMOCAPDataset(Dataset):

    def __init__(self, path=None, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, self.trainVid, \
        self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) if i < 6 else dat[i].tolist() for i in dat]


class IEMOCAPRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None):

        self.speakers, self.labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        '''
        label index mapping = {'hap':0, 'sad':1, 'neu':2, 'ang':3, 'exc':4, 'fru':5}
        '''

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'train-valid':
            # we randomly extract 10% or 20% of the training data as validation split
            # since no pre-defined train/val split is provided in original IEMOCAP dataset
            self.keys = [x for x in self.trainIds] + [x for x in self.validIds]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])), \
               torch.FloatTensor([[1, 0] if x == 'M' else [0, 1] for x in self.speakers[vid]]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in dat]


class AVECDataset(Dataset):

    def __init__(self, path, train=True):
        self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
        self.videoAudio, self.videoVisual, self.videoSentence, \
        self.trainVid, self.testVid = pickle.load(open(path, 'rb'), encoding='latin1')

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoVisual[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor([[1, 0] if x == 'user' else [0, 1] for x in \
                                  self.videoSpeakers[vid]]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.FloatTensor(self.videoLabels[vid])

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 4 else pad_sequence(dat[i], True) for i in dat]


class MELDDataset(Dataset):

    def __init__(self, path=None, n_classes=-1, train=True):
        if n_classes == 3:
            self.videoIDs, self.videoSpeakers, _, self.videoText, \
            self.videoAudio, self.videoSentence, self.trainVid, \
            self.testVid, self.videoLabels = pickle.load(open(path, 'rb'))
        elif n_classes == 7:
            self.videoIDs, self.videoSpeakers, self.videoLabels, self.videoText, \
            self.videoAudio, self.videoSentence, self.trainVid, \
            self.testVid, _ = pickle.load(open(path, 'rb'))

        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''

        self.keys = [x for x in (self.trainVid if train else self.testVid)]

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(self.videoText[vid]), \
               torch.FloatTensor(self.videoAudio[vid]), \
               torch.FloatTensor(self.videoSpeakers[vid]), \
               torch.FloatTensor([1] * len(self.videoLabels[vid])), \
               torch.LongTensor(self.videoLabels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 3 else pad_sequence(dat[i], True) if i < 5 else dat[i].tolist() for i in dat]


class MELDRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None, classify='emotion'):


        self.speakers, self.emotion_labels, self.sentiment_labels, \
        self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
        self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])), \
               torch.FloatTensor(self.speakers[vid]), \
               torch.FloatTensor([1] * len(self.labels[vid])), \
               torch.LongTensor(self.labels[vid]), \
               vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in dat]


class DailyDialogRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None, classify='emotion'):

        self.speakers, self.emotion_labels, \
            self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
            self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        elif split == 'train-valid-test':
            self.keys = [x for x in self.trainIds] + [x for x in self.validIds] + [x for x in self.testIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])), \
            torch.FloatTensor([[1, 0] if x == '0' else [0, 1] for x in self.speakers[vid]]), \
            torch.FloatTensor([1] * len(self.labels[vid])), \
            torch.LongTensor(self.labels[vid]), \
            vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]


class EmoryNLPRobertaCometDataset(Dataset):

    def __init__(self, path=None, split=None, classify='emotion'):

        self.speakers, self.emotion_labels, \
            self.roberta1, self.roberta2, self.roberta3, self.roberta4, \
            self.sentences, self.trainIds, self.testIds, self.validIds \
            = pickle.load(open(path, 'rb'), encoding='latin1')

        '''
        label index mapping = {'neutral': 0, 'surprise': 1, 'fear': 2, 'sadness': 3, 'joy': 4, 'disgust': 5, 'anger':6}
        '''

        if split == 'train':
            self.keys = [x for x in self.trainIds]
        elif split == 'test':
            self.keys = [x for x in self.testIds]
        elif split == 'valid':
            self.keys = [x for x in self.validIds]
        elif split == 'train-valid-test':
            self.keys = [x for x in self.trainIds] + [x for x in self.validIds] + [x for x in self.testIds]

        if classify == 'emotion':
            self.labels = self.emotion_labels
        else:
            self.labels = self.sentiment_labels

        self.len = len(self.keys)

    def __getitem__(self, index):
        vid = self.keys[index]
        return torch.FloatTensor(np.array(self.roberta1[vid])), \
            torch.FloatTensor(self.speakers[vid]), \
            torch.FloatTensor([1] * len(self.labels[vid])), \
            torch.LongTensor(self.labels[vid]), \
            vid

    def __len__(self):
        return self.len

    def collate_fn(self, data):
        dat = pd.DataFrame(data)
        return [pad_sequence(dat[i]) if i < 2 else pad_sequence(dat[i], True) if i < 4 else dat[i].tolist() for i in
                dat]


if __name__ == '__main__':
    # ../data/meld/meld_features_roberta.pkl
    # train_loader, valid_loader, test_loader = get_EmoryNLP_bert_loaders('../data/emorynlp/emorynlp_features_roberta.pkl', batch_size=8, classify='emotion', num_workers=0)
    # for data in train_loader:
    #
    #     # roberta_fea: CLS embedding of last hidden layer in RoBERTa
    #     roberta_fea, qmask, umask, label = data[:-1]
    #     print(label.size())
    # train_loader, valid_loader, test_loader = get_MELD_bert_loaders(
    #     '../data/meld/meld_features_roberta.pkl', batch_size=8, classify='emotion', num_workers=0)
    # for data in train_loader:
    #
    #     # roberta_fea: CLS embedding of last hidden layer in RoBERTa
    #     roberta_fea, qmask, umask, label = data[:-1]
    #     print(qmask.size())

    train_loader, valid_loader, test_loader = get_DailyDialog_bert_loaders(path='../data/dailydialog/dailydialog_features_roberta.pkl', batch_size=32, num_workers=0)
    a0,a1,a2,a3,a4,a5,a6 = 0, 0, 0, 0, 0, 0, 0
    for data in train_loader:

        # roberta_fea: CLS embedding of last hidden layer in RoBERTa
        roberta_fea, qmask, umask, label2 = data[:-1]
        seq_lengths = [(umask[j] == 1).nonzero().tolist()[-1][0] + 1 for j in range(len(umask))]
        label = torch.cat([label2[j][:seq_lengths[j]] for j in range(len(label2))])
        count0 = torch.eq(label, 0).sum().item()
        count1 = torch.eq(label, 1).sum().item()
        count2 = torch.eq(label, 2).sum().item()
        count3 = torch.eq(label, 3).sum().item()
        count4 = torch.eq(label, 4).sum().item()
        count5 = torch.eq(label, 5).sum().item()
        count6 = torch.eq(label, 6).sum().item()
        a0 = a0 + count0
        a1 = a1 + count1
        a2 = a2 + count2
        a3 = a3 + count3
        a4 = a4 + count4
        a5 = a5 + count5
        a6 = a6 + count6
        print(count0,count1,count2,count3,count4,count5,count6)
    print(a0,a1,a2,a3,a4,a5,a6)