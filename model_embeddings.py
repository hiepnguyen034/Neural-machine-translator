import torch.nn as nn
import codecs
import json
import numpy as np

class ModelEmbeddings(nn.Module): 
    """
    Class that converts input words to their embeddings.
 
    """
    def __init__(self, embed_size, vocab):
        """
        Init the Embedding layers.

        @param embed_size (int): Embedding size (dimensionality)
        @param vocab (Vocab): Vocabulary object containing src and tgt languages
                              See vocab.py for documentation.
        """
        super(ModelEmbeddings, self).__init__()
        self.embed_size = embed_size
        src_pad_token_idx = vocab.src['<pad>']
        tgt_pad_token_idx = vocab.tgt['<pad>']
        
        self.source = nn.Embedding(len(vocab.src),self.embed_size, padding_idx = src_pad_token_idx)
        self.target = nn.Embedding(len(vocab.tgt) ,self.embed_size, padding_idx = tgt_pad_token_idx)
        emb_matrix = self.get_embedding_matrix(vocab=vocab, embed_size=self.embed_size)
        self.target.weight = nn.Parameter(torch.tensor(emb_matrix,dtype=torch.float32))

        self.target.weight.requires_grad= False #freeze word2vec embedding

    @staticmethod
    def open_vocab(file = 'eng_vn_vocab.json', src='source'):
        entry = json.load(codecs.open(file_path, 'r',encoding='cp720'))

        if src == 'source':
            word2id = entry['src_word2id']
        elif src == 'target':
            word2id = entry['tgt_word2id']

        return word2id

    @staticmethod
    def get_embedding_matrix(vocab, embed_size, w2v_file = 'eng_vie_data/glove.6B.50d.txt'):
        #open gloves
        with codecs.open(w2v_file,'r', encoding="cp720") as f:
            words = set()
            word_to_vec_map = {}
            for line in f:
                line = line.strip().split()
                curr_word = line[0]
                words.add(curr_word)
                word_to_vec_map[curr_word] = np.array(line[1:], dtype=np.float64)

        emb_matrix = np.zeros((len(vocab.src), embed_size))

        for word, index in self.open_vocab(src='target'):
            emb_matrix[index, :] =word_to_vec_map[word]

        return emb_matrix
