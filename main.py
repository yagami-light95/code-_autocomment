import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
from tensorflow import keras
import time
from nltk.translate.bleu_score import *
import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from code2tensor import creat_dataset,tokenize,seq2seq_getTestTensor,getTestCodeTensor
from seq2seq import Encoder,Decoder
from transformer import Transformer,CustomSchedule
from seq2seq_onlycode import sDecoder,sEncoder,seq2seq_onlycode_translate
from eval import seq2seq_result,sentenceBleu,transformer_result,seq2trans_result,seqonlycode2trans_result

# seq2seq模型恢复并测试
def seq2seq_infer(vocab_maxlen,sbt_lang,nl_lang,nl_maxlen,code_tensor, sbt_tensor):

    batch_size = 128
    units = 256
    embedding_dim = 256

    train_code_encoder = Encoder(vocab_maxlen + 1, embedding_dim, units, batch_size)
    train_sbt_encoder = Encoder(len(sbt_lang.word_index) + 1, embedding_dim, units, batch_size)
    decoder = Decoder(vocab_maxlen + 1, embedding_dim, units, batch_size)
    optimizer = keras.optimizers.Adam()

    # 从checkpoint文件中恢复网络
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, train_code_encoder=train_code_encoder,
                                     train_sbt_encoder=train_sbt_encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint('seq2seq/checkpoints'))

    seq2seq_path = 'seq2seq/result.txt'
    test_code_tensor, test_sbt_tensor= seq2seq_getTestTensor(code_tensor, sbt_tensor)
    seq2seq_result(test_code_tensor, test_sbt_tensor, seq2seq_path, units, nl_lang, nl_maxlen, train_code_encoder,
                   train_sbt_encoder, decoder)


def seq2seq_onlycode_infer(vocab_maxlen,nl_lang,nl_maxlen,code_tensor):
    batch_size = 128
    units = 256
    embedding_dim = 256

    train_code_encoder = sEncoder(vocab_maxlen + 1, embedding_dim, units, batch_size)
    decoder = sDecoder(vocab_maxlen + 1, embedding_dim, units, batch_size)
    # 从checkpoint文件中恢复网络
    checkpoint = tf.train.Checkpoint(train_code_encoder=train_code_encoder,decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint('seq2seq_onlycode/checkpoints'))

    seq2seq_path = 'seq2seq_onlycode/result.txt'
    test_code_tensor = getTestCodeTensor(code_tensor)
    # seq2seq_onlycode_translate(test_code_tensor, seq2seq_path, units, nl_lang, nl_maxlen, train_code_encoder, decoder, num=100)

    seq2seq_onlycode_translate(test_code_tensor,seq2seq_path,units,nl_lang,nl_maxlen,train_code_encoder,decoder,num=None)


# tranformer模型恢复并测试
def transformer_infer(vocab_maxlen,code_tensor,nl_lang,nl_maxlen):
    # 超参数
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    input_vocab_size = vocab_maxlen
    target_vocab_size = vocab_maxlen
    dropout_rate = 0.1

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98,epsilon=1e-9)
    # 从checkpoint文件中恢复模型
    checkpoint = tf.train.Checkpoint(transformer=transformer,optimizer=optimizer)
    checkpoint.restore(tf.train.latest_checkpoint('transformer/checkpoints/train'))

    test_code_tensor = getTestCodeTensor(code_tensor)
    transformer_result_path='transformer/result.txt'
    transformer_result(test_code_tensor,transformer_result_path,nl_lang,nl_maxlen,transformer)


# start seq2seq与transformer的集成
def seq2seq_layer(vocab_maxlen,sbt_lang):
    batch_size = 128
    units = 256
    embedding_dim = 256

    train_code_encoder = Encoder(vocab_maxlen + 1, embedding_dim, units, batch_size)
    train_sbt_encoder = Encoder(len(sbt_lang.word_index) + 1, embedding_dim, units, batch_size)
    decoder = Decoder(vocab_maxlen + 1, embedding_dim, units, batch_size)
    optimizer = keras.optimizers.Adam()

    # 从checkpoint文件中恢复网络
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, train_code_encoder=train_code_encoder,
                                     train_sbt_encoder=train_sbt_encoder, decoder=decoder)
    checkpoint.restore(tf.train.latest_checkpoint('seq2seq/checkpoints'))
    return train_code_encoder,train_sbt_encoder,decoder


def tranformer_layer(vocab_maxlen):
    # 超参数
    num_layers = 4
    d_model = 128
    dff = 512
    num_heads = 8

    input_vocab_size = vocab_maxlen
    target_vocab_size = vocab_maxlen
    dropout_rate = 0.1

    transformer = Transformer(num_layers, d_model, num_heads, dff,
                              input_vocab_size, target_vocab_size,
                              pe_input=input_vocab_size,
                              pe_target=target_vocab_size,
                              rate=dropout_rate)

    learning_rate = CustomSchedule(d_model)
    optimizer = tf.keras.optimizers.Adam(learning_rate, beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    # 从checkpoint文件中恢复模型
    checkpoint = tf.train.Checkpoint(transformer=transformer, optimizer=optimizer)
    checkpoint.restore(tf.train.latest_checkpoint('transformer/checkpoints/train'))
    return  transformer


def seq2trans(vocab_maxlen,sbt_lang,nl_lang,nl_maxlen,code_tensor, sbt_tensor):
    train_code_encoder, train_sbt_encoder, decoder=seq2seq_layer(vocab_maxlen,sbt_lang)
    transformer=tranformer_layer(vocab_maxlen)
    units=256
    path='ensemble/result.txt'
    test_code_tensor, test_sbt_tensor = seq2seq_getTestTensor(code_tensor, sbt_tensor)
    seq2trans_result(test_code_tensor,test_sbt_tensor,path,units,nl_lang,nl_maxlen,train_code_encoder,train_sbt_encoder,decoder,transformer,num=100)

# end seq2seq与transformer的集成


# start seq2seq_onlycode与transformer的集成
def seq2seq_onlycode_layer(vocab_maxlen):
    batch_size = 128
    units = 256
    embedding_dim = 256

    train_code_encoder = sEncoder(vocab_maxlen + 1, embedding_dim, units, batch_size)
    decoder = sDecoder(vocab_maxlen + 1, embedding_dim, units, batch_size)

    # 从checkpoint文件中恢复网络
    checkpoint = tf.train.Checkpoint(train_code_encoder=train_code_encoder, decoder=decoder)

    checkpoint.restore(tf.train.latest_checkpoint('seq2seq_onlycode/checkpoints'))
    return train_code_encoder, decoder

def seqonlycode2trans(vocab_maxlen,nl_lang,nl_maxlen,code_tensor):
    train_code_encoder, decoder = seq2seq_onlycode_layer(vocab_maxlen)
    transformer = tranformer_layer(vocab_maxlen)
    units = 256
    path = 'ensemble/sct_result.txt'
    test_code_tensor= getTestCodeTensor(code_tensor)
    seqonlycode2trans_result(test_code_tensor, path, units, nl_lang, nl_maxlen, train_code_encoder,
                      decoder, transformer, num=None)


# end seq2seq_onlycode与transformer的集成


#BLEU 评分 model=['seq2seq','transformer','seq2trans']
def result_bleu(model):
    seq2seq_ava_score = sentenceBleu(model)
    print("avg:",seq2seq_ava_score)


def main():

    # 定义文件路径
    code_path = 'dataset/camel_code.txt'
    sbt_path = 'dataset/sbt.txt'
    nl_path = 'dataset/comment.txt'

    # 读取文件
    code, sbt, nl = creat_dataset(code_path, sbt_path, nl_path)

    # 定义tensor长度及词典大小
    code_maxlen = 500
    sbt_maxlen = 500
    nl_maxlen = 30
    vocab_maxlen = 30000

    # code向量化
    code_tensor, code_lang = tokenize(vocab_maxlen,code, code_maxlen)
    sbt_tensor, sbt_lang = tokenize(vocab_maxlen,sbt, sbt_maxlen)
    nl_tensor, nl_lang = tokenize(vocab_maxlen,nl, nl_maxlen)

    print(code_tensor.shape)
    print(sbt_tensor.shape)
    print(nl_tensor.shape)
    print(len(sbt_lang.word_index))

    #seq2seq_infer(vocab_maxlen,  sbt_lang, nl_lang, nl_maxlen, code_tensor, sbt_tensor)
    result_bleu(model='seq2seq')


    #transformer_infer(vocab_maxlen, code_tensor, nl_lang, nl_maxlen)
    #result_bleu(model='transformer')

    #seq2seq_onlycode_infer(vocab_maxlen, nl_lang, nl_maxlen, code_tensor)
    #result_bleu(model='seq2seq_onlycode')

    # seq2trans(vocab_maxlen, sbt_lang, nl_lang, nl_maxlen, code_tensor, sbt_tensor)
    # result_bleu(model='seq2trans')

    # seqonlycode2trans(vocab_maxlen, nl_lang, nl_maxlen, code_tensor)
    # result_bleu(model='seqonlycode2trans')
    # 无权重：0.563853652334339
    # 带权重：100:0.5721481119418715  20000:0.6027096657625092


if __name__ == '__main__':
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    main()


   # 权重计算
    seq=result_bleu(model='seq2seq')
   #  seq=sentenceBleu(model='seq2seq_onlycode')
   #  print(seq)
    trans=sentenceBleu(model='transformer')
   #
   #  seq_=seq/(seq+trans)
   #  trans_=trans/(seq+trans)
    #print(seq)  #0.5920943071978862

    #print(trans) #0.40790569280211364


