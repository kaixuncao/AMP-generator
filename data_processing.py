import pickle
from Bio import SeqIO
import pandas as pd
import numpy as np
import re
from tokenizer.simple_tokenizer import SimpleTokenizer  # 假设 simple_tokenizer.py 在 tokenizer 子目录中

class SimpleTokenizer:
    def __init__(self, amino_acids="ACDEFGHIKLMNPQRSTVWY", special_tokens=None):
        self.amino_acids = amino_acids
        self.special_tokens = special_tokens or {'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'}
        
        # 构建词汇表
        self.vocab = {token: i for i, token in enumerate(self.special_tokens.values())}
        for i, amino in enumerate(self.amino_acids):
            self.vocab[amino] = i + len(self.special_tokens)
        
        # 构建逆向词汇表
        self.inv_vocab = {v: k for k, v in self.vocab.items()}
    
    def encode(self, sequence, add_special_tokens=True):
        encoded = [self.vocab.get(aa.upper(), self.vocab['[PAD]']) for aa in sequence]  # 使用 [PAD] 替代未知字符
        if add_special_tokens:
            encoded = [self.vocab['[BOS]']] + encoded + [self.vocab['[EOS]']]
        return encoded
    
    def decode(self, ids):
        tokens = [self.inv_vocab.get(i, '[PAD]') for i in ids]
        return ''.join(tokens)
    
    @property
    def vocab_size(self):
        return len(self.vocab)

def is_valid_sequence(sequence):
    """检查序列是否只包含合法的氨基酸字符"""
    valid_amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    return bool(re.fullmatch(f"[{valid_amino_acids}]*", sequence.upper()))

def read_fasta(file_path):
    """读取FASTA格式的文件并返回序列列表"""
    sequences = []
    illegal_sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        sequence = str(record.seq).upper()  # 将序列转换为大写
        if is_valid_sequence(sequence):
            sequences.append(sequence)
        else:
            illegal_sequences.append(sequence)
    
    if illegal_sequences:
        print(f"Illegal characters found in the following sequences:")
        for seq in illegal_sequences:
            print(seq)
    
    return sequences

def encode_sequence(sequence, tokenizer, max_length):
    encoded = tokenizer.encode(sequence, add_special_tokens=True)
    
    # 确保所有序列长度相同
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    elif len(encoded) < max_length:
        encoded.extend([tokenizer.vocab['[PAD]']] * (max_length - len(encoded)))
    return encoded

def preprocess_data(file_path, max_length=129):
    amino_acids = "ACDEFGHIKLMNPQRSTVWY"
    special_tokens = {'pad_token': '[PAD]', 'bos_token': '[BOS]', 'eos_token': '[EOS]'}
    
    # 创建 SimpleTokenizer
    tokenizer = SimpleTokenizer(amino_acids, special_tokens)
    
    # 读取FASTA文件
    sequences = read_fasta(file_path)

    # 将序列转换为DataFrame
    data = pd.DataFrame({"sequence": sequences})

    # 将序列编码为整数，并确保长度一致
    data["encoded"] = data["sequence"].apply(lambda seq: encode_sequence(seq, tokenizer, max_length))

    return data, tokenizer

if __name__ == "__main__":
    file_path = "AMPuniqmoree5_clean_rmdup.fasta"
    max_length = 129
    data, tokenizer = preprocess_data(file_path, max_length=max_length)
    
    # 保存 DataFrame
    data.to_pickle("processed_data.pkl")
    
    # 保存 tokenizer
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tokenizer, f)
