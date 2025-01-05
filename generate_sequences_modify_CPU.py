import torch
import pickle
from tqdm import tqdm
import logging
from tokenizer.simple_tokenizer import SimpleTokenizer
from model import SimpleModel

# 配置日志
logging.basicConfig(filename='generation_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')

def load_tokenizer(tokenizer_path):
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return tokenizer

def calculate_net_charge(sequence, tokenizer):
    charge = 0
    for token in sequence:
        amino_acid = tokenizer.decode([token])
        if amino_acid in ['H', 'K', 'R']:
            charge += 1
        elif amino_acid in ['D', 'E']:
            charge -= 1
    return charge

def calculate_hydrophobic_ratio(sequence, tokenizer):
    hydrophobic_count = 0
    hydrophobic_amino_acids = set(['A', 'I', 'L', 'M', 'F', 'W', 'Y', 'V', 'G', 'P'])
    for token in sequence:
        amino_acid = tokenizer.decode([token])
        if amino_acid in hydrophobic_amino_acids:
            hydrophobic_count += 1
    return hydrophobic_count / len(sequence)

def contains_aromatic_amino_acid(sequence, tokenizer):
    aromatic_amino_acids = set(['W', 'Y', 'F'])
    for token in sequence:
        amino_acid = tokenizer.decode([token])
        if amino_acid in aromatic_amino_acids:
            return True
    return False

def count_cysteines(sequence, tokenizer):
    return sum(1 for token in sequence if tokenizer.decode([token]) == 'C')

def is_valid_sequence(sequence, tokenizer):
    return (
        calculate_net_charge(sequence, tokenizer) >= 3 and
        0.4 <= calculate_hydrophobic_ratio(sequence, tokenizer) <= 0.6 and
        contains_aromatic_amino_acid(sequence, tokenizer) and
        count_cysteines(sequence, tokenizer) in [0, 2]
    )

def generate_one_sequence(tokenizer, model_state_dict, device, min_length, max_length, temperature):
    model = SimpleModel(tokenizer.vocab_size, 128, 256).to(device)
    model.load_state_dict(model_state_dict)
    model.eval()

    input_seq = torch.tensor([[tokenizer.vocab['[BOS]']]], dtype=torch.long).to(device)
    output_seq = []

    with torch.no_grad():
        while len(output_seq) < min_length:
            for _ in range(max_length):
                output = model(input_seq)
                output = output[:, -1, :] / temperature
                probabilities = torch.softmax(output, dim=-1)
                predicted_token = torch.multinomial(probabilities, 1).item()

                if predicted_token == tokenizer.vocab['[EOS]'] and len(output_seq) >= min_length:
                    break
                
                if predicted_token != tokenizer.vocab['[EOS]']:
                    output_seq.append(predicted_token)
                    input_seq = torch.cat((input_seq, torch.tensor([[predicted_token]], dtype=torch.long).to(device)), dim=1)

                if len(output_seq) >= max_length:
                    break

            if len(output_seq) < min_length:
                output_seq.clear()
                input_seq = torch.tensor([[tokenizer.vocab['[BOS]']]], dtype=torch.long).to(device)

    return output_seq

def generate_sequences(tokenizer, model_state_dict, device, num_sequences=20, min_length=5, max_length=10, temperature=1.0):
    generated_sequences = []

    with tqdm(total=num_sequences, desc="Generating sequences") as progress_bar:
        while len(generated_sequences) < num_sequences:
            sequence = generate_one_sequence(tokenizer, model_state_dict, device, min_length, max_length, temperature)
            if is_valid_sequence(sequence, tokenizer):
                generated_sequences.append(sequence)
                progress_bar.update(1)

    return generated_sequences

def decode_sequences(generated_sequences, tokenizer):
    decoded_sequences = []
    for encoded_sequence in generated_sequences:
        decoded_sequence = tokenizer.decode(encoded_sequence)
        decoded_sequence = decoded_sequence.replace('[BOS]', '').replace('[EOS]', '')
        decoded_sequences.append(decoded_sequence)
    return decoded_sequences

def write_sequences_to_file(decoded_sequences, filename="generated_sequences.txt"):
    with open(filename, 'w') as f:
        for i, seq in enumerate(decoded_sequences):
            f.write(f"Generated sequence {i+1}: {seq}\n")

if __name__ == "__main__":
    try:
        device = torch.device('cpu')  # 强制使用CPU
        tokenizer_path = "tokenizer.pkl"
        model_path = "trained_model.pth"
        num_sequences = 200000
        min_length = 5
        max_length = 10
        temperature = 1.0  # 调整此值以改变生成的随机性

        # 加载 tokenizer
        tokenizer = load_tokenizer(tokenizer_path)

        # 加载模型
        vocab_size = tokenizer.vocab_size
        embedding_dim = 128
        hidden_dim = 256
        model = SimpleModel(vocab_size, embedding_dim, hidden_dim).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))

        # 生成并解码序列
        generated_sequences = generate_sequences(tokenizer, model.state_dict(), device, num_sequences, min_length, max_length, temperature)
        decoded_sequences = decode_sequences(generated_sequences, tokenizer)

        # 写入文件
        write_sequences_to_file(decoded_sequences)
        
        print("finish")
    except Exception as e:
        logging.error(f"Error occurred: {str(e)}")
        print("error")