import argparse

def txt_to_fasta(input_file, output_file):
    with open(input_file, 'r') as infile, open(output_file, 'w') as outfile:
        for line in infile:
            # 去掉行末的换行符
            line = line.strip()
            if line.startswith("Generated sequence"):
                # 提取序列编号和序列
                parts = line.split(": ")
                sequence_id = parts[0].replace("Generated sequence ", "")
                sequence = parts[1]
                # 写入FASTA格式
                outfile.write(f">sequence_{sequence_id}\n")
                outfile.write(f"{sequence}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert text file to FASTA format")
    parser.add_argument('-i', '--input', type=str, required=True, help="Input text file")
    parser.add_argument('-o', '--output', type=str, required=True, help="Output FASTA file")

    args = parser.parse_args()
    
    txt_to_fasta(args.input, args.output)