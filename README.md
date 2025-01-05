# AMP-generator

1.Introduction
The software is a GRU-based AMP generation software, as shown in Figure 1 is the structure of the software schematic, the software is mainly composed of the following four modules, respectively, the data processing module (data_processing), sequence encoding and decoding module (simple_tokenizer), the generator module (generate_sequences) and the model training module (train).
1.1 Operation and Design
Device Selection: Supports CPU and GPU (CUDA) acceleration.
Batch Processing: Batch loading of data using PyTorch DataLoader.
Cyclic Training: Iterate training over multiple rounds and save the model after each round.
Quality control: Ensure that the generated antimicrobial peptides meet specific biochemical properties.
![image](https://github.com/user-attachments/assets/9070a700-0135-49ec-b2c2-430ddcec256c)


2. Technical characteristics

2.1 Deep learning model
Model architecture: GRU (Gated Recurrent Unit) is used as the core model, and the features of amino acid sequences are automatically extracted through multi-layer neural networks.
Training data: a large-scale dataset of known amino acid sequences is used for training to ensure that the model has good generalisation ability.

2.2 Biological constraints
Net charge: the net charge of the generated sequence needs to be greater than or equal to 3.
Hydrophobicity ratio: the hydrophobic amino acid ratio of the generated sequence should be between 0.4 and 0.6.
Aromatic amino acids: the resulting sequence must contain at least one aromatic amino acid.
Cysteine residues: the number of cysteine residues in the generated sequence must be 0 or 2.

2.3 Logging and Debugging
Logging: Key information during the generation process will be recorded in the log file, including generated sequences, validation results, etc.
Debugging support: Detailed error messages and exception handling mechanisms are provided to facilitate debugging and optimisation by developers.

4. Development Environment
   
3.1 Hardware Requirements
CPU: It is recommended to use a multi-core processor, this software development uses NVIDIA GeForce RTX 3090, driver version 550.90.07, CUDA version 12.4 as the environment for training models.
Memory: At least 4GB
Storage: at least 10GB of available space

3.2 Software Requirements
Operating system: Linux
Python: version 3.7 and above
PyTorch: 1.7 and above
Other dependent libraries: pickle, tqdm, logging

4. Installation and use

4.1 Installation steps
Install Python: Make sure Python 3.7 or above is installed on your system.
Install dependencies: Use pip to install the required dependencies.
pip install -r requirements.txt
And make sure the following files are in the current folder
data_processing.py
generate_sequences_CPU.py
generate_sequences_GPU.py
model.py
processed_data.pkl
tokeniser.pkl
train.py
trained_model.pth
and the folder: tokenizer
The folder tokenizer contains the file simple_tokenizer.py
![image](https://github.com/user-attachments/assets/b6426e9b-2508-42de-b032-e79822d8ebcd)

4.2 Usage
Load the Tokenizer:
tokenizer_path = ‘tokenizer.pkl’
tokeniser = load_tokenizer(tokenizer_path)
Load the model:
model_path = ‘trained_model.pth’
vocab_size = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 256
model = SimpleModel(vocab_size, embedding_dim, hidden_dim).to(device)
model.load_state_dict(torch.load(model_path, map_location=device))
Generate sequences:
num_sequences = 20
min_length = 5
max_length = 10
temperature = 1.0
generated_sequences = generate_sequences(tokenizer, model.state_dict(), device, num_sequences, min_length, max_length, temperature)
Decode the sequences:
decoded_sequences = decode_sequences(generated_sequences, tokenizer)
write_sequences_to_file(decoded_sequences, filename=‘generated_sequences.txt’)

5. Run the example
You can directly use our trained model to run, here we provide both AMP generator using CPU and AMP generator using GPU accelerated computation:
# Run the generator using CPU
#This parameter determines the number of generated sequences num_sequences = 20
#This parameter determines the minimum length of the generated sequences min_length = 5
#This parameter determines the maximum length of the generated sequences max_length = 10
#This parameter determines the randomness of the generated sequences temperature = 1.0
Python generate_sequences_CPU.py
After running you will see the following progress bar showing the current progress and estimated time remaining until 100%.
![image](https://github.com/user-attachments/assets/efe9fa49-3cdb-41c7-b566-3c541bee9382)
![image](https://github.com/user-attachments/assets/0581ff2a-1ac1-4a43-8e1d-16dd957fd26a)
![image](https://github.com/user-attachments/assets/7a59514a-47dc-4154-9f53-1e098f9db321)

#Running the generator on the GPU is much faster, so we recommend that you run it this way!
Python generate_sequences_GPU.py
![image](https://github.com/user-attachments/assets/0f4d11cd-b857-4c9b-8bec-dec62e1581bb)
![image](https://github.com/user-attachments/assets/9f5413b5-98af-4e47-a93e-fb6ab3c81b5d)

6. Sample output
At the end of the process you will see two files: generation_log.txt and generated_sequences.txt.
The log file generation_log.txt will show the detection and generation during the run. If you set the number of generation to be too high, the file may not contain any content to avoid memory overflow and other problems, the content of the file is as follows:
![image](https://github.com/user-attachments/assets/3ed3d216-cfdd-4ff4-ab51-e2e7e3bb495d)

The generated_sequences.txt file has the following contents:
![image](https://github.com/user-attachments/assets/7f9fbf55-aa9e-4863-a945-f078e6834a36)

7. Training the model
If you need to train your own model to use the algorithm we have provided an example of how to do this, first make sure you have installed the environment required in requirements.txt and then prepare the fasta file which contains the AMP sequences in fasta format.
First run data_processing.py to regularise the sequence file and tokenise the sequences (tokenizer)
python data_processing.py

You will get processed_data.pkl and tokenizer.pkl for subsequent processing
![image](https://github.com/user-attachments/assets/67bf97aa-fb59-4c01-b82e-e51d74b96fa3)
![image](https://github.com/user-attachments/assets/53fa1479-cb91-4063-912d-bd77971f2bd3)

Then run train.py to train the model, after that you will see the current training progress and the number of rounds and the change in the Loss value for each round
#This function determines the number of training rounds num_epochs = 15
Python train.py
![image](https://github.com/user-attachments/assets/2db3c62c-3059-4ebd-90bf-a29130f744fc)

After that you will get the trained_model.pth model file for generation.
