# Fast Accurate Symbolic Empirical Representation Of Histograms Project - GSoC 2025 Evaluation Test

This repository contains the work for the **FASEROH Project** evaluation for GSoC 2025. It includes two Jupyter notebooks and one python file demonstrating solutions for the following tasks completed:

- **Common Task 1: Dataset Preprocessing** 
  Used Sympy to generate dataset of functions with their Taylor expansions (up to the fourth order) and tokenize the dataset.

- **Common Task 2: LSTM Model** 
  Trained an LSTM-based sequence-to-sequence model to learn the Taylor expansion of each function.

- **Specific Task 3: Transformer Model**  
  Trained a Transformer-based model to learn the Taylor expansion of each function.

---

## Repository Contents

- **[dataset.py](https://github.com/ShubhamChauhan22222/FASEROH_test/blob/main/dataset.py)**
  
  Functions are generated using Sympy.
  - **Functionality:**  
    - Randomly generates symbolic mathematical functions using a predefined vocabulary of operators and functions.
    - Supports unary (sin, exp, etc.) and binary (+, *, etc.) operations.
    - Ensures generated functions are valid SymPy expressions.
    - It takes the parameter to generate the no. of function. In LSTM notebook I only generated 100 functions but we can generate as many as we want.
  
  - **Key Methods:**  
    - generate_expression(): Recursively builds function strings.
    - sequence_to_sympy(): Converts token sequences into SymPy expressions.
    - generate_functions(): Produces a list of num_samples valid functions.
  
- **[LSTM.ipynb](https://github.com/ShubhamChauhan22222/FASEROH_test/blob/main/LSTM.ipynb)**
  
  Contains the implementation of the LSTM model for learning Taylor series expansions.
  
  **Approach:**  
  - **Dataset Preprocessing:**
    - Functions are generated using the class GenerateFunction from dataset.py. Only 100 functions were genertaed to keep it simple.
    - Taylor expansions are computed up to the fourth order.
    - The dataset is tokenized to build the vocabulary.
    - Added special tokens `<SOS>`, `<EOS>`, and `<UNK>`.
  - **LSTM Model Architecture:**  
    - An embedding layer is used to represent tokens.
    - LSTM encoder and decoder are used to form a sequence-to-sequence architecture.
    - Teacher forcing is applied during training.
  - **Model:**
    ```bash
      LSTMSeq2Seq(
          (encoder): Encoder(
            (embedding): Embedding(67, 32, padding_idx=0)
            (lstm): LSTM(32, 64, num_layers=2, batch_first=True)
          )
          (decoder): Decoder(
            (embedding): Embedding(67, 32, padding_idx=0)
            (lstm): LSTM(32, 64, num_layers=2, batch_first=True)
            (fc): Linear(in_features=64, out_features=67, bias=True)
            (log_softmax): LogSoftmax(dim=1)
          ))
  
  - **Architecture diagram:**
     ![LSTM Architecture](https://i.imgur.com/atonrRY.png) 

- **[transformer.ipynb](https://github.com/ShubhamChauhan22222/FASEROH_test/blob/main/transformer.ipynb)**
  
  Contains the implementation of the Transformer model for learning Taylor series expansions.
  
  **Approach:**  
  - **Dataset Preprocessing:**
    - Same as LSTM.
  - **Transformer Model Architecture:**  
    - Uses source and target embeddings with positional encoding.
    - Transformer encoder and decoder layers are employed.
    - A linear output layer projects the hidden representations to the vocabulary space.
  - **Model:**
    ```bash
      TransformerModel(
        (src_embedding): Embedding(37, 32, padding_idx=0)
        (tgt_embedding): Embedding(37, 32, padding_idx=0)
        (pos_encoder): PositionalEncoding(
          (dropout): Dropout(p=0.1, inplace=False)
        )---
        (transformer): Transformer(
          (encoder): TransformerEncoder(---)
          (decoder): TransformerDecoder(---)
            (norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
          )
        )
        (fc_out): Linear(in_features=32, out_features=37, bias=True)
      )
  
  - **Architecture diagram:**
    ![Transformer Architecture](https://i.imgur.com/j8GN4wK.png)


## Results
- The data generation pipeline was working good and we can generate as many functions and even add more types and variations.
- Tokenization of both the function and its expansion were done using word_tokenize from nltk library with added EOS and SOS tokens.
- Two dictionaries were created vocab_to_int and int_to_vocab for using further during testing.
- Both the models were trained on GPU P100 on kaggle and the loss was decreasing with epochs indicating smooth training.
- Some samples were tested and outputs were correct as expected.
- Finally, if we want we can generate thousands of functions and their expansions, train them to get better results.
- For future, we can further implement reinforcement learning to ensure correct and valid function representations.
