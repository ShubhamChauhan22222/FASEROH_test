# Fast Accurate Symbolic Empirical Representation Of Histograms Project - GSoC 2025 Evaluation Test

This repository contains the work for the **FASEROH Project** evaluation for prospective GSoC 2025 applicants. It includes two Jupyter notebooks demonstrating solutions for the following tasks:

- **Common Task 1: Dataset Preprocessing**  
  Use Sympy (or Mathematica) to generate datasets of functions with their Taylor expansions (up to the fourth order) and tokenize the dataset, with added special tokens `<SOS>`, `<EOS>`, and `<UNK>`.

- **Common Task 2: LSTM Model**  
  Train an LSTM-based sequence-to-sequence model to learn the Taylor expansion of each function.

- **Specific Task 3: Transformer Model**  
  Train a Transformer-based model to learn the Taylor expansion of each function.

---

## Repository Contents

- **LSTM.ipynb**  
  Contains the implementation of the LSTM model for learning Taylor series expansions.
  
  **Approach:**  
  - **Dataset Preprocessing:**  
    - Functions are generated using Sympy.
    - Taylor expansions are computed up to the fourth order.
    - The dataset is tokenized to build the vocabulary.
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
     ![LSTM Architecture](https://i.imgur.com/HnwDUkJ.png) 

- **transformer.ipynb**  
  Contains the implementation of the Transformer model for learning Taylor series expansions.
  
  **Approach:**  
  - **Dataset Preprocessing:**  
    - Same as for LSTM.
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
        )
        (pos_decoder): PositionalEncoding(
          (dropout): Dropout(p=0.1, inplace=False)
        )
        (transformer): Transformer(
          (encoder): TransformerEncoder(
            (layers): ModuleList(
              (0-1): 2 x TransformerEncoderLayer(
                (self_attn): MultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
                )
                (linear1): Linear(in_features=32, out_features=64, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (linear2): Linear(in_features=64, out_features=32, bias=True)
                (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                (dropout1): Dropout(p=0.1, inplace=False)
                (dropout2): Dropout(p=0.1, inplace=False)
              )
            )
            (norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
          )
          (decoder): TransformerDecoder(
            (layers): ModuleList(
              (0-1): 2 x TransformerDecoderLayer(
                (self_attn): MultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
                )
                (multihead_attn): MultiheadAttention(
                  (out_proj): NonDynamicallyQuantizableLinear(in_features=32, out_features=32, bias=True)
                )
                (linear1): Linear(in_features=32, out_features=64, bias=True)
                (dropout): Dropout(p=0.1, inplace=False)
                (linear2): Linear(in_features=64, out_features=32, bias=True)
                (norm1): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                (norm2): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                (norm3): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
                (dropout1): Dropout(p=0.1, inplace=False)
                (dropout2): Dropout(p=0.1, inplace=False)
                (dropout3): Dropout(p=0.1, inplace=False)
              )
            )
            (norm): LayerNorm((32,), eps=1e-05, elementwise_affine=True)
          )
        )
        (fc_out): Linear(in_features=32, out_features=37, bias=True)
      )
  
  - **Architecture diagram:**
    ![Transformer Architecture](https://i.imgur.com/DkbNnIa.png)  


---

## How to Run

1. **Clone the Repository:**
   ```bash
   git clone <repository-url>
   cd <repository-directory>
2. **Environment Setup:**

3. 
   Create a virtual environment and install the necessary packages:
   ```bash
   pip install -r requirements.txt
4. **Run the Notebooks: Launch Jupyter Notebook:**
   ```bash
   jupyter notebook
   
Open and run the following notebooks:

`LSTM.ipynb`

`transformer.ipynb`
