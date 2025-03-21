# Fast Accurate Symbolic Empirical Representation Of Histograms Project - GSoC 2025 Evaluation Test

This repository contains the work for the **FASEROH Project** evaluation for prospective GSoC 2025 applicants. It includes two Jupyter notebooks demonstrating solutions for the following tasks:

- **Common Task 1: Dataset Preprocessing**  
  Use Sympy (or Mathematica) to generate datasets of functions with their Taylor expansions (up to the fourth order) and tokenize the dataset.

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
  - **Architecture:**
  ```bash
  Seq2Seq(
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
  
 ![LSTM Architecture](https://i.imgur.com/HnwDUkJ.png) 

- **transformer.ipynb**  
  Contains the implementation of the Transformer model for learning Taylor series expansions.
  
  **Approach:**  
  - **Dataset Preprocessing:**  
    - Same as for LSTM, with added special tokens `<SOS>`, `<EOS>`, and `<UNK>`.
  - **Transformer Model Architecture:**  
    - Uses source and target embeddings with positional encoding.
    - Transformer encoder and decoder layers are employed.
    - A linear output layer projects the hidden representations to the vocabulary space.
  - **Architecture Diagram:**  
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
