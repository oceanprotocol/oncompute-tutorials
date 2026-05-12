# Theory
## Tokenization
### What is Tokenization?

**Tokenization** is the process of breaking down a stream of text into smaller units called **tokens**. These tokens can be words, subwords, characters, or even symbols. Tokenization is one of the foundational steps in Natural Language Processing (NLP) because it transforms raw text into a structured format that models can understand and process.
### Why is Tokenization Necessary?

Human language is complex, ambiguous, and often unstructured. For machines to work with text, it must be converted into a numeric form. Tokenization is essential because machine learning models cannot directly process text. Tokenization provides a way to represent text in numerical form. These numerical representations are what the models can process and learn from.
### How Tokenization Works

The method of tokenization depends on the type of model and the task at hand. Here's an overview of common approaches:

1. Word-Level Tokenization

* Each word becomes a token.
* Large vocabulary, sensitive to spelling variations and OOV (out-of-vocabulary) words.

2. Subword Tokenization

* Splits words into smaller, more frequent units (subwords).
* Balances between vocabulary size and coverage.
* Common subword algorithms:
    * Byte-Pair Encoding (BPE)
    * WordPiece
    * Unigram Language Model
* Generally these tokenizers also have special tokens that have distinct use-cases (e.g., [CLS], [PAD])

3. Character-Level Tokenization

* Each character is a token.
* Small vocabulary, but long sequences.
### Examples of Tokenizers in NLP Models
#### Traditional Word Embedding Models

| Model        | Tokenization Method | Notes                                     |
| ------------ | ------------------- | ----------------------------------------- |
| **GloVe**    | Word-level          | Fixed vocabulary; OOV words are an issue. |
| **Word2Vec** | Word-level          | No subword handling; large vocab needed.  |

These models treat each word as a distinct token. Once trained, they assign a dense vector to each token in the vocabulary.
#### Modern Transformer-Based Models

| Model           | Tokenization Method        | Notes                                                                       |
| --------------- | -------------------------- | --------------------------------------------------------------------------- |
| **BERT**        | WordPiece                  | Splits rare words into subwords; e.g., “playing” → “play”, “##ing”.         |
| **GPT (2/3/4)** | BPE                        | Efficient subword tokenization; balances vocab size with language coverage. |
| **T5**          | SentencePiece (Unigram LM) | Fully unsupervised; language-independent approach.                          |
| **LLaMA**       | SentencePiece (BPE)        | Uses BPE; optimized for open-domain tasks in low-resource settings.         |

These models use subword tokenization to handle rare or unknown words while maintaining efficient and expressive representations.
## Padding
### Why is it necessary?

Padding ensures that all input sequences in a batch are of uniform length. This is important because:
1. Models usually process multiple sequences at once (in batches), and most neural networks expect each input in a batch to have the same length.
2. Fixed-length inputs allow for parallel computation, making the training and inference processes more efficient.
### How it Works?

In text data, sequences can have varying lengths (e.g., one sentence might have 10 tokens, while another might have 15).

Padding adds tokens (e.g., a special token like [PAD]) to the shorter sequences until they reach the length of the longest sequence in the batch (or a predefined maximum length).
### Types of Padding

Pre-padding: Padding tokens are added to the beginning of the sequence. This is usually used for decoder-based models.

Post-padding: Padding tokens are added to the end of the sequence. This is generally used for encoder seq2seq models.
### Impact on training models

Padding tokens usually do not carry any useful information for the model, so special attention is needed to ensure that these tokens don't influence the learning process. This is often handled by masking out the padded tokens (e.g., setting their corresponding labels to -100) in models, so they don't contribute to loss calculations during training.
## Training Guide
### Choosing the Right Model

* **Pretrained Models**: Start with a pretrained transformer model. They often perform well on a wide range of tasks. If you have limited resources, use smaller variants for faster training and inference.

* **Task-Specific Models**: Some models are better suited for specific tasks. For example:

  * Encoder models are excellent for tasks like classification, named entity recognition, or information retrieval.
  * Encoder-Decoder (Sequence-to-Sequence) models are great for tasks like text summarization and machine translation.
  * Decoder models are suited for generative tasks like text generation and completion.
  
* **Fine-tuning vs. Training From Scratch**: Fine-tuning a pretrained transformer model is typically faster and more effective than training from scratch, especially with smaller datasets.
### Fine-tuning Best Practices

* **Learning Rate**: When fine-tuning a transformer model, use a smaller learning rate (e.g., 2e-5 to 5e-5). Transformer models are sensitive to the learning rate, and larger values can cause catastrophic forgetting of the pretrained weights.
* **Optimizer**: Use the AdamW optimizer, which includes weight decay to regularize the model and helps prevent overfitting. It is commonly used for transformer-based models.
* **Warmup Steps**: Implement learning rate warm-up for the first part of training (e.g., 5% of total training steps). This helps stabilize training when fine-tuning large models.
* **Batch Size**: A batch size of 16, 32 or 64 is common for transformer-based models. Due to memory constraints, batch sizes may need to be smaller for larger models. Ensure that your batch size fits within your GPU memory limits.
* **Gradient Accumulation**: If you cannot fit a large batch size into memory, use gradient accumulation. This technique simulates a larger batch size by accumulating gradients over multiple smaller batches before updating the model's weights.
* **Early Stopping**: Monitor validation loss or another relevant metric during training and stop early if the model's performance plateaus or starts to degrade.
### Handling Long Sequences

* **Sequence Length**: Transformer models, have a maximum sequence length (e.g., 512 tokens for BERT). Ensure that you don't exceed this length when processing long documents.
* **Truncation and Padding**: If your sequences exceed the model's maximum length, truncate them. If they are too short, pad them. Be careful with truncation as you might lose important information. You can try splitting long documents into smaller chunks.
* **Sliding Window**: For tasks like document classification, you can use a sliding window approach to process longer texts. This involves dividing the text into overlapping windows and aggregating results.
### Handling Imbalanced Data

* **Class Weights**: For classification tasks with imbalanced classes, consider adjusting class weights during training. This will help the model pay more attention to the minority classes.
* **Sampling Techniques**: Use oversampling or undersampling techniques to balance the dataset. Alternatively, use data augmentation methods to synthetically generate additional examples for the minority class.
### Evaluation and Hyperparameter Tuning

* **Evaluation Strategy**: Regularly evaluate the model on a validation set during training. Make sure you're not overfitting to the training data, and you can also track performance metrics like F1-score, precision, and recall, depending on the task.
* **Hyperparameter Tuning**: Tune hyperparameters like the learning rate, batch size, and weight decay. These can be done through methods like grid search or random search, or Bayesian optimization for more efficient hyperparameter exploration. Two packages you might be interested in for hyperparameter tuning are Optuna and Ray-Tune.
###Dealing with Overfitting

* **Dropout**: Use dropout in the transformer model to prevent overfitting, especially for larger models.
* **Regularization**: Apply weight decay (as part of the AdamW optimizer) to avoid overfitting by penalizing overly large weights.
* **Data Augmentation**: For tasks like text classification, augment the data using techniques like backtranslation, or easy data agumentation(EDA).