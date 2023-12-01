# Custom-Large-Language-Model

</br>
<div align="center">
<a href="https://www.python.org/"><img src="./readme_files/Python.png" width="75" height="75"></a>
<a href="https://pytorch.org/"><img src="./readme_files/Pytorch.png" width="75" height="75"></a>
<a href="https://jupyter.org/"><img src="./readme_files/Jupyter.png" width="75" height="75"></a>
</div>

</br>

## Overview

- This repository is the result of following this tutorial from Freecodecamp regarding how to build a large language model from scratch using Python
- The purpose of following [this tutorial](https://www.youtube.com/watch?v=UU1WVnMk4E8) was to learn more about large language models, how they work, how they are built, and the machine learning processes required of them
- The datasets used in the creation of these models are not included in this repository, due to their file size

## Model Architecture

### Head Class

The Head class represents an individual head in a multi-head self-attention mechanism, a key component of the Transformer architecture. During initialization, it sets up linear transformations for key, query, and value projections, and employs a buffer for a lower-triangular matrix to ensure causality in the attention mechanism. The forward method takes an input tensor of shape (batch, time-step, channels) and computes the self-attention scores. Key, query, and value projections are created and used to calculate attention scores, which are then adjusted for causality and normalized using a softmax function. The resulting attention weights are applied to the values, producing an output tensor of shape (batch, time-step, head size). The class incorporates dropout for regularization during training.

The Head class encapsulates the operations involved in self-attention, providing a modular and reusable component for building multi-head attention layers within the Transformer model. It demonstrates the key steps in attention computation, including the use of linear projections and the application of attention weights to input values. The class also emphasizes the importance of dropout for enhancing model generalization.

### MultiHeadAttention Class

The MultiHeadAttention class implements the parallel application of multiple self-attention heads within the Transformer architecture. During initialization, it creates a ModuleList containing individual Head instances based on the specified number of heads and head size. Additionally, the class incorporates a linear projection layer to combine the outputs of the individual heads, followed by dropout regularization for improved model robustness. The forward method takes an input tensor of shape (batch, time-step, features) and applies each self-attention head independently. The outputs from all heads are concatenated along the last dimension, and a linear projection is applied to produce the final multi-head attention output.

This class serves as a crucial component in the Transformer model, enabling the model to capture diverse patterns and relationships within the input data simultaneously through parallel self-attention mechanisms. The use of ModuleList ensures the encapsulation and easy management of individual attention heads, facilitating modularity and reusability. The linear projection layer helps consolidate information from multiple heads, and the incorporation of dropout further enhances the generalization capabilities of the model during training.

### FeedForward Class

The FeedForward class defines a feedforward neural network module within the Transformer architecture, consisting of two linear layers separated by a rectified linear unit (ReLU) activation function. During initialization, the class takes the embedding dimension (n_embd) as a parameter, creating a sequential neural network module (self.net) with two linear layers. The first linear layer transforms the input tensor from n_embd dimensions to 4 times n_embd dimensions, followed by a ReLU activation function. Subsequently, the second linear layer reduces the tensor back to n_embd dimensions. Additionally, a dropout layer is included after the second linear transformation, introducing regularization to prevent overfitting during training. The forward method takes an input tensor x and passes it through the sequential network, returning the final output.

This class serves as a vital component in the Transformer model, providing non-linearity and expressive power to the network. The incorporation of the ReLU activation function introduces non-linear transformations to the input data, enabling the model to capture complex patterns and relationships. The inclusion of dropout further enhances the generalization capabilities of the model, making it more robust and preventing overfitting. Overall, the FeedForward module contributes to the flexibility and effectiveness of the Transformer architecture in handling various tasks, including natural language processing and sequence modeling.

### Block Class

The Block class represents a fundamental building block within the Transformer model, encapsulating the essential components for effective sequence processing. Initialized with parameters for embedding dimension (n_embd) and the desired number of attention heads (n_head), this class orchestrates communication and computation processes. The embedding dimension is divided by the number of heads to determine the head size, a crucial factor in the subsequent MultiHeadAttention module's initialization. Additionally, the Block class contains instances of MultiHeadAttention (self.sa) and FeedForward (self.ffwd) modules, providing attention mechanisms and non-linear transformations to the input sequence, respectively. Two Layer Normalization (LN) modules, self.ln1 and self.ln2, are employed to normalize the outputs after the attention and feedforward stages, contributing to stable training dynamics.

During the forward pass, input tensor x undergoes a self-attention operation (y = self.sa(x)), followed by layer normalization (x = self.ln1(x + y)). Subsequently, the output is processed through a feedforward neural network (y = self.ffwd(x)), and the result is once again normalized (x = self.ln2(x + y)). This structure emphasizes the Transformer's design philosophy, where attention mechanisms facilitate communication between sequence elements, and feedforward networks introduce non-linearity for enhanced modeling capabilities. The use of layer normalization ensures that the outputs maintain a consistent scale throughout the network, contributing to stable training and improved generalization. Overall, the Block class encapsulates the core operations essential for effective sequence modeling in the Transformer architecture.

### GPTLanguageModel Class

The GPTLanguageModel class defines a Generative Pre-trained Transformer (GPT) language model, which is a neural network designed for natural language processing tasks. Initialized with the vocabulary size (vocab_size), the class sets up essential components for sequence modeling. It includes an embedding table for tokens (self.token_embedding_table) and another for positional information (self.position_embedding_table). The model's core architecture consists of a stack of Transformer blocks (self.blocks), where each block is an instance of the previously defined Block class. Additionally, layer normalization (self.ln_f) and a linear layer (self.lm_head) for generating output logits are incorporated. The \_init_weights method ensures appropriate weight initialization for linear and embedding layers, contributing to effective training.

During the forward pass, the input index tensor undergoes token and positional embedding transformations, followed by processing through the stack of Transformer blocks. Layer normalization is applied to the output, and final logits are generated using the linear layer. If targets are provided, the method computes the cross-entropy loss between the predicted logits and target indices. The generate method extends the model's capabilities by generating new tokens based on the provided context index. It samples tokens iteratively, focusing on the last time step, and appends them to the running sequence. This functionality enables the GPT language model to generate coherent and contextually relevant sequences of text, making it suitable for various natural language generation tasks.

## Repo Contents

### gpt-v1.ipynb

This file represents the implementation of a language model using PyTorch and the Transformer architecture. The model is designed for training on text data, particularly focusing on self-attention mechanisms within the Transformer blocks. It incorporates elements such as argument parsing, GPU availability check, and the creation of a language model with specific hyperparameters like batch size, block size, and learning rate. The training loop involves estimating the loss, updating model parameters using the AdamW optimizer, and saving the model periodically. Additionally, there is a function for generating text based on a given prompt.

The script involves the creation of a MultiHeadAttention module, a FeedForward module, and a GPTLanguageModel module, which includes the main Transformer blocks. The model is trained on text data from the OpenWebText dataset, with the training process involving random chunk sampling and batch creation. The generated language model can be saved and later used for text generation based on a user-provided prompt. The script provides a comprehensive example of implementing a Transformer-based language model using PyTorch for natural language processing tasks.

### data-extract.py

This file performs preprocessing on the OpenWebText dataset, specifically focusing on extracting and splitting compressed files, generating training and validation sets, and creating a vocabulary file. The script begins by identifying files with the ".xz" extension in the specified directory. It calculates a split index to allocate 90% of the files for training and the remaining 10% for validation. The text content of these files is processed and concatenated, while simultaneously updating a vocabulary set with unique characters encountered during the processing.

The training and validation sets are written to separate output files, namely "output_train.txt" and "output_val.txt," respectively. Additionally, the script generates a "vocab.txt" file containing the unique characters present in the processed text. The "lzma" library is employed for reading compressed files, and the "tqdm" library is used to display progress bars during the file processing steps. Overall, the script efficiently handles the extraction, split, and vocabulary generation tasks for the OpenWebText dataset, facilitating subsequent use in natural language processing applications.

### training.py

This script revolves around a GPT (Generative Pre-trained Transformer) language model implementation using PyTorch. The script includes the definition of the model architecture, training loop, and saving of the trained model. It utilizes an argparse module to specify and parse command-line arguments, with a particular focus on the batch size. The model parameters, such as embedding dimension, number of heads, layers, and dropout rate, are configured along with optimization-related parameters like learning rate.

The GPT model comprises self-attention mechanisms, feedforward layers, and transformer blocks. It employs memory mapping to efficiently handle large text files during training. Additionally, the script demonstrates the training process, evaluation, and periodic output of training and validation losses. The model is saved after training completion. It's noteworthy that parts of the script related to loading a pre-trained model from a file are commented out, implying the potential for model reusability. Overall, the script provides a comprehensive implementation of a GPT language model and its training procedure.

### chatbot.py

This script implements a GPT (Generative Pre-trained Transformer) language model using PyTorch. The script starts by defining command-line arguments, where the batch size is specified using argparse. It then checks for the availability of a CUDA-enabled GPU and sets the device accordingly. The script initializes various parameters such as embedding dimension, number of heads, layers, and dropout rate.

The GPT model architecture is composed of self-attention heads, a feedforward layer, and transformer blocks. The model incorporates memory mapping to handle large text files efficiently. The training and generation processes are demonstrated. For training, the script loads a pre-trained model, defines a PyTorch optimizer, and iterates through training steps. The training loop also includes periodic evaluation and output of training and validation losses. After training, the script saves the trained model parameters to a file.

In the interactive section, the script enters a loop that continuously prompts the user for input. The user can provide a prompt, and the model generates a completion based on the input prompt. The completion is displayed, allowing users to interactively explore the model's language generation capabilities.

## Papers Referenced

### [Attention is All You Need](https://arxiv.org/pdf/1706.03762.pdf)

### [A Survey of LLMs](https://arxiv.org/pdf/2303.18223.pdf)

### [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/pdf/2305.14314.pdf)
