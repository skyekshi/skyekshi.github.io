---
layout: post
title: "Efficient fine-tuning methods for large language model"
author: "Letian Shi, Keyan Shi"
---

# Efficient Fine-tuning Methods for Large Language Model

Large language models (LLMs) revolutionize natural language processing (NLP) through their advanced abilities and complex solutions. These models are built using deep learning techniques, particularly neural networks with billions of parameters, enabling them to capture the nuances of language, context, and even subtle patterns in data. Trained on extensive text datasets, these models can handle various tasks such as text generation, translation, summarization, and answering questions. 

Most LLMs exhibit high levels of generalization, enabling
them to apply their acquired knowledge to new tasks not included in their original training. This capability is often referred to as _zero-shot learning_. However, LLMs may fall short when it comes to specific task-oriented issues. This leads to the fact that fine-tuning is crucial for further improving LLMs to achieve optimal performance on new user datasets and tasks without the need to develop new models from scratch.

Unfortunately, as LLMs increase in size, the costs associated with fine-tuning and storing all parameters become prohibitively expensive and eventually practically unfeasible.
In this blog, we explore why classical fine-tuning LLMs cannot work well and discuss how different _efficient_ fine-tuning methods become a critical component of LLM-powered solutions.

## Classical Techniques to Update Pretrained LLM Weights for Fine-tuning

### Finetune Whole Model Parameters

If conditions permit, using the user's own data to update the weights of pretrained LLMs will naturally yield the best results. 
However, as shown in the table below, popular open-source LLMs today are characterized by large model sizes, a high number of parameters, and a significant number of tokens.

| **Model** | **# Parameters** | **# Tokens** |
|-----------|------------------|--------------|
| **LLaMA2** | **70B**          | **2T**       |
| **PAML2**  | **340B**         | **3.6T**     |
| **MPT**    | **7B**           | **1T**       |
| **Falcon** | **180B**         | **1.5T**     |
| **Gemini** | **3.25B**        | **5.5T**     |

As shown in the table above, LLMs are larger in scale compared to other models. Therefore, fine-tuning large language models requires significant computational power, which may not be available on standard PCs or regular servers. For example, the GeForce 4080 only has 16GB GPU memory can only put a Geimini model, but it is impossible to train anymore on this device.

One may suggest working with transfer learning to resolve the limitations, while fine-tuning the entire model requires considerable computational resources, and it has a typical trade-off with model performance.

### Transfer Learning

Transfer learning (also so known as linear probing) is the method that we freeze nearly all the layers of the pretrained model and only train the last layer during the fine-tuning process. The last layer maybe change for different tasks. If the downstream tasks is "Yes/No" answering question, the last layer only has 2 dim. Moreover, if the new tasks is text generalization, the last layer is the number of used tokenizers in the model.

![](/images/1.png)

Transfer learning works very well in CV models but not very well in LLMs. The main reason is the domain mismatch. There are many downstrem tasks for the LLMs. For instance, classification, multiple choice, text generation, translation, personalized chat-bots, and summarization. Even though the pre-trained model is carefully trained on a large amount of data, we cannot guarantee that the similarity between the data used for pre-training and the data for the user's own task. Only modifying paramters of one layers or several lasting layers are not able to bridge the difference between the original and new tasks.

### In-context Learning (Prompt Engineering)

Prompts are typically used as means to interact with LLMs, where users provide to which the model is to respond. A prompt can be texts, images, videos, etc. in different use cases. Prompt engineering, in turn, refers to the process of designing the prompts for target outputs. For example, we can ask:

_Q: What is the capital of Germany?_<br>
_A: <mark>Germany.</mark>_

The model generated incorrect response to our question in this _zero-shot_ prompt, which indicate that it might not understand the word "capital" using its existing knowledge. A simple way to overcome this would be providing a few examples before asking the question:

_Q: What is the capital of Japan?_<br>
_A: Tokyo._<br>
_Q: What is the capital of the US?_<br>
_A: Washington._<br>
_Q: What is the capital of Germany?_<br>
_A: <mark>Berlin.</mark>_

Hence, the _few-shot_ prompt enables the model to learn without parameter tuning. However, one can already spot some shortcomings of prompt engineering in the process - a few examples must be pre-appended, affecting the token budget.

## Parameter Efficient Fine-tuning (PEFT)

Because of the size of LLMs, a commonly used approach to fine-tune LLMs involves modifying a subset of LLM parameters while leaving the rest unchanged. This method, known as _Parameter-Efficient Fine-Tuning_ (PEFT), selectively adjusts a small fraction of the parameters while maintaining the majority unchanged. 
Specifically, PEFT involves modifying the parameters of a pre-trained large model to tailor it for a particular task or domain, aiming to minimize the introduction of additional parameters or computational resources needed.

There are three types of PEFT algorithms. 
1) **Additive PEFT**: add new trainable modules or parameters.
2) **Selective PEFT**: select a subset of parameters trainable during fine-tuning,
3) **Reparameterization PEFT**: create a reparameterization of the original model parameters for training and then converts it back to its original form for inference.

### 1. Additive PEFT
Standard full fine-tuning involves significant computational costs.  To address this issue, a common strategy is to keep the pre-trained model intact and add only a small set of trainable parameters strategically placed within the model architecture. During fine-tuning for a specific downstream task, only the weights of these additional parameters are updated.

#### 1.1. Adapter 
Adapter methods entail adding small adapter layers into Transformer blocks. 

#### Parameter-Efficient Transfer Learning for NLP (Houlsby et al., 2019)


**Parameter-Efficient Transfer Learning** for NLP is a **serial adapter**. The adapter layers mostly consist of two multiple layer perceptrons (MLPs) and activation functions. The formulation is shown below.

<center>
$$ 
Adapter(x) = W_{up}\sigma(W_{down}x) + x
$$
</center>
where \\(W_{down} \in \mathbb{R}^{d\times r}\\), \\(W_{up} \in \mathbb{R}^{r\times d}\\) with rank \\(r \ll d\\). 

d represents the dimension of the hidden layer, and r serves as the bottleneck dimension. These details shown in the left picture below.

![](/images/2.png) 

In the method, each Transformer block is augmented by adding two adapter modules: one placed after the self-attention layer and the other after the feed-forward network (FFN) layer.

The adventages of serial adapter are: 1) By adding a few trainable parameters, it is possible to nearly achieve the effect of full parameter training. 2) This significantly reduces training time, ensuring time efficiency. 3) Barely decreases the model's performance on downstream tasks.

To demonstrate the effectiveness of adapters, they transfer the recently proposed BERT Transformer model to 26 diverse text classification tasks, including the GLUE benchmark. Adapters achieve near best performance while adding only a few parameters per task. On GLUE, they attained performance within 0.4% of full fine-tuning by adding only 3.6% parameters per task.

There are also many additional models related to adapters in the works. 
**Adapter Fusion (Pfeiffer et al., 2020)** was proposed, where adapter layers are inserted only after the ’Add & Norm’ step following the FFN layer.
To avoid placing adapter layers as bottlenecks
within the Transformer blocks, **Parallel Adapter (He et al. 2021)** change sequential adapter into parallel side-network that deploys alongside each Transformer sublayer. In addition to the parallel design, **CoDA (Lei et al., 2023)** utilizes a sparse activation mechanism to enhance inference efficiency.

#### 1.2 Soft Prompt
Alternatively, prompt tuning offers another method to finetune the model for enhanced performance. This can be represented
as follows:

<center>
$$ 
\mathbf{X}^{(l)} = [\mathbf{s}_1^{(l)}, \ldots, \mathbf{s}_{N_s}^{(l)}, \mathbf{x}_1^{(l)}, \ldots, \mathbf{x}_{N_x}^{(l)}]
$$
</center>
where \(\mathbf{X}^{(l)}\) is the sequence of input tokens for layer \(l\), including soft prompt tokens \(\mathbf{s}_i^{(l)}\) followed by the original input tokens \(\mathbf{x}_i^{(l)}\). \(N_s\) is the number of soft prompt tokens, and \(N_x\) is the number of original input tokens.

#### Prefix-tuning (Li & Liang, 2021)

Different from all Transformer parameters (the red Transformer box in the following picture) and storing a full model copy for each task, **Prefix-tuning** introduces a light alternative approach to fine-tuning for NLP tasks, which keeps pretrained language model parameters frozen, but optimizes a small _continuous task-specific_ vector known as prefix (the red blocks below).  Prefix-tuning is inspired by prompting, enabling subsequent tokens to interact with this prefix as though they were "virtual tokens." 

![](/images/3.png) 

Let's use an example to demonstrate how to apply the prefix-tuning method. In the following graph, we put the prefix encoders before the source and target encoder as prefix setting can be done without being affected by the length of the sentence and padding. In the example, $h_1$ and $h_2$ are prefix encoders. $z=[\text{prefix}, x, y]$ is the final tokenizers after prefix-tuning.

![](/images/4.png) 

P$_{\text{idx}}$ denotes the sequence of prefix indices. The training objective is to maximize p(y|x).  The language model parameters $\phi$ are fixed and the prefix parameters $\theta$ are the only trainable parameters. Thus, any hidden states is a function of the trainable model P$_{\text{idx}}$. 

$$
\begin{align}
h_{i} = 
\begin{cases}
P_{\theta}[i,:],      & \text{if } i \in P_{\text{idx}} \text{,}\\
LM_\phi(z_i, h_{<i}),    & \text{otherwise.}
\end{cases}
\end{align}
$$

**Prefix-tuning** adopts an MLP layer to generate these prefix vectors rather than
optimizing them directly. After fine-tuning, only the prefix vectors are saved for inference. Besides, **Prompt-tuning (Lester et al., 2021)** apply learnable vectors only at the initial word embedding layer rather than all layers to enhance training and inference efficiency. 

#### P-tuning (Liu et al., 2023)

P-tuning, or _prompt tuning_, is a parameter efficient tuning technique to solve this challenge. P-tuning uses a small trainable model before the LLM. The small model encodes the text prompt and creates task-specific virtual tokens. These tokens are pre-appended to the prompt and then passed to the LLM. Once tuning is finished, the virtual tokens are saved in a lookup table and used during inference, taking the place of the smaller model.

![](/images/para.png)

Therefore, per set up of P-tuning naturally gives us a continuous way to feed prompts into the frozen model. The trainable continuous prompt embeddings can be further concatenated with discrete prompts to achieve better performance. We take an example to illustrate the process (notice we are restricted to NLU, or _Natural Language Processing_, tasks in the following discourse):

_Q: Where is Berlin located?_<br>
_A: Germany._

Berlin is labeled as [X] and Germany [Y] here. If we tweak the language a bit:

![](/images/discrete.png)

We can see a big drop of performance in some of the modification cases, while with P-tuning the result is more stable. One may ask, what exactly is the magic wand - how are the differentiable virtual tokens introduced?

![](/images/comp.png)

So basically \\([P_i]\\) is the continuous prompt embedding that needs to be learned, and \\(h_i\\) is the input to be fed to the model. The prompt encoder that maps \\([P_i]\\) to \\(h_i\\) can be anything, while they experimented the popular choices with LSTM giving the best performance.

#### P-tuning v2 (Liu et al., 2022)

Although P-tuning has demonstrated advantages in some NLU benchmarks, it performs worse than fine-tuning for hard sequence labeling tasks (e.g., extractive question answering), showing lack of generality.

In the previous set up, continuous prompts are only inserted into the input embedding sequence of the first layer of the transformer. In subsequent transformer layers, the embedding at the inserted continuous prompt position is calculated by the previous transformer layer, which may lead to two potential optimization challenges:

* The number of tunable parameters is limited due to the restriction on sequence length.<br>
* Input embeddings have only a relatively indirect impact on model prediction.

Therefore, P-tuning v2 is a deep prompt optimization of P-tuning to resolve the generalization issue, where they basically add prompts in different layers are added only as prefix tokens.

![](/images/v2.png)

Then by design P-tuning v2 has more tunable task-specific parameters and more direct impact on model predictions. The following experiments summarize the performances of fine-tuning, P-tuning, and P-tuning v2 across different model scales and different NLU tasks. 

![](/images/result.png)

### 2. Selective PEFT
Instead of additive PEFT, which increases model complexity by introducing additional parameters, selective PEFT fine-tunes a subset of the existing parameters to improve model performance on downstream tasks.

**Selective PEFT** involves modifying a model's parameters $\theta = \{\theta_1, \theta_2, ..., \theta_n\}$, where each $\theta_i$ represents an individual parameter and $n$ denotes the total number of parameters. This method utilizes a binary mask $M = \{m_1, m_2, ..., m_n\}$ applied to these parameters. Each $m_i$ in $M$ is binary (0 or 1), indicating whether the corresponding parameter $\theta_i$ should be included (1) or excluded (0) during fine-tuning. The updated parameter set after fine-tuning is computed as:
\[
\theta_{i+1} = \theta_i - \eta \cdot m_i \cdot \frac{\partial \mathcal{L}}{\partial \theta_i}
\]
Here, $\eta$ denotes the learning rate, and $\frac{\partial \mathcal{L}}{\partial \theta_i}$ represents the gradient of the loss function $\mathcal{L}$ with respect to $\theta_i$. During backpropagation, only the parameters that are selected ($m_i = 1$) are updated, optimizing the model effectively while minimizing computational overhead.

**Diff pruning (Guo et al., 2020)** is an influential study that employs a trainable binary mask on model weights during fine-tuning. To enhance parameter efficiency, the mask is controlled using a differentiable approximation of the $L_0$-norm penalty. **PaFi (Liao et al., 2023)** simply select the smallest absolute value model parameters as trainable ones. **SAM (Fu et al., 2023)** introduces a second-order approximation method to assist in determining the parameter mask. This method approximates the original problem using an optimization function that can be solved analytically.


### 3. Reparamatrization PEFT

Reparameterization involves converting a model's architecture from one form to another by transforming its parameters. In the context of PEFT, this typically means creating a low-rank parameterization to enhance parameter efficiency during training. For inference, the model can be reverted to its original weight parameterization, maintaining the same inference speed.

To be specific, if we view the fine-tuning problem as minimizing the target loss function \\(L(X,y;W+\Delta W)\\) where \\(X\\) and \\(y\\) are the data, and \\(W\\) the pre-trained model parameters, the tuning parameters \\(\Delta W\\) could be approximated.

#### LoRA (Hu et al., 2021)
LoRA, or _Low-Rank Adaptation_, by its name uses low rank approximation \\(\Delta\Phi \approx \Delta W\\) with \\(|\Delta\Phi| \ll |\Delta W|\\). The observation is that neural nets have many dense layers performing matrix multiplication, and while they typically have full-rank during pre-training, when adapting to a specific task the weight updates will have a low “intrinsic dimension”.

Considering the update \\(\Delta w_i \in \Delta w\\) (\\(\in \Delta W\\)) for the \\(i\\) th weight in the network, LoRA approximates it with: 

<center>
$$ 
\Delta w_i \approx  \Delta\phi_i = BA
$$
</center>

where \\(B\in \mathbb{R}^{d\times r}\\), \\(A\in \mathbb{R}^{r\times d}\\) with rank \\(r \ll d\\). Under this decomposition the computation reduces significantly from \\(d\times d\\) to \\(d\times r+r\times d\\). The following graph illustrates the process and the respective initialization of matrices \\(B\\) and \\(A\\).

![](/images/lora.png)

During the training, the pretrained model (blue part) will be frozen, while the reparameterization module (yellow part) will adjust parameters during fine-tuning. During the inference, the two matrices within the reparameterization module will be multiplied and merged to the original paramter matrix, achieving efficient results.

Several subsequent researches aim to improve LoRA’s performance in various aspects. **Laplace-LoRA (Yang et al., 2023)** applies a Bayesian approach, specifically a Laplace approximation, which predicts the posterior of the LoRA to aviod overfitting. **LoRA+ (Hayou et al., 2024)** proposes to set different learning rates for the LoRA matrices A and B seperatively. **MoSLoRA (Wu et al., 2024a)** decomposes LoRA into subspaces using structural reparameterization. It then utilizes a trainable mixer, trained alongside the original LoRA weights, to blend these subspaces together.

## Efficient Optimization Method



## Conclusion


## Reference
[1] Parameter-Efficient Fine-Tuning for Large Models: A Comprehensive Survey <br>
[2] Large Language Models: A Survey <br>
[3] Parameter-Efficient Transfer Learning for NLP <br>
[4] Prefix-Tuning: Optimizing Continuous Prompts for Generation <br>
[5] The Power of Scale for Parameter-Efficient Prompt Tuning <br>
[6] GPT Understands, Too <br>
[7] P-Tuning v2: Prompt Tuning Can Be Comparable to Fine-tuning Universally Across Scales and Tasks <br>
[8] Parameter-Efficient Transfer Learning with Diff Pruning <br>
[9] LoRA: Low-Rank Adaptation of Large Language Models <br>