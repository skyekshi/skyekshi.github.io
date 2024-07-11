---
layout: post
title: "Efficient fine-tuning methods for large language model"
author: "Letian Shi, Keyan Shi"
---

# Efficient fine-tuning methods for large language model

Below is the content.

## Additive

### Adpater tuning

#### 1. Parameter-Efficient Transfer Learning for NLP

### Prompt engineering

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

One may suggest working with transfer learning to resolve the limitations, while fine-tuning the entire model (BERT-base 110-345M, BERT-large 340-770M, GPT-3 175B paratemers) requires considerable computational resources, and it has a typical trade-off with model performance.

#### 2. Prefix-tuning 

#### 3. Prompt tuning

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

## Reparamatrization

Another approach to reduce compute resource consumption is to simplify the tuning process through reparamatrization. To be specific, if we view the fine-tuning problem as minimizing the target loss function \\(L(X,y;W+\Delta W)\\) where \\(X\\) and \\(y\\) are the data, and $W$ the pre-trained model parameters, the tuning parameters \\(\Delta W\\) could be approximated.

#### LoRA (Hu et al., 2021)
LoRA, or _Low-Rank Adaptation_, by its name uses low rank approximation \\(\Delta\Phi \approx \Delta W\\) with \\(|\Delta\Phi| \ll |\Delta W|\\). The observation is that neural nets have many dense layers performing matrix multiplication, and while they typically have full-rank during pre-training, when adapting to a specific task the weight updates will have a low “intrinsic dimension”.

Considering the update \\(\Delta w_i \in \Delta w\\) (\\(\in \Delta W\\)) for the \\(i\\) th weight in the network, LoRA approximates it with: 
$$ 
\Delta w_i \approx  \Delta\phi_i = BA
$$

where \\(B\in \mathbb{R}^{d\times r}\\), \\(A\in \mathbb{R}^{r\times d}\\) with rank \\(r \ll d\\). Under this decomposition the computation reduces significantly from \\(d\times d\\) to \\(d\times r+r\times d\\). The following graph illustrates the process and the respective initialization of matrices \\(B\\) and \\(A\\).

![](/images/lora.png)

## Efficient optimization method



