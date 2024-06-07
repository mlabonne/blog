---
aliases:
- /frankenmoe/
categories:
- Large Language Models
colab: <a href="https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb?usp=sharing"><img src="images/colab.png" alt="Open In Colab"></a>
date: '2024-03-24'
image: /images/frankenmoe/thumbnail.jpg
title: "Create Mixtures of Experts with MergeKit"
subtitle: "Combine multiple experts into a single frankenMoE"
---

<center><img src="/images/frankenmoe/thumbnail.jpg"></center>

::: {.column-margin}
Find many more architectures and applications using graph neural networks in my book, [**Hands-On Graph Neural Networks**](https://mlabonne.github.io/blog/book.html) 👇
<a href="https://packt.link/a/9781804617526"><img src="/images/gnnbook/cover.png" alt="Hands-On Graph Neural Networks Using Python" id="gnn-book"></a>
:::

Thanks to the release of Mixtral, the **Mixture of Experts** (MoE) architecture has become popular in recent months. This architecture offers an interesting tradeoff: higher performance at the cost of increased VRAM usage. While Mixtral and other MoE architectures are pre-trained from scratch, another method of creating MoE has recently appeared. Thanks to Arcee's [MergeKit](https://github.com/arcee-ai/mergekit) library, we now have a new way of creating MoEs by ensembling several pre-trained models. These are often referred to as **frankenMoEs** or **MoErges** to distinguish them from the pre-trained MoEs.

In this article, we will detail how the MoE architecture works and how frankenMoEs are created. Finally, we will make our [own frankenMoE](https://huggingface.co/mlabonne/Beyonder-4x7B-v3) with MergeKit and evaluate it on several benchmarks. The code is available on Google Colab in a wrapper called [LazyMergeKit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb#scrollTo=d5mYzDo1q96y).

Special thanks to [Charles Goddard](https://github.com/cg123), the creator of MergeKit, for proofreading this article.

## 🔀 Introduction to MoEs

A Mixture of Experts is an architecture designed for improved efficiency and performance. It uses multiple specialized subnetworks, known as "**experts**." Unlike dense models, where the entire network is activated, MoEs only activate relevant experts based on the input. This results in faster training and more efficient inference.

There are two components at the core of an MoE model:

1. **Sparse MoE Layers**: These replace the dense feed-forward network layers in the transformer architecture. Each MoE layer contains several experts, and only a subset of these experts are engaged for a given input.
2. **Gate Network or Router**: This component determines which tokens are processed by which experts, ensuring that each part of the input is handled by the most suitable expert(s).

In the following example, we show how a Mistral-7B block is transformed into an MoE block with a sparse MoE layer (feedforward network 1, 2, and 3) and a router. This example represents an MoE with three experts, where two are currently engaged (FFN 1 and FFN 3).

![](https://i.imgur.com/9Fl8xxD.png)

MoEs also come with their own set of challenges, especially in terms of fine-tuning and memory requirements. The fine-tuning process can be difficult due to the model's complexity, with the need to **balance expert usage** during training to properly train the gating weights to select the most relevant ones. In terms of memory, even though only a fraction of the total parameters are used during inference, the entire model, including all experts, needs to be **loaded into memory**, which requires high VRAM capacity.

More specifically, there are two essential parameters when it comes to MoEs:

* **Number of experts** (`num_local_experts`): This determines the total number of experts in the architecture (e.g., 8 for Mixtral). The higher the number of experts, the higher the VRAM usage.
* **Number of experts/token** (`num_experts_per_tok`): This determines the number of experts that are engaged for each token and each layer (e.g., 2 for Mixtral). There is a tradeoff between a high number of experts per token for accuracy (but diminishing returns) vs. a low number for fast training and inference.

Historically, MoEs have underperformed dense models. However, the release of [Mixtral-8x7B](https://arxiv.org/abs/2401.04088) in December 2023 shook things up and showed impressive performance for its size. Additionally, GPT-4 is also rumored to be an MoE, which would make sense as it would be a lot cheaper to run and train for OpenAI compared to a dense model. In addition to these recent excellent MoEs, we now have a new way of creating MoEs with MergeKit: frankenMoEs, also called MoErges.

## 🧟‍♂️ True MoEs vs. frankenMoEs

The main difference between true MoEs and frankenMoEs is how they're trained. In the case of true MoEs, the experts and the router are trained jointly. In the case of frankenMoEs, we upcycle existing models and initialize the router afterward.

In other words, we copy the weights of the layer norm and self-attention layers from a base model, and then copy the weights of the FFN layers found in each expert. This means that besides the FFNs, all the other parameters are shared. This explains why Mixtral-8x7B with eight experts doesn't have 8*7 = 56B parameters, but about 45B. This is also why using two experts per token gives the inference speed (FLOPs) of a 12B dense model instead of 14B.

FrankenMoEs are about selecting the most relevant experts and initializing them properly. MergeKit currently implements three ways of initializing the routers:

1. [**Random**](https://github.com/arcee-ai/mergekit/blob/9c691527f7192b5a2fc388555bfd3105e0898480/mergekit/scripts/mixtral_moe.py#L139-L142): Random weights. Be careful when using it as the same experts might be selected every time (it requires further fine-tuning or `num_local_experts = num_experts_per_tok`, which means you don't need any routing).
2. [**Cheap embed**](https://github.com/arcee-ai/mergekit/blob/9c691527f7192b5a2fc388555bfd3105e0898480/mergekit/scripts/mixtral_moe.py#L91C1-L109C37): It uses the raw embeddings of the input tokens directly and applies the same transformation across all layers. This method is computationally inexpensive and suitable for execution on less powerful hardware.
3. [**Hidden**](https://github.com/arcee-ai/mergekit/blob/9c691527f7192b5a2fc388555bfd3105e0898480/mergekit/scripts/mixtral_moe.py#L70-L88): It creates hidden representations of a list of positive and negative prompts by extracting them from the last layer of the LLM. They are averaged and normalized to initialize the gates. More information about it is available on [Charles Goddard's blog](https://goddard.blog/posts/clown-moe/).

As you can guess, the "hidden" initialization is the most efficient to correctly route the tokens to the most relevant experts. In the next section, we will create our own frankenMoE using this technique.

## 💻 Creating a frankenMoE

To create our frankenMoE, we need to select `n` experts. In this case, we will rely on Mistral-7B thanks to its popularity and relatively small size. However, eight experts like in Mixtral is quite a lot, as we need to fit all of them in memory. For efficiency, I'll only use four experts in this example, with two of them engaged for each token and each layer. In this case, we will end up with a model with 24.2B parameters instead of 4*7 = 28B parameters.

Here, our goal is to create a well-rounded model that can do pretty much everything: write stories, explain articles, code in Python, etc. We can decompose this requirement into four tasks and select the best expert for each of them. This is how I decomposed it:

* **Chat model**: A general-purpose model that is used in most interactions. I used [mlabonne/AlphaMonarch-7B](https://huggingface.co/mlabonne/AlphaMonarch-7B), which perfectly satisfies the requirements.
* **Code model**: A model capable of generating good code. I don't have a lot of experience with Mistral-7B-based code models, but I found [beowolx/CodeNinja-1.0-OpenChat-7B](https://huggingface.co/beowolx/CodeNinja-1.0-OpenChat-7B) particularly good compared to others.
* **Math model**: Math is tricky for LLMs, which is why we want a model specialized in math. Thanks to its high MMLU and GMS8K scores, I chose [mlabonne/NeuralDaredevil-7B](https://huggingface.co/mlabonne/NeuralDaredevil-7B) for this purpose.
* **Role-play model**: The goal of this model is to write high-quality stories and conversations. I selected [SanjiWatsuki/Kunoichi-DPO-v2-7B](https://huggingface.co/SanjiWatsuki/Kunoichi-DPO-v2-7B) because of its good reputation and high MT-Bench score (8.51 vs. 8.30 for Mixtral).

Now that we've identified the experts we want to use, we can create the YAML configuration that MergeKit will use to create our frankenMoE. This uses the mixtral branch of MergeKit. You can find more information about how to write the configuration [on this page](https://github.com/arcee-ai/mergekit/blob/mixtral/docs/moe.md). Here is our version:

```yaml
base_model: mlabonne/AlphaMonarch-7B
experts:
  - source_model: mlabonne/AlphaMonarch-7B
    positive_prompts:
    - "chat"
    - "assistant"
    - "tell me"
    - "explain"
    - "I want"
  - source_model: beowolx/CodeNinja-1.0-OpenChat-7B
    positive_prompts:
    - "code"
    - "python"
    - "javascript"
    - "programming"
    - "algorithm"
  - source_model: SanjiWatsuki/Kunoichi-DPO-v2-7B
    positive_prompts:
    - "storywriting"
    - "write"
    - "scene"
    - "story"
    - "character"
  - source_model: mlabonne/NeuralDaredevil-7B
    positive_prompts:
    - "reason"
    - "math"
    - "mathematics"
    - "solve"
    - "count"
```

For each expert, I provide five basic positive prompts. You can be a bit fancier and write entire sentences if you want. The best strategy consists of using real prompts that should trigger a particular expert. You can also add negative prompts to do the opposite.

Once this is ready, you can save your configuration as `config.yaml`. In the same folder, we will download and install the [mergekit](https://github.com/arcee-ai/mergekit) library (mixtral branch).

```bash
git clone -b mixtral https://github.com/arcee-ai/mergekit.git
cd mergekit && pip install -e .
pip install -U transformers
```

If your computer has enough RAM (roughly 24-32 GB of RAM), you can run the following command:

```bash
mergekit-moe config.yaml merge --copy-tokenizer
```

If you don't have enough RAM, you can shard the models instead as follows (it will take longer):

```bash
mergekit-moe config.yaml merge --copy-tokenizer --allow-crimes --out-shard-size 1B --lazy-unpickle
```

This command automatically downloads the experts and creates the frankenMoE in the `merge` directory.

Alternatively, you can copy your configuration into [LazyMergekit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb#scrollTo=d5mYzDo1q96y), a wrapper I made to simplify model merging. In this Colab notebook, you can input your model name, select the `mixtral` branch, specify your Hugging Face username/token, and run the cells. After creating your frankenMoE, it will also upload it to the Hugging Face Hub with a nicely formatted model card.

I called my model [Beyonder-4x7B-v3](https://huggingface.co/mlabonne/Beyonder-4x7B-v3) and created [GGUF versions](https://huggingface.co/mlabonne/Beyonder-4x7B-v3-GGUF) of it using [AutoGGUF](https://colab.research.google.com/drive/1P646NEg33BZy4BfLDNpTz0V0lwIU3CHu#scrollTo=fD24jJxq7t3k). If you can't run GGUF versions on your local machine, you can also perform inference using this [Colab notebook](https://colab.research.google.com/drive/1SIfwhpLttmoZxT604LGVXDOI9UKZ_1Aq?usp=sharing).

To get a good overview of its capabilities, it has been evaluated on three different benchmarks: Nous' benchmark suite, EQ-Bench, and the Open LLM Leaderboard. This model is not designed to excel in traditional benchmarks, as the code and role-playing models generally do not apply to those contexts. Nonetheless, it performs remarkably well thanks to strong general-purpose experts.

**Nous**: Beyonder-4x7B-v3 is one of the best models on Nous' benchmark suite (evaluation performed using [LLM AutoEval](https://github.com/mlabonne/llm-autoeval)) and significantly outperforms the v2. See the entire leaderboard [here](https://huggingface.co/spaces/mlabonne/Yet_Another_LLM_Leaderboard).

![](https://i.imgur.com/NO7Wc5l.png)

**EQ-Bench**: It's also the best 4x7B model on the [EQ-Bench leaderboard](https://eqbench.com/), outperforming older versions of ChatGPT and Llama-2-70b-chat. Beyonder is very close to Mixtral-8x7B-Instruct-v0.1 and Gemini Pro, which are (supposedly) much bigger models.

![](https://i.imgur.com/NQNBKOd.png)

**Open LLM Leaderboard**: Finally, it's also a strong performer on the Open LLM Leaderboard, significantly outperforming the v2 model.

![](https://i.imgur.com/w5UepM5.png)

On top of these quantitative evaluations, I recommend checking the model's outputs in a more qualitative way using a GGUF version on [LM Studio](https://lmstudio.ai/). A common way of testing these models is to gather a private set of questions and check their outputs. With this strategy, I found that Beyonder-4x7B-v3 is quite robust to changes in the user and system prompts compared to other models, including AlphaMonarch-7B. This is pretty cool as it improves the usefulness of the model in general.

FrankenMoEs are a promising but still experimental approach. The trade-offs, like higher VRAM demand and slower inference speeds, can make it challenging to see their advantage over simpler merging techniques like SLERP or DARE TIES. Especially, when you use frankenMoEs with just two experts, they might not perform as well as if you had simply merged the two models. However, frankenMoEs excel in preserving knowledge, which can result in stronger models, as demonstrated by Beyonder-4x7B-v3. With the right hardware, these drawbacks can be effectively mitigated.

## Conclusion

In this article, we introduced the Mixture of Experts architecture. Unlike traditional MoEs that are trained from scratch, MergeKit facilitates the creation of MoEs by ensembling experts, offering an innovative approach to improving model performance and efficiency. We detailed the process of creating a frankenMoE with MergeKit, highlighting the practical steps involved in selecting and combining different experts to produce a high-quality MoE.

Thanks for reading this article. I encourage you to try to make your own FrankenMoEs using [LazyMergeKit](https://colab.research.google.com/drive/1obulZ1ROXHjYLn6PPZJwRR6GzgQogxxb#scrollTo=d5mYzDo1q96y): select a few models, create your config based Beyonder's, and run the notebook to create your own models! If you liked this article, please follow me on [Hugging Face](https://huggingface.co/mlabonne) and X/Twitter [@maximelabonne](https://twitter.com/maximelabonne).

## References

* [Mixtral of Experts](https://arxiv.org/abs/2401.04088) by Jiang et al. (2023)
* [Mixture of Experts for Clowns](https://goddard.blog/posts/clown-moe/) by Charles Goddard (2023)
* [Mixture of Experts Explained](https://huggingface.co/blog/moe) by Sanseviero et al. (2023)
* [Adaptive Mixture of Local Experts](https://www.cs.toronto.edu/~hinton/absps/jjnh91.pdf) by Jacobs et al. (1991)
* [Sparse Upcycling: Training Mixture-of-Experts from Dense Checkpoints](https://arxiv.org/abs/2212.05055) by Komatsuzaki et al. (2022)