---
aliases:
- /orpo/
categories:
- Large Language Models
colab: <a href="https://colab.research.google.com/drive/1eHNWg9gnaXErdAa8_mcvjMupbSS6rDvi?usp=sharing"><img src="images/colab.png" alt="Open In Colab"></a>
date: '2024-04-19'
image: /images/orpo/thumbnail.jpg
title: "Fine-tune Llama 3 with ORPO"
subtitle: "A cheaper and faster unified fine-tuning technique"
---

![](https://i.imgur.com/ZHwzQvI.png)

ORPO is a **new exciting fine-tuning technique** that combines the traditional supervised fine-tuning and preference alignment stages into a single process. This reduces the computational resources and time required for training. Moreover, empirical results demonstrate that ORPO outperforms other alignment methods on various model sizes and benchmarks.

In this article, we will fine-tune the new Llama 3 8B model using ORPO with the TRL library. The code is available on [Google Colab](https://colab.research.google.com/drive/1eHNWg9gnaXErdAa8_mcvjMupbSS6rDvi?usp=sharing) and in the [LLM Course](https://github.com/mlabonne/llm-course) on GitHub.

## ⚖️ ORPO

Instruction tuning and preference alignment are essential techniques for adapting Large Language Models (LLMs) to specific tasks. Traditionally, this involves a multi-stage process: 1/ **Supervised Fine-Tuning** (SFT) on instructions to adapt the model to the target domain, followed by 2/ **preference alignment methods** like Reinforcement Learning with Human Feedback (RLHF) or Direct Preference Optimization (DPO) to increase the likelihood of generating preferred responses over rejected ones.

![](https://i.imgur.com/ftrth4Q.png)

However, researchers have identified a limitation in this approach. While SFT effectively adapts the model to the desired domain, it inadvertently **increases the probability of generating undesirable answers** alongside preferred ones. This is why the preference alignment stage is necessary to widen the gap between the likelihoods of preferred and rejected outputs.

![](https://i.imgur.com/zWnTNlH.png)
<center><i>Note how the probability of rejected responses increases during supervised fine-tuning (from the ORPO paper).</i></center>

Introduced by [Hong and Lee (2024)](https://arxiv.org/abs/2403.07691), ORPO offers an elegant solution to this problem by combining instruction tuning and preference alignment into a single, monolithic training process. ORPO modifies the standard language modeling objective, combining the negative log-likelihood loss with an odds ratio (OR) term. This OR loss weakly penalizes rejected responses while strongly rewarding preferred ones, allowing the model to simultaneously learn the target task and align with human preferences.

$$\mathscr{L}_{ORPO} = \mathbb{E}_{(x, y_{w}, y_l)}[\mathscr{L}_{SFT} + \lambda \cdot \mathscr{L}_{OR}]$$
ORPO has been implemented in the major fine-tuning libraries, like [TRL](https://github.com/huggingface/trl), [Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl), and [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory). In the next section, we will see how to use with TRL.

## 💻 Fine-tuning Llama 3 with ORPO

[Llama 3](https://github.com/meta-llama/llama3/tree/main) is the latest family of LLMs developed by Meta. The models were trained on an extensive dataset of **15 trillion tokens** (compared to 2T tokens for Llama 2). Two model sizes have been released: a 70 billion parameter model and a smaller 8 billion parameter model. The 70B model has already demonstrated impressive performance, scoring 82 on the MMLU benchmark and 81.7 on the HumanEval benchmark.

Llama 3 models also increased the context length up to 8,192 tokens (4,096 tokens for Llama 2), and potentially scale up to 32k with RoPE. Additionally, the models use a new tokenizer with a 128K-token vocabulary, reducing the number of tokens required to encode text by 15%. This vocabulary also explains the bump from 7B to 8B parameters.

![](https://i.imgur.com/IFeK7DO.png)
<center><i>Samples from ORPO-DPO-mix-40k.</i></center>

ORPO requires a preference dataset, including a prompt, a chosen answer, and a rejected answer. In this example, we will use [`mlabonne/orpo-dpo-mix-40k`](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k), a combination of the following high-quality DPO datasets:

- [`argilla/distilabel-capybara-dpo-7k-binarized`](https://huggingface.co/datasets/argilla/distilabel-capybara-dpo-7k-binarized): highly scored chosen answers >=5 (2,882 samples)
- [`argilla/distilabel-intel-orca-dpo-pairs`](https://huggingface.co/datasets/argilla/distilabel-intel-orca-dpo-pairs): highly scored chosen answers >=9, not in GSM8K (2,299 samples)
- [`argilla/ultrafeedback-binarized-preferences-cleaned`](https://huggingface.co/datasets/argilla/ultrafeedback-binarized-preferences-cleaned): highly scored chosen answers >=5 (22,799 samples)
- [`argilla/distilabel-math-preference-dpo`](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo): highly scored chosen answers >=9 (2,181 samples)
- [`unalignment/toxic-dpo-v0.2`](https://huggingface.co/datasets/unalignment/toxic-dpo-v0.2) (541 samples)
- [`M4-ai/prm_dpo_pairs_cleaned`](https://huggingface.co/datasets/M4-ai/prm_dpo_pairs_cleaned) (7,958 samples)
- [`jondurbin/truthy-dpo-v0.1`](https://huggingface.co/datasets/jondurbin/truthy-dpo-v0.1) (1,016 samples)

Thanks to [argilla](https://huggingface.co/argilla), [unalignment](https://huggingface.co/unalignment), [M4-ai](https://huggingface.co/M4-ai), and [jondurbin](https://huggingface.co/jondurbin) for providing the source datasets.

As per usual, let's start by installing the required libraries:

```bash
pip install -U transformers datasets accelerate peft trl bitsandbytes wandb
```

Once it's installed, we can import the necessary libraries and log in to W&B (optional):

```python	
import gc
import os

import torch
import wandb
from datasets import load_dataset
from google.colab import userdata
from peft import LoraConfig, PeftModel, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    pipeline,
)
from trl import ORPOConfig, ORPOTrainer, setup_chat_format

wb_token = userdata.get('wandb')
wandb.login(key=wb_token)
```

If you have a recent GPU, you should also be able to use the [Flash Attention library](https://github.com/Dao-AILab/flash-attention) to replace the default eager attention implementation with a more efficient one.

```python	
if torch.cuda.get_device_capability()[0] >= 8:
    !pip install -qqq flash-attn
    attn_implementation = "flash_attention_2"
    torch_dtype = torch.bfloat16
else:
    attn_implementation = "eager"
    torch_dtype = torch.float16
```

In the following, we will load the Llama 3 8B model in 4-bit precision thanks to [bitsandbytes](https://github.com/TimDettmers/bitsandbytes). We then set the LoRA configuration using [PEFT](https://github.com/huggingface/peft) for QLoRA. I'm also using the convenient `setup_chat_format()` function to modify the model and tokenizer for [ChatML](https://huggingface.co/docs/transformers/en/chat_templating#what-template-should-i-use) support. It automatically applies this chat template, adds special tokens, and resizes the model's embedding layer to match the new vocabulary size.

Note that you need to submit a request to access [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) and be logged in to your Hugging Face account. Alternatively, you can load ungated copies of the model, like [NousResearch/Meta--Llama-3-8B](https://huggingface.co/NousResearch/Meta-Llama-3-8B).

```python
# Model
base_model = "meta-llama/Meta-Llama-3-8B"
new_model = "OrpoLlama-3-8B"

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch_dtype,
    bnb_4bit_use_double_quant=True,
)

# LoRA config
peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=['up_proj', 'down_proj', 'gate_proj', 'k_proj', 'q_proj', 'v_proj', 'o_proj']
)

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(base_model)

# Load model
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    quantization_config=bnb_config,
    device_map="auto",
    attn_implementation=attn_implementation
)
model, tokenizer = setup_chat_format(model, tokenizer)
model = prepare_model_for_kbit_training(model)
```

Now that the model is ready for training, we can take care of the dataset. We load [`mlabonne/orpo-dpo-mix-40k`](https://huggingface.co/datasets/mlabonne/orpo-dpo-mix-40k) and use the `apply_chat_template()` function to convert the "chosen" and "rejected" columns into the ChatML format. Note that I'm only using 1,000 samples and not the entire dataset, as it would take too long to run.

```python	
dataset_name = "mlabonne/orpo-dpo-mix-40k"
dataset = load_dataset(dataset_name, split="all")
dataset = dataset.shuffle(seed=42).select(range(100))

def format_chat_template(row):
    row["chosen"] = tokenizer.apply_chat_template(row["chosen"], tokenize=False)
    row["rejected"] = tokenizer.apply_chat_template(row["rejected"], tokenize=False)
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc= os.cpu_count(),
)
dataset = dataset.train_test_split(test_size=0.01)
```

First, we need to set a few hyperparameters:
* `learning_rate`: ORPO uses very low learning rates compared to traditional SFT or even DPO. This value of 8e-6 comes from the original paper, and roughly corresponds to an SFT learning rate of 1e-5 and a DPO learning rate of 5e-6. I would recommend increasing it around 1e-6 for a real fine-tune.
* `beta`: It is the $\lambda$ parameter in the paper, with a default value of 0.1. An appendix from the original paper shows how it's been selected with an ablation study.
* Other parameters, like `max_length` and batch size are set to use as much VRAM as available (~20 GB in this configuration). Ideally, we would train the model for 3-5 epochs, but we'll stick to 1 here.

Finally, we can train the model using the ORPOTrainer, which acts as a wrapper.

```python
orpo_args = ORPOConfig(
    learning_rate=8e-6,
    beta=0.1,
    lr_scheduler_type="linear",
    max_length=1024,
    max_prompt_length=512,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=4,
    optim="paged_adamw_8bit",
    num_train_epochs=1,
    evaluation_strategy="steps",
    eval_steps=0.2,
    logging_steps=1,
    warmup_steps=10,
    report_to="wandb",
    output_dir="./results/",
)

trainer = ORPOTrainer(
    model=model,
    args=orpo_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    peft_config=peft_config,
    tokenizer=tokenizer,
)
trainer.train()
trainer.save_model(new_model)
```

Training the model on these 1,000 samples took about 2 hours on an L4 GPU. Let's check the W&B plots:

![](https://i.imgur.com/r78hGrl.png)

While the loss goes down, the difference between the chosen and rejects answers is not clear: the average margin and accuracy are only slightly above zero and 0.5, respectively. 

In the original paper, the authors trained models on the [`Anthropic/hh-rlhf`](https://huggingface.co/datasets/Anthropic/hh-rlhf) dataset (161k samples) for 10 epochs, which is a lot longer than our quick run. They also experimented with Llama 3 and kindly [shared their logs](https://huggingface.co/orpo-explorers/hf-llama3-8b-orpo-v0.0/tensorboard) with me (thanks [Jiwoo Hong](https://twitter.com/jiwoohong98)).

To end this tutorial, let's merge the QLoRA adapter with the base model and push it to the Hugging Face Hub.

```python
# Flush memory
del trainer, model
gc.collect()
torch.cuda.empty_cache()

# Reload tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(base_model)
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    low_cpu_mem_usage=True,
    return_dict=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
model, tokenizer = setup_chat_format(model, tokenizer)

# Merge adapter with base model
model = PeftModel.from_pretrained(model, new_model)
model = model.merge_and_unload()

model.push_to_hub(new_model, use_temp_dir=False)
tokenizer.push_to_hub(new_model, use_temp_dir=False)
```

Congrats, we finished this quick fine-tune of Llama 3: [mlabonne/OrpoLlama-3-8B](https://huggingface.co/mlabonne/OrpoLlama-3-8B). You can play with it using this [Hugging Face Space](https://huggingface.co/spaces/mlabonne/OrpoLlama-3-8B) (here's a [notebook](https://colab.research.google.com/drive/1LcVUW5wsJTO2NGmozjji5CkC--646LgC?usp=sharing) to make your own). Although the model is undertrained, as highlighted by the W&B curves, I ran some evaluations on Nous' benchmark suite using [LLM AutoEval](https://github.com/mlabonne/llm-autoeval).

| Model                                                                                                                                                                     |   Average |   AGIEval |   GPT4All | TruthfulQA |  Bigbench |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------: | --------: | --------: | ---------: | --------: |
| [teknium/OpenHermes-2.5-Mistral-7B](https://huggingface.co/teknium/OpenHermes-2.5-Mistral-7B) [📄](https://gist.github.com/mlabonne/88b21dd9698ffed75d6163ebdc2f6cc8)     |     52.42 |     42.75 |     72.99 |      52.99 |     40.94 |
| [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct) [📄](https://gist.github.com/mlabonne/8329284d86035e6019edb11eb0933628) |     51.34 |     41.22 |     69.86 |      51.65 |     42.64 |
| [mistralai/Mistral-7B-Instruct-v0.1](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) [📄](https://gist.github.com/mlabonne/7a0446c3d30dfce72834ef780491c4b2)   |     49.15 |     33.36 |     67.87 |      55.89 |     39.48 |
| [**mlabonne/OrpoLlama-3-8B**](https://huggingface.co/mlabonne/OrpoLlama-3-8B) [📄](https://gist.github.com/mlabonne/f41dad371d1781d0434a4672fd6f0b82)                     | **46.76** | **31.56** | **70.19** |  **48.11** | **37.17** |
| [meta-llama/Meta-Llama-3-8B](https://huggingface.co/meta-llama/Meta-Llama-3-8B) [📄](https://gist.github.com/mlabonne/616b6245137a9cfc4ea80e4c6e55d847)                   |     45.42 |      31.1 |     69.95 |      43.91 |      36.7 |

Our ORPO fine-tune is actually pretty decent and improves the base model's performance on every benchmark. This is encouraging and likely means that a fine-tune on the entire 40k samples would yield great results.

This is an exciting time for the open-source community, with more and more high-quality open-weight models being released. The gap between closed-source and open-weight models is slowly closing, and fine-tuning is an essential tool to get the best performance for your use cases.

![](https://i.imgur.com/id852fz.png)

## Conclusion

In this article, we introduced the ORPO algorithm and explained how it unifies the SFT and preference alignment stages into a single process. Then, we used TRL to fine-tune a Llama 3 8B model on a custom preference dataset. The final model shows encouraging results and highlights ORPO's potential as a new fine-tuning paradigm.

I hope it was useful, and I recommend running the [Colab notebook](https://colab.research.google.com/drive/1eHNWg9gnaXErdAa8_mcvjMupbSS6rDvi?usp=sharing) to fine-tune your own Llama 3 models. In future articles, we will see how to create high-quality datasets — a point that is often overlooked. If you liked this article, please follow me on [Hugging Face](https://huggingface.co/mlabonne/) and Twitter [@maximelabonne](https://twitter.com/maximelabonne).

## References

* J. Hong, N. Lee, and J. Thorne, [ORPO: Monolithic Preference Optimization without Reference Model](https://arxiv.org/abs/2403.07691). 2024.
* L. von Werra et al., TRL: Transformer Reinforcement Learning. GitHub, 2020. [Online]. Available: https://github.com/huggingface/trl
* Bartolome, A., Martin, G., & Vila, D. (2023). Notus. In GitHub Repository. GitHub. https://github.com/argilla-io/notus
* AI at Meta, [Introducing Meta Llama 3](https://ai.meta.com/blog/meta-llama-3/), 2024.