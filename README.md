# Modern Deep Learning Tutorials

## 📖 简介 (Introduction)
内容涵盖了从 PyTorch 基础、底层网络实现（MLP），到进阶模型架构（CNN, RNN, Transformers, GNN），以及当前热门的生成式 AI（VAE, DDPM）和大模型微调技术。

非常适合深度学习初学者以及想要系统性巩固工程实践能力的开发者参考。

## 🛠️ 环境依赖 (Prerequisites)
运行本仓库代码需要用到以下主要库：
* PyTorch
* HuggingFace `transformers` & `datasets`
* PyTorch Geometric (PyG)
* 以及其他常规数据科学库 (`numpy`, `matplotlib` 等)

## 📁 项目目录结构 (Project Structure)

```
Deep-Learning-Labs/
├── README.md
├── requirements.txt
├── 01_Basics_and_PyTorch/
│   └── pytorch_setup_linear_regression.ipynb
├── 02_MLP_and_Debugging/
│   ├── mlp_from_scratch.ipynb
│   └── debugging_training.md
├── 03_Advanced_Architectures/
│   ├── cnn_with_pytorch.ipynb
│   ├── rnn_lstm_text_classification.ipynb
│   ├── huggingface_transformer_finetuning.ipynb
│   ├── pyg_gnn_basics.ipynb
│   └── pruning_and_quantization.ipynb
├── 04_Generative_AI_and_LLMs/
│   ├── prompting_lora_finetuning.ipynb
│   ├── vae_latent_walk.ipynb
│   └── ddpm_sampling_demo.ipynb
└── 05_Projects_and_QA/
    └── qa_project_highlights.md
```

## 💡 如何使用 (How to use)
建议按照模块顺序从 01 到 04 依次学习。大部分代码以 Jupyter Notebook (`.ipynb`) 形式提供，可以直接在本地环境或 Google Colab 中打开运行。
