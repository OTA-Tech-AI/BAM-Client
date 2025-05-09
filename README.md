<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <img width="1280" alt="beholder-tool-kit-long" src="https://github.com/user-attachments/assets/6179e165-259a-432a-bf86-fb1b57eebcb4" />
</div>
<br>
<div align="center" style="line-height: 1;">
  <a href="https://www.otatech.ai/ota-agent"><img alt="Homepage"
    src="https://img.shields.io/badge/Home-Page-blue"/></a>
  <a href="https://huggingface.co/OTA-AI/OTA-v1"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-OTA%20AI-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="https://github.com/OTA-Tech-AI/BAM-Client/blob/main/LICENSE"><img alt="Code License"
    src="https://img.shields.io/badge/Code_License-MIT-f5de53?&color=f5deff"/></a>
  <br><br><br>
</div>

# BAM-Client

This repository provides an out-of-the-box tool for utilizing [OTA's Browser Agent Model (BAM)](https://github.com/OTA-Tech-AI/OTA-v1), with enhanced features built on top of existing frameworks. It is intended as a supplementary repo to the model, enabling seamless interaction with web environments through a browser-based agent system.

This repo is **forked from**:

- [Browser-Use](https://github.com/browser-use/browser-use) (last commit Mar 16 2025) – provides the core browser action framework.
- [WebVoyager](https://github.com/MinorJerry/WebVoyager) – contributes the concurrency design and result-saving mechanism.

## 🧠 About OTA's BAM

The [Browser Action Model (BAM)](https://github.com/OTA-Tech-AI/OTA-v1) is a lightweight, non-generative model designed by OTA Technology Inc. for intelligent browser-based automation. This repository makes it easy to plug BAM into a fully functional browser action loop with minimal setup.

## 💻 Demo

[![Video Thumbnail](https://img.youtube.com/vi/cVLcILFg-Gc/0.jpg)](https://youtu.be/cVLcILFg-Gc)

## 📦 Quick Start

### Set up conda environment

```
conda create -n BAM python=3.12
conda activate BAM
```

### Hardware Requirement

Make sure your Dedicated GPU Memory > 20 GB 

Please refer to the [OTA-v1](https://github.com/OTA-Tech-AI/OTA-v1) for detailed model info.

### Run the follwing command to pull OTA-v1 from Hugging Face:

```
ollama pull hf.co/OTA-AI/OTA-v1
```

### env setup
Setup your virtual environment using pip:

```
pip install -r requirements.txt
```

### Prepare Tasks

To create your own tasks, follow the format used in the test files under the `testcases/` directory. For example, a task in `OTA_testdataset_mini.jsonl` looks like this:

```json
{"web_name": "Allrecipes", "id": "Allrecipes--4", "ques": "Find a recipe for Baked Salmon that takes less than 30 minutes to prepare and has at least a 4 star rating based on user reviews.", "web": "https://www.allrecipes.com/"}
```

**web_name**: the website name you want to visit in this task

**id**: a unique ID for the task

**ques**: what you want browser-use to do

**web**: link to the website

please refer to [WebVoyager](https://github.com/MinorJerry/WebVoyager) for more information.

### Execute your tasks
Run the following command to start the task:

```
python run_tasks.py --model-provider ollama --max-concurrent 1 --task_jsonl_path testcases/OTA_testdataset_mini.jsonl
```


## 🔧 Improvements Over the Original `browser-use` Framework

We have extended and improved the `browser-use` framework with the following key features:

1. **Similarity-Based Element Selection**  
   We integrate **similarity search** into the web page content symbol space to **select only the top-K relevant interactive elements**. These are chosen based on their relevance to the agent's next sub-goal, improving both efficiency and model performance.

2. **Action History Limiting**  
   To manage token usage and avoid overwhelming the LLM, we limit the number of previous actions included in the prompt. This helps maintain a concise and effective context for decision-making.


## 📝 License

This project inherits licensing terms from its upstream forks. Refer to each respective repository for license details.

---

Maintained by [OTA Technologies Inc.](https://www.otatech.ai/)
