# RepreGuard Server (API Edition)

**RepreGuard Server** 是 [RepreGuard](https://github.com/NLP2CT/RepreGuard) 的服务端实现。它提供了一个高性能的 RESTful API 接口，利用 LLM 的隐藏层表示模式（Hidden Representation Patterns）来检测文本是由 AI 生成的还是人类编写的。

本项目基于 FastAPI 构建，支持 Docker 部署，并提供了从模型初始化到服务启动的完整流程。

---

## 🚀 快速开始 / Quick Start

### 1. 环境准备 (Environment Setup)

建议使用 Conda 创建独立的虚拟环境。

```bash
# 1. 创建环境
conda create -n repre_guard_server python=3.10
conda activate repre_guard_server

# 2. 安装依赖
# 注意：请确保你的机器安装了适配 PyTorch 的 CUDA 版本（如果使用 GPU）
pip install -r requirements.txt

# 3. 安装 Uvicorn (用于启动 FastAPI 服务)
pip install uvicorn

```

### 2. 初始化检测模型 (Initialize Model)

在启动服务之前，必须先训练/拟合 RepReader（表示读取器）并生成权重文件 `saved_rep_reader.pt`。

本项目提供了一个初始化脚本 `init_tiny_model.py`。默认情况下，它使用轻量级的 `tiny-gpt2` 模型进行演示。

**运行初始化脚本：**

```bash
python init_tiny_model.py

```

> **📝 注意：** > * 脚本会读取 `train_MIXED_ALL.json` 数据。
> * 执行成功后，会在当前目录下生成 `saved_rep_reader.pt` 文件。
> * **更换模型**：如果你需要更强的检测能力（例如使用 Qwen2.5-7B），请修改 `init_tiny_model.py` 中的 `MODEL_NAME` 变量：
> ```python
> # init_tiny_model.py
> # MODEL_NAME = "sshleifer/tiny-gpt2" 
> MODEL_NAME = "Qwen/Qwen2.5-7B"  # <--- 修改这里
> 
> ```
> 
> 
> *修改模型后，请务必调整 `batch_size` 以防止显存溢出（OOM）。*
> 
> 

### 3. 启动 API 服务 (Start Server)

确认 `saved_rep_reader.pt` 生成后，即可启动 FastAPI 服务。

```bash
uvicorn server:app --host 0.0.0.0 --port 8000 --reload

```

服务启动后，你应该能看到类似以下的日志：

```text
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
...
INFO:     Startup complete.

```

---

## 🔌 API 使用指南 (API Usage)

服务默认运行在 `http://localhost:8000`。同时也提供了交互式文档：

* **Swagger UI:** `http://localhost:8000/docs`
* **ReDoc:** `http://localhost:8000/redoc`

### 检测文本 (Detect Text)

* **Endpoint:** `/detect`
* **Method:** `POST`
* **Content-Type:** `application/json`

#### 请求示例 (Request)

```bash
curl -X 'POST' \
  'http://localhost:8000/detect' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "text": "The rapid advancement of artificial intelligence has sparked significant debate regarding its potential impact on the workforce."
}'

```

#### 响应示例 (Response)

```json
{
  "score": 0.982,
  "threshold": 0.5,
  "label": "AI",
  "model_name": "sshleifer/tiny-gpt2"
}

```

* `score`: 检测得分（通常越高越倾向于 AI）。
* `threshold`: 判定阈值。
* `label`: 最终判定结果 (`AI` 或 `HUMAN`)。
* `model_name`: 后端使用的代理模型名称。

---

## 🛠️ 项目结构 (File Structure)

```text
.
├── server.py                 # FastAPI 服务入口文件
├── init_tiny_model.py        # 模型初始化与 RepReader 训练脚本
├── repreGuard_service.py     # 核心服务逻辑封装
├── repreGuard_detector.py    # 核心检测算法实现
├── metrics.py                # 评估指标计算
├── requirements.txt          # Python 依赖列表
├── saved_rep_reader.pt       # (生成文件) 训练好的 RepReader 权重
└── train_MIXED_ALL.json      # 训练/校准用的数据集

```

---

## ⚙️ 高级配置 (Configuration)

### 更换基础 LLM (Surrogate Model)

为了在生产环境中获得最佳性能（如论文所述），建议将 `tiny-gpt2` 替换为更强大的模型（如 Llama-3, Qwen-2.5 等）。

1. 打开 `init_tiny_model.py`。
2. 修改 `MODEL_NAME` 指向你的目标模型路径或 HuggingFace ID。
3. 调整 `truncate_data` 中的 `max_length`（大模型建议 2048）。
4. 调整 `batch_size`（大模型建议设为 1 或 2）。
5. **重新运行** `python init_tiny_model.py` 生成新的 `.pt` 文件。
6. **重启** `server.py`（服务会自动加载新的配置）。

---

## 📚 引用 (Citation)

本项目是 RepreGuard 论文的官方实现扩展。如果您觉得有用，请引用我们的论文：

```bibtex
@article{chen2025repreguard,
  author       = {Xin Chen, Junchao Wu, Shu Yang, Runzhe Zhan, Zeyu Wu, Ziyang Luo, Di Wang, Min Yang, Lidia S. Chao and Derek F. Wong},
  title        = {RepreGuard: Detecting LLM-Generated Text by Revealing Hidden Representation Patterns},
  journal      = {Transactions of the Association for Computational Linguistics},
  year         = {2025},
  url          = {https://arxiv.org/abs/2508.13152},
  note         = {Accepted at TACL 2025}
}

```