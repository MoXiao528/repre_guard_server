# RepreGuard

AIDetector 的文本检测服务，基于 FastAPI 提供 AI/Human 检测和可选的 Evidence 路由。

```text
AIDetector-web → AIDetector-Back → RepreGuard
```

本服务负责模型推理；用户、配额和检测历史由后端管理。

## 模型

主检测使用 [DetectRL-X XLM-RoBERTa Detector](https://huggingface.co/WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All)，模型版本由 `config.py` 固定。

`score` 是模型对 AI 类别 `LABEL_1` 的概率输出。默认阈值为 `0.0028`：达到阈值返回 `AI`，否则返回 `HUMAN`。

支持 CUDA 和 CPU，默认优先使用 CUDA。

## 快速开始

需要 Python 3.10 或以上版本。以下命令均在仓库根目录执行。

### 1. 安装依赖

```bash
python -m pip install -r requirements.txt
cp .env.example .env
```

### 2. 配置环境

编辑 `.env`：

```dotenv
REPRE_GUARD_SERVICE_TOKEN=<与后端一致的服务令牌>
REPRE_GUARD_MODEL_PATH=./models/detector
REPRE_GUARD_MODEL_CACHE_DIR=./models/cache
```

服务令牌至少包含 32 个无空白的可打印 ASCII 字符。`.env` 和模型文件不要提交到仓库。

### 3. 下载模型

以下命令下载配置中锁定的模型版本及必需文件：

```bash
python -c "from huggingface_hub import snapshot_download; from config import REQUIRED_MODEL_FILES, settings; snapshot_download(repo_id=settings.model_name, revision=settings.model_revision, local_dir=settings.model_path, allow_patterns=list(REQUIRED_MODEL_FILES))"
```

### 4. 启动服务

```bash
python run_roberta_server.py
```

默认监听 `0.0.0.0:9000`，启动时自动加载并预热模型。服务采用单 worker，访问范围应由网络或防火墙限制。

## 接口

下列三个服务接口都需要请求头：

```text
X-RepreGuard-Token: <服务令牌>
```

| 方法 | 路径 | 用途 |
| --- | --- | --- |
| GET | `/health` | 健康检查 |
| POST | `/detect` | AI/Human 检测 |
| POST | `/evidence/route` | Evidence 语言、领域路由 |

### 文本检测

请求：

```json
{
  "text": "需要检测的文本。"
}
```

响应示例：

```json
{
  "score": 0.037,
  "threshold": 0.0028,
  "label": "AI",
  "model_name": "WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All",
  "score_type": "probability"
}
```

客户端应检查 `score_type`，不能将其他类型的分数当作概率使用。

输入限制：请求体最多 128 KiB；文本为 1～20,000 个 Unicode 码点且不能全为空白；主检测最多 512 tokens，包含特殊 tokens。token 超限返回 `422 INPUT_TOO_LONG`，不会静默截断。

默认同时执行一个主检测请求，最多三个请求等待，等待上限为 15 秒。队列满或等待超时返回 `503`。

## Evidence 路由

Evidence 默认关闭，通过 `.env` 中的 `REPRE_GUARD_EVIDENCE_*` 配置启用，需要单独提供 Router 模型目录和产物 SHA-256。

Router 在 CPU 上判断整篇文本的语言和领域，供后端选择写作特点参考统计。它不生成 AI 分数，不修改主检测结果。

该接口的业务结果使用 HTTP `200`，调用方必须检查 `status` 和 `reason`；未启用或加载失败时返回 `failed/model_unavailable`。鉴权和输入错误使用非 2xx 状态。

## 测试

```bash
python -B -m unittest discover -s tests -v
```

普通测试使用模拟或小型模型，不下载生产模型；部分依赖缺失时会跳过对应检查。

详细配置、Router 协议和验收命令见 [运行说明](docs/operations.md)。历史真实模型验收及运行限制见 [验收记录](docs/archive/evidence-runtime-acceptance-2026-09-07.md)。本地验收与生产部署分别记录。
