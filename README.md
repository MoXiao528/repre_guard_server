# RepreGuard

RepreGuard 是 AIDetector 的内部文本推理服务。主检测使用 `DetectRL-X-XLM-RoBERTa-Detector-All`，返回 AI 类别概率 `score`、判定阈值 `threshold` 和 `AI` / `HUMAN` 标签。默认阈值为 `0.0028`，分数达到阈值即返回 `AI`。模型版本锁定在 [`config.py`](config.py)，模型文件见 [Hugging Face](https://huggingface.co/WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All)。

可选的 Evidence Router 判断整篇文本的语言和领域，供后端选择写作特点的参考统计。用户认证、配额和检测历史由后端管理。关联仓库见 [AIDetector-web](https://github.com/MoXiao528/AIDetector-web) 和 [AIDetector-Back](https://github.com/MoXiao528/AIDetector-Back)。

## 快速开始

以下使用 Python 3.11。首次下载模型需要访问 Hugging Face，也可提前准备同版本的本地模型目录。获取代码后，以下命令均在仓库根目录执行。

```bash
git clone https://github.com/MoXiao528/repre_guard_server.git
cd repre_guard_server
```

### 1. 创建环境并安装依赖

Linux（Bash）

```bash
python3.11 -m venv .venv
source .venv/bin/activate
test -f .env || cp .env.example .env
```

Windows（PowerShell）

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
if (-not (Test-Path -LiteralPath .env)) { Copy-Item .env.example .env }
```

激活环境后安装依赖。

```bash
python -m pip install -r requirements.txt
```

### 2. 配置令牌和模型路径

编辑从 [`.env.example`](.env.example) 复制的 `.env`。将令牌占位值替换为实际值。接入后端时，两端的 `REPRE_GUARD_SERVICE_TOKEN` 必须完全一致。

```dotenv
REPRE_GUARD_SERVICE_TOKEN=REPLACE_ME
REPRE_GUARD_MODEL_PATH=./models/detector
REPRE_GUARD_MODEL_CACHE_DIR=./models/cache
REPRE_GUARD_HOST=127.0.0.1
REPRE_GUARD_PORT=9000
```

令牌至少 32 个无空白的可打印 ASCII 字符。首次部署可用 `python -c "import secrets; print(secrets.token_urlsafe(32))"` 生成令牌，保存到部署配置。`.venv`、真实令牌和本地模型留在本机。

### 3. 下载模型并启动

以下命令按当前配置下载指定版本的必需文件。已经预置模型的环境可直接启动。

```bash
python -c '
from config import REQUIRED_MODEL_FILES, settings
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id=settings.model_name,
    revision=settings.model_revision,
    local_dir=settings.model_path,
    allow_patterns=list(REQUIRED_MODEL_FILES),
)
'
```

```bash
python run_roberta_server.py
```

主模型加载并预热成功后，服务开始接收请求。按上面的配置，地址为 `http://127.0.0.1:9000`。

### 4. 发送第一条请求

打开另一个终端，进入仓库根目录并激活同一虚拟环境。下面的命令从当前配置读取令牌并发送健康检查和文本检测请求。

Linux（Bash）

```bash
token="$(python -c 'from config import settings; print(settings.require_service_token())')"
curl -fsS -H "X-RepreGuard-Token: $token" http://127.0.0.1:9000/health
curl -fsS -H "X-RepreGuard-Token: $token" -H 'Content-Type: application/json' \
  --data '{"text":"The museum opens at nine on weekdays."}' http://127.0.0.1:9000/detect
```

Windows（PowerShell）

```powershell
$headers = @{ 'X-RepreGuard-Token' = (python -c 'from config import settings; print(settings.require_service_token())') }
Invoke-RestMethod -Headers $headers -Uri 'http://127.0.0.1:9000/health'
Invoke-RestMethod -Method Post -Headers $headers -Uri 'http://127.0.0.1:9000/detect' -ContentType 'application/json' -Body '{"text":"The museum opens at nine on weekdays."}'
```

健康检查返回 `{"status":"ok"}`。主检测返回以下结构，`score` 和 `label` 是示例值，以实际推理结果为准。

```json
{
  "score": 0.037,
  "threshold": 0.0028,
  "label": "AI",
  "model_name": "WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All",
  "score_type": "probability"
}
```

## 接口

以下接口均要求 `X-RepreGuard-Token`。POST 请求使用 `Content-Type: application/json`。

| 方法与路径 | 请求 | 结果 |
| --- | --- | --- |
| `GET /health` | 无 | `{"status":"ok"}` |
| `POST /detect` | `{"text":"待测文本"}`，也可显式加 `"action":"detect"` | AI 类别概率、阈值和标签 |
| `POST /detect` | `{"text":"完整原文","action":"evidence"}` | Evidence 语言与领域路由结果 |
| `POST /evidence/route` | `{"text":"完整原文"}` | 兼容已有直连客户端的 Evidence 路由入口 |

Evidence 请求需先启用并加载 Router，默认关闭。

POST 请求体最多 128 KiB（含 JSON 开销），`text` 为 1～20,000 个 Unicode 字符且不能全为空白。主检测默认最多 512 tokens（含特殊 tokens），超限返回 `422 INPUT_TOO_LONG`。

## 部署

- 使用 `python run_roberta_server.py` 启动单 worker。代码默认绑定 `0.0.0.0:9000`。仅本机访问时可绑定 `127.0.0.1`。后端容器或其他主机访问时，用 `REPRE_GUARD_HOST` 和 `REPRE_GUARD_PORT` 选择其可达的地址和端口，并限制检测端口的访问范围。
- Linux 上配置实际模型与缓存路径，建议用绝对路径。进程环境变量优先于 `.env`，相对路径以工作目录为准。离线迁移时保留下载生成的 `.cache/huggingface/download/*.metadata`，或提供版本清单。格式见 [主模型配置与校验](docs/operations.md#主检测配置与模型校验)。
- 默认优先 CUDA，不可用时回退 CPU。要求 GPU 的环境应在启动前用 `python -c "import torch; assert torch.cuda.is_available()"` 检查运行环境。默认同时执行一次主检测，允许三个等待请求。
- 直连时从后端实际网络验证 `/health` 和 `/detect`。PHP 固定转发到 `/detect` 时，还要从后端走实际 PHP 地址验证主检测响应，检查令牌、正文、响应字段和 HTTP 状态码。


## 测试

在已安装运行依赖的虚拟环境中执行。
```bash
python -m pip install httpx
python -B -m unittest discover -s tests -v
```

单元测试使用模拟或小型模型。
