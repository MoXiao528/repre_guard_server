# RepreGuard 运行说明

首次安装与启动见 [README](../README.md)。本文保留按需查阅的配置、边界和验收方法；所有命令从仓库根目录执行，`python` 指向已准备好的环境。

## 主检测配置与模型校验

服务读取 `config.py` 同目录下的 `.env`，进程或容器环境变量优先。相对模型路径仍相对工作目录；启动前必须进入仓库根目录，或将路径改为部署环境的绝对路径。

```dotenv
REPRE_GUARD_MODEL_NAME=WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All
REPRE_GUARD_MODEL_REVISION=76649a0257a812a81cf36b5de9cc5f2430aeaa7f
REPRE_GUARD_MODEL_PATH=./models/detector
REPRE_GUARD_MODEL_CACHE_DIR=./models/cache
REPRE_GUARD_LOCAL_FILES_ONLY=true
REPRE_GUARD_THRESHOLD=0.0028
REPRE_GUARD_AI_LABEL_ID=1
REPRE_GUARD_TOKENIZER_USE_FAST=false
REPRE_GUARD_MAX_INPUT_TOKENS=512
REPRE_GUARD_MAX_PENDING_REQUESTS=3
REPRE_GUARD_QUEUE_TIMEOUT_SECONDS=15
REPRE_GUARD_DEVICE=auto
REPRE_GUARD_HOST=127.0.0.1
REPRE_GUARD_PORT=9000
REPRE_GUARD_LOG_LEVEL=info
REPRE_GUARD_SERVICE_TOKEN=REPLACE_WITH_SHARED_SERVICE_TOKEN
```

上面的目录和绑定地址是便携示例，需覆盖 `.env.example` 未设置的模型路径。代码默认绑定 `0.0.0.0`；需要其他容器或主机访问时按部署网络配置地址和访问边界。`DEVICE=auto` 优先 CUDA，不可用时回退 CPU；显式 `cuda` 不可用时也会回退。

本地模型目录必须包含以下五个文件：

```text
config.json
model.safetensors
sentencepiece.bpe.model
special_tokens_map.json
tokenizer_config.json
```

目录存在时，启动先检查文件是否齐全，再核对固定 revision。可保留 `snapshot_download` 写入的 `.cache/huggingface/download/*.metadata`；也可提供 `.repre_guard_model_manifest.json` 或 `model_manifest.json`，其中 `revision` 或 `model_revision` 必须等于配置值。不要丢弃版本元数据后仅复制权重。

目录不存在时，加载器改用配置的 Hugging Face 仓库、revision 和模型缓存目录。默认 `LOCAL_FILES_ONLY=true` 仍要求文件已在本地；明确下载或预热缓存时才临时设置为 `false`。本地目录存在但不完整或版本不符时，启动失败，不回退另一个模型。

启动先验证 token，再加载主模型并执行一次完整推理预热。加载或预热异常会阻止服务启动；完成后才开始接收请求。启动入口使用 `workers=1`、不开启 reload；增加进程会复制模型和各自的准入状态。

## 鉴权与入口边界

后端和检测端使用同一 `REPRE_GUARD_SERVICE_TOKEN`，至少 32 个无空白的可打印 ASCII 字符。替换示例占位值，将真实值保存在部署配置中，不提交或打印到日志。

`/health`、`/detect`、`/evidence/route` 均要求唯一且正确的 `X-RepreGuard-Token`。缺失、错误或重复 token 返回 `401 DETECT_SERVICE_UNAUTHORIZED`，先于请求体解析和请求推理。

| 检查 | 行为 |
|---|---|
| `/detect`、`/evidence/route` 请求体 | 最多 128 KiB，包含 JSON 开销，同时检查声明长度和实际 ASGI 流 |
| 无效或重复 `Content-Length` | `400 INVALID_CONTENT_LENGTH` |
| 请求体过大 | `413 REQUEST_BODY_TOO_LARGE` |
| 主检测 `text` | 1～20,000 字符，去除首尾空白后不得为空 |
| 主检测 token 数 | 最多配置的 512 tokens，包含特殊 tokens；超限返回 `422 INPUT_TOO_LONG` |
| Evidence 输入 | 严格字符串、1～20,000 Unicode 码点且非空白；拒绝额外字段和类型强转 |

主检测接口先完整计数再推理，不会静默截断过长输入。Evidence 请求校验失败返回固定 `422 INVALID_EVIDENCE_REQUEST`，不回显原文；其模型截断策略见下文。

## 同入口 action 分流与 PHP 转发

`POST /detect` 接受下列请求；两种分支均复用现有服务令牌与 128 KiB 请求体限制：

| JSON | 执行与响应 |
|---|---|
| `{"text":"文本"}` 或 `{"text":"文本","action":"detect"}` | 主模型分数、阈值和标签 |
| `{"text":"完整原文","action":"evidence"}` | Evidence Router 五字段响应，不进入主模型队列 |

Evidence 分支保持严格字符串和非空白校验，拒绝 `text/action` 之外的字段；其他 action 为 422。其并发、超时、失败降级和模型 SHA 语义与 `/evidence/route` 相同。原 `/evidence/route` 仍仅接受 `{"text":"完整原文"}`，保留旧客户端兼容。

PHP 可始终将同一外部地址转发到检测端 `/detect`，不必判断 action；但必须完整转发 JSON 请求、`X-RepreGuard-Token`、JSON 响应和上游 HTTP 状态码。后端的 Evidence 请求发送整篇原文一次，不为每个主检测分段追加 Router 推理。

先部署本版本检测端，再部署新后端；新后端遇到旧检测端或丢失 action 的转发器，可能收到普通主检测结果，应将其视为无效 Evidence 响应并仅降级 Evidence。验收必须通过实际 PHP 入口确认 `action=evidence` 得到 `status=routed`、完整响应和匹配 SHA，不能仅检查 HTTP 200。

## 主检测排队与调参

默认同时执行一次主检测推理，最多三个等待请求；CPU 回退时同样受此限制。准入容量已满时立即返回 `503 DETECT_QUEUE_FULL`；等待超过 15 秒返回 `503 DETECT_QUEUE_TIMEOUT`。两者均包含整数 `Retry-After`，默认 5 秒。等待中的客户端断开后移出队列，不执行推理。

`MAX_PENDING_REQUESTS` 可取 0～32；`QUEUE_TIMEOUT_SECONDS` 必须有限、严格大于 0 且不超过 60。先测量预热后的单请求推理 P95，再选择队列长度：

```text
max pending <= floor(max acceptable wait / p95 inference time)
```

默认三个等待位置和 15 秒仅以推理 P95 不超过 5 秒为调参假设，不是性能承诺。队列超时应小于后端调用检测服务的超时；增加队列不能提高模型吞吐量。

## 直接压测

下述工具绕过后端配额、用户鉴权、数据库和历史记录，仍使用服务 token。它会从同目录 `.env` 或进程环境读取 token，不需要把密钥放入命令参数。

```sh
python loadtest_detector_api.py --url http://127.0.0.1:9000/detect --users 100 --rounds 1 --chars 500
```

报告写入 `loadtest_results/detector-api-YYYYMMDD-HHMMSS.json`，包含每次请求的耗时、状态及失败摘要。默认请求超时 300 秒，可用 `--timeout` 调整；TLS 校验默认开启。

每轮同时发出 `--users` 个请求。100 并发用于观察突发准入，超容量出现 `503` 是预期行为；这些响应仍计入 `failure_count`，脚本写完报告后以退出码 1 结束。不要用“全部 200”作为这条突发命令的验收条件。

报告的总 `latency` 混合成功和失败请求，快速失败会影响 P95；它不是纯模型推理 P95。调队列时应分别查看成功请求延迟和状态分布，先取得预热后单请求基线。

## Evidence 配置与制品身份

Evidence 是独立可选的整文语言/领域 Router，默认关闭，不改变主检测分数和阈值。以下值需显式配置：

```dotenv
REPRE_GUARD_EVIDENCE_ENABLED=false
REPRE_GUARD_EVIDENCE_MODEL_PATH=./models/evidence-router
REPRE_GUARD_EVIDENCE_ARTIFACT_SHA256=ROUTER_ARTIFACT_SHA256
REPRE_GUARD_EVIDENCE_TIMEOUT_SECONDS=10
```

开启值接受 `true/1/yes/on`，关闭值接受 `false/0/no/off`。超时必须有限、严格大于 0 且不超过 60 秒。可选配置由消费者校验；错误配置或加载失败不阻止主检测启动。

Router 实例有 `off/ready/failed` 加载状态。关闭时不读模型目录、不加载语言识别器；开启后在启动阶段验证并加载一次。失败状态也被保留，重试加载需要新建实例或重启进程。请求复用模型和缓存身份，不逐次访问文件或重新计算 SHA。

模型目录必须是不可变的六文件目录，不能指向含研究报告的父目录：

```text
config.json
router_config.json
router_model.safetensors
sentencepiece.bpe.model
tokenizer.json
tokenizer_config.json
```

仅接受普通文件；缺失、额外条目、symlink/reparse point、加载中发生的普通文件变化、SHA 不符或配置不兼容都会被拒绝。两个配置 JSON 必须使用规范序列化且各不超过 64 KiB；类别顺序、模型标签映射、快速 tokenizer 词表、特殊 tokens 和 512-token 契约均会校验。

制品 SHA 是六文件 claim 映射 `{filename: {bytes, sha256}}` 的聚合 SHA-256：UTF-8 JSON、`ensure_ascii=False`、键排序、`allow_nan=False`、缩进两个空格并保留末尾换行。它不是权重文件 SHA。既有冻结 Router 的 pin：

```text
9a1a5be8f0d7599e682d30c7f50e3deb5f747b59ef720e288f6342e7c944f896
```

文件身份与 stat 快照仅能发现普通加载变化，不能证明目录不会被恶意改写；部署需保证目录不可变。运行消费者不检查研究 lineage，也不另设逐文件部署 pin。

## Evidence 推理与响应契约

模型保留一个 XLM-R 编码器、一个语言 head 和八个领域 head，严格加载 safetensors，使用 CPU FP32、eval/inference 模式。不引入研究 Trainer，也不修改 CUDA 状态、训练随机种子或进程线程限制。

支持语言为 `ar/de/en/es/fr/pt/ru/zh`，领域为 `academic/news/novel/seo/webtext/wiki`。`py3langid` 保留全部候选语言、启用 `norm_probs=True`，不设置信度阈值。每次请求在 XLM-R 分词前，对规范化的完整文档调用一次 `rank`：

- 与缓存的空输入排名完全相同，或首选语言为 `und/zxx`：`failed/language_undetermined`。
- 首选语言不在八种支持语言内：`unsupported/unsupported_language`；不将 `wuu/yue/ary/arz` 映射到 `zh/ar`。
- 首选语言受支持：执行冻结 Router，不以语言识别置信度或与 Router 的一致性拦截。

短文本和混合语言仍按整文处理，不做分段投票或返回 Mixed。这是有误差的支持范围检查，不能保证识别所有无意义输入。

模型预处理执行 NFKC 和空白规范化，分词一次；超过 510 个 payload tokens 时保留前 255 和后 255，加入两个特殊 tokens，并按八的倍数补齐至最多 512。该策略只影响 Router 模型输入，不截断后端整文特征分析。

每个受支持请求做一次模型 forward，以温度 1 得到联合分布；先按领域质量聚合选择语言，再在该语言内选领域。全局 48 类 argmax 不是实际路由规则，低置信度不弃权。`confidence.language/domain` 是诊断值，不是正确率。

内部消费者成功状态为 `predicted`，端点转换为 `routed`。内部 `invalid_text`、`invalid_evidence_router_config` 不直接作为端点业务 reason 返回。

HTTP 业务响应固定五字段：`schemaVersion=1`、`status`、`routerArtifactSha256`、`route`、`reason`；不含原文、生成器、路径、AI/Human 标签、主检测分数或内部异常。

| status | route / reason |
|---|---|
| `routed` | `{language, domain, confidence:{language,domain}}` / `null` |
| `unsupported` | `null` / `unsupported_language` |
| `failed` | `null` / `language_undetermined`、`model_unavailable`、`model_failure`、`busy` 或 `timeout` |

以上业务结果均为 HTTP 200，包括 `busy/timeout`。关闭或加载失败返回 `failed/model_unavailable` 且 SHA 为 null；就绪实例的业务响应保留已校验 SHA。鉴权、请求体和输入错误使用非 2xx；等待中的请求断开可返回 499。后端对应契约见其 `docs/detection-contract.md` 第 10 节。

Evidence 容量独立于主检测：一次执行、零排队，CPU 工作在线程中运行。超时或取消后，容量直到线程真正结束才释放，期间后续请求返回 `busy`；关闭时排空两类工作。此约束不能终止卡死的原生线程，也不能隔离进程 CPU/RSS。

## 测试与依赖

在已安装开发依赖的环境中执行：

```sh
python -B -m unittest discover -s tests -p test_evidence_router.py -v
python -B -m unittest discover -s tests -v
```

测试使用合成文件、假模型或 tiny XLM-R 和无监听器的实际 ASGI handler，覆盖语言决策、鉴权/请求体、私有失败、超时/取消后的容量保留及主检测回归，不下载或加载生产模型。真实 `py3langid` API 测试缺包时会跳过，应检查 skip 而不是仅看退出码。

Router 复用 `torch/transformers/safetensors/tokenizers/sentencepiece`。`requirements.txt` 声明 `py3langid==0.4.0`，需要 Python ≥3.10、NumPy ≥2.0；目标环境需具备这些依赖。压测工具需要 `httpx`，真实验收工具还需要 `psutil` 和后端验证器的运行依赖。既有 PowerShell 启动 wrapper 使用本机专用路径，不作为可移植启动入口。

## 显式真实模型验收

`smoke_evidence_runtime.py` 只使用准备好的本地主模型、Router 和 Bundle；下列大写参数是待替换的制品摘要和后端仓库路径，输出文件每次换新名称：

```sh
python -B smoke_evidence_runtime.py --router-model ./models/evidence-router --router-sha ROUTER_ARTIFACT_SHA256 --bundle ./artifacts/evidence-v1.bundle --bundle-sha BUNDLE_SHA256 --backend-root BACKEND_REPO_ROOT --output loadtest_results/evidence-runtime-run-01.json
```

若使用隔离安装的语言识别器，可追加 `--lid-site PY3LANGID_SITE`；环境已安装时省略。Router 与 Bundle 必须匹配，不能将 Router SHA 作为 Bundle SHA。

工具创建单 worker 子进程、临时 token 和回环临时端口，不修改 `.env`、启动后端、安装依赖或下载模型。常规退出排空准入；内存保留不足或关闭卡住时只停止它创建的子进程。报告记录实际设备、调用/forward 次数、时延和内存，不记录原文或 token。

当前验收 profile 要求主模型在 CUDA、Router 在 CPU FP32；通用服务的 CPU 回退不满足该 profile。三组分别为 Evidence 关闭、开启、以及注入 20 ms deadline 的真实超时。主检测样本 500 字符且检查 token 上限；三次主模型预热后，空闲/并发各测 30 次。Evidence 使用八语言、意大利语、无信息输入及 20,000 字符 EN/ZH 功能样本；实际响应通过后端 Bundle 验证器，不是准确率评估。

预声明预算：主模型标签和阈值一致、分数差 ≤1e-6；主检测 P95 增幅 ≤`max(基线*20%, 50 ms)`；正常 Evidence 成功响应低于 10 秒；可用内存高于 `max(2 GiB, 总内存*10%)`。CPU 线程配置不变，失败不会自动调参后改称通过。RSS 每 50 ms 采样，观察值包括同进程的两个模型。

历史测量、依赖差异与限制保留于 [2026-09-07 验收归档](archive/evidence-runtime-acceptance-2026-09-07.md)。历史通过、安装状态或 Evidence 开关不代表当前环境，也不构成部署、公开服务或认证许可；本说明不重跑或改写那些结果。
