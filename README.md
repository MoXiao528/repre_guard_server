# RepreGuard

AIDetector ���ı������񣬻��� FastAPI �ṩ AI/Human ���Ϳ�ѡ�� Evidence ·�ɡ�

```text
AIDetector-web �� AIDetector-Back �� RepreGuard
```

��������ģ���������û������ͼ����ʷ�ɺ�˹�����

## ģ��

�����ʹ�� [DetectRL-X XLM-RoBERTa Detector](https://huggingface.co/WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All)��ģ�Ͱ汾�� `config.py` �̶���

`score` ��ģ�Ͷ� AI ��� `LABEL_1` �ĸ��������Ĭ����ֵΪ `0.0028`���ﵽ��ֵ���� `AI`�����򷵻� `HUMAN`��

֧�� CUDA �� CPU��Ĭ������ʹ�� CUDA��

## ���ٿ�ʼ

��Ҫ Python 3.10 �����ϰ汾������������ڲֿ��Ŀ¼ִ�С�

### 1. ��װ����

```bash
python -m pip install -r requirements.txt
python -c "from pathlib import Path; import shutil; shutil.copyfile('.env.example', '.env') if not Path('.env').exists() else None"
```

### 2. ���û���

�༭ `.env`��

```dotenv
REPRE_GUARD_SERVICE_TOKEN=<����һ�µķ�������>
REPRE_GUARD_MODEL_PATH=./models/detector
REPRE_GUARD_MODEL_CACHE_DIR=./models/cache
```

�����������ٰ��� 32 ���޿հ׵Ŀɴ�ӡ ASCII �ַ���`.env` ��ģ���ļ���Ҫ�ύ���ֿ⡣

### 3. ����ģ��

������������������������ģ�Ͱ汾�������ļ���

```bash
python -c "from huggingface_hub import snapshot_download; from config import REQUIRED_MODEL_FILES, settings; snapshot_download(repo_id=settings.model_name, revision=settings.model_revision, local_dir=settings.model_path, allow_patterns=list(REQUIRED_MODEL_FILES))"
```

### 4. ��������

```bash
python run_roberta_server.py
```

Ĭ�ϼ��� `0.0.0.0:9000`������ʱ�Զ����ز�Ԥ��ģ�͡�������õ� worker�����ʷ�ΧӦ����������ǽ���ơ�

## �ӿ�

������������ӿڶ���Ҫ����ͷ��

```text
X-RepreGuard-Token: <��������>
```

| ���� | ·�� | ��; |
| --- | --- | --- |
| GET | `/health` | ������� |
| POST | `/detect` | AI/Human ��� |
| POST | `/evidence/route` | Evidence ���ԡ�����·�� |

### �ı����

����

```json
{
  "text": "��Ҫ�����ı���"
}
```

��Ӧʾ����

```json
{
  "score": 0.037,
  "threshold": 0.0028,
  "label": "AI",
  "model_name": "WUJUNCHAO/DetectRL-X-XLM-RoBERTa-Detector-All",
  "score_type": "probability"
}
```

�ͻ���Ӧ��� `score_type`�����ܽ��������͵ķ�����������ʹ�á�

�������ƣ���������� 128 KiB���ı�Ϊ 1��20,000 �� Unicode ����Ҳ���ȫΪ�հף��������� 512 tokens���������� tokens��token ���޷��� `422 INPUT_TOO_LONG`�����ᾲĬ�ضϡ�

Ĭ��ͬʱִ��һ����������������������ȴ����ȴ�����Ϊ 15 �롣��������ȴ���ʱ���� `503`��

## Evidence ·��

Evidence Ĭ�Ϲرգ�ͨ�� `.env` �е� `REPRE_GUARD_EVIDENCE_*` �������ã���Ҫ�����ṩ Router ģ��Ŀ¼�Ͳ��� SHA-256��

Router �� CPU ���ж���ƪ�ı������Ժ����򣬹����ѡ��д���ص�ο�ͳ�ơ��������� AI ���������޸����������

�ýӿڵ�ҵ����ʹ�� HTTP `200`�����÷������� `status` �� `reason`��δ���û����ʧ��ʱ���� `failed/model_unavailable`����Ȩ���������ʹ�÷� 2xx ״̬��

## ����

```bash
python -B -m unittest discover -s tests -v
```

��ͨ����ʹ��ģ���С��ģ�ͣ�����������ģ�ͣ���������ȱʧʱ��������Ӧ��顣

��ϸ���á�Router Э������������ [����˵��](docs/operations.md)����ʷ��ʵģ�����ռ��������Ƽ� [���ռ�¼](docs/archive/evidence-runtime-acceptance-2026-09-07.md)��������������������ֱ��¼��
