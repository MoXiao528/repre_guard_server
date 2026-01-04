import os
import json
import random
import numpy as np
import logging
import torch
from tqdm import tqdm
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, precision_score, recall_score,
    accuracy_score, f1_score, roc_auc_score
)
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, PreTrainedTokenizer, pipeline, set_seed
)
from metrics import get_roc_by_threshold,get_roc_metrics
import argparse
from repe import repe_pipeline_registry
repe_pipeline_registry()

class AIHumanFunctionModel:
    def __init__(
        self,
        model_name_or_path,
        ntrain,
        rep_token,
        batch_size,
        random_seed=2025,
        ai_weight=1,
        human_weight=1,
        n_difference=1,
        direction_method='pca',
        device=None,
    ):
        set_seed(random_seed)
        random.seed(random_seed)
        np.random.seed(random_seed)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model_name = os.path.basename(model_name_or_path)
        self.model = AutoModelForCausalLM.from_pretrained(model_name_or_path)
        self.model.to(self.device)
        use_fast_tokenizer = "LlamaForCausalLM" not in self.model.config.architectures
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, use_fast_tokenizer=use_fast_tokenizer, padding_side="left", legacy=False)
        self.tokenizer.pad_token_id = 0
        self.rep_reading_pipeline =  pipeline("rep-reading", model=self.model, tokenizer=self.tokenizer)
        self.ntrain = ntrain
        self.hidden_layers = list(range(-1, -self.model.config.num_hidden_layers, -1))
        self.rep_token = rep_token
        self.batch_size = batch_size
        self.n_difference = n_difference
        self.direction_method = direction_method
        self.ai_weight = ai_weight
        self.human_weight = human_weight
        self.rep_reader = None
        
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    def _validate_dataset_schema(self, dataset, dataset_name: str) -> None:
        if not isinstance(dataset, list):
            raise ValueError(f"{dataset_name} 数据格式错误，期望 list，实际为 {type(dataset)}。")
        if not dataset:
            raise ValueError(f"{dataset_name} 数据为空，无法继续训练/评估。")
        required_pair_keys = {"direct_prompt", "human_text"}
        required_label_keys = {"text", "label"}
        for idx, item in enumerate(tqdm(dataset, desc=f"{dataset_name}校验", leave=False)):
            if not isinstance(item, dict):
                raise ValueError(
                    f"{dataset_name} 第 {idx} 条数据格式错误，期望 dict，实际为 {type(item)}。"
                )
            has_pair = required_pair_keys.issubset(item.keys())
            has_label = required_label_keys.issubset(item.keys())
            if not has_pair and not has_label:
                raise KeyError(
                    f"{dataset_name} 第 {idx} 条数据字段不匹配，"
                    "请确保包含 direct_prompt/human_text 或 text/label。"
                )
            if has_pair and (not item["direct_prompt"] or not item["human_text"]):
                raise ValueError(
                    f"{dataset_name} 第 {idx} 条数据文本为空，请检查 direct_prompt/human_text。"
                )
            if has_label and not item["text"]:
                raise ValueError(
                    f"{dataset_name} 第 {idx} 条数据文本为空，请检查 text。"
                )

    def _normalize_label_dataset(self, dataset, dataset_name: str):
        ai_texts = []
        human_texts = []
        for idx, item in enumerate(dataset):
            label = str(item.get("label", "")).strip().lower()
            text = item.get("text", "")
            if not text:
                continue
            if label in {"llm", "ai", "machine"}:
                ai_texts.append(text)
            elif label in {"human"}:
                human_texts.append(text)
            else:
                raise ValueError(
                    f"{dataset_name} 第 {idx} 条数据 label={item.get('label')} 不被支持，"
                    "请使用 human/llm(ai)。"
                )
        pair_count = min(len(ai_texts), len(human_texts))
        if pair_count == 0:
            raise ValueError(
                f"{dataset_name} 未找到可用的 AI/HUMAN 配对样本，无法继续训练/评估。"
            )
        if len(ai_texts) != len(human_texts):
            logging.warning(
                "%s AI/HUMAN 数量不一致，将按最小数量配对。AI=%s HUMAN=%s",
                dataset_name,
                len(ai_texts),
                len(human_texts),
            )
        return [
            {"direct_prompt": ai_texts[i], "human_text": human_texts[i]}
            for i in range(pair_count)
        ]

    def _normalize_dataset(self, dataset, dataset_name: str):
        sample = dataset[0]
        if "direct_prompt" in sample and "human_text" in sample:
            return dataset
        if "text" in sample and "label" in sample:
            return self._normalize_label_dataset(dataset, dataset_name)
        raise ValueError(
            f"{dataset_name} 无法识别的数据结构，请检查字段是否包含 "
            "direct_prompt/human_text 或 text/label。"
        )

    def ai_human_function_dataset(self, train_dataset: str, tokenizer: PreTrainedTokenizer):
        self._validate_dataset_schema(train_dataset, "训练集")
        train_dataset = self._normalize_dataset(train_dataset, "训练集")
        pos_statements = []
        neg_statements = []
        # ai_datasets = [item for item in train_dataset if item.get("label") == "llm"]
        # human_datasets = [item for item in train_dataset if item.get("label") == "human"]
        ai_datasets = [item['direct_prompt'] for item in train_dataset]
        human_datasets = [item['human_text'] for item in train_dataset]

        for ai_data, human_data in zip(ai_datasets, human_datasets):
            # if ai_data['id'] == human_data['id'] and ai_data['domain'] == human_data['domain']:
                tokens_pos_statement = tokenizer.tokenize(ai_data)
                tokens_neg_statement = tokenizer.tokenize(human_data)

                string_tokens_pos_statement = tokenizer.convert_tokens_to_string(tokens_pos_statement)
                string_tokens_neg_statement = tokenizer.convert_tokens_to_string(tokens_neg_statement)
                pos_statements.append(string_tokens_pos_statement)
                neg_statements.append(string_tokens_neg_statement)

        combined_data = [[pos, neg] for pos, neg in zip(pos_statements, neg_statements)]
        train_data = combined_data
        train_labels = []
        for d in train_data:
            true_s = d[0]
            random.shuffle(d)
            train_labels.append([s == true_s for s in d])

        train_data = np.concatenate(train_data).tolist()

        return {
            'train': {'data': train_data, 'labels': train_labels}
        }

    def process_data(self, data, mode="train"):
        dataset_name = "训练集" if mode == "train" else "测试集"
        self._validate_dataset_schema(data, dataset_name)
        if self.rep_reader is None:
            raise RuntimeError(
                "rep_reader 尚未初始化，请先调用 process_train_data/fit_rep_reader，"
                "或确保已成功加载方向向量。"
            )
        data = self._normalize_dataset(data, dataset_name)
        input_statements = []
        input_labels = []
        # ai_datasets = [item for item in data if item.get("label") == "llm"]
        # human_datasets = [item for item in data if item.get("label") == "human"]
        ai_datasets = [item['direct_prompt'] for item in data]
        human_datasets = [item['human_text'] for item in data]
    
        for ai_data, human_data in zip(ai_datasets, human_datasets):
            # if ai_data['id'] == human_data['id'] and ai_data['domain'] == human_data['domain']:
                input_statements.append(ai_data)
                input_labels.append(1)
                input_statements.append(human_data)
                input_labels.append(0)
        
        all_sentence_scores = []
        for statement in tqdm(input_statements):
            H_test_token = self.rep_reading_pipeline([statement],
                                    rep_reader=self.rep_reader,
                                    rep_token=0,
                                    hidden_layers=self.hidden_layers)
            all_token_scores = []
            
            num_tokens = len(H_test_token[0][-1][0])

            for token_idx in range(1,num_tokens,1):
                token_scores = []

                for layer in self.hidden_layers:
                    # 将当前 token 在当前层的分数添加到 token_scores 列表中
                    token_score_in_layer = H_test_token[0][layer][0][token_idx] * self.rep_reader.direction_signs[layer][0]
                    token_scores.append(token_score_in_layer)
                
                # 将当前 token 的所有层分数添加到 all_token_scores 中
                all_token_scores.append(token_scores)
            all_sentence_scores.append(all_token_scores)
    
        json_data = []
        for statement, sentence_score, label in zip(input_statements, all_sentence_scores, input_labels):
            data = {
                f"{mode}_input_statement": statement,
                "rep_reader_scores_dict": np.mean(sentence_score),
                f"{mode}_input_label": label
            }
            json_data.append(data)
    
        return json_data

    def save_json(self, data, file_path):
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)

    def process_train_data(self,train_data):
        # logging.info(f"Train in {test_data_path}")
        # train_data = json.load(open(train_data_path, "r"))
        dataset = self.ai_human_function_dataset(train_data, self.tokenizer)
   
        self.rep_reader = self.rep_reading_pipeline.get_directions(
            dataset['train']['data'],
            rep_token=self.rep_token,
            hidden_layers=self.hidden_layers,
            n_difference=self.n_difference,
            train_labels=dataset['train']['labels'],
            direction_method=self.direction_method,
            batch_size=self.batch_size,
            ai_weight=self.ai_weight,
            human_weight=self.human_weight,
        )

        train_json_data = self.process_data(train_data, mode="train")
        # train_file_name = os.path.basename(f"{self.train_data_path.split('.json')[0]}_ntrain_{self.ntrain}_reptoken_{self.rep_token}")
        # self.save_json(train_json_data, f'results/{train_file_name}.json')

        return train_json_data

    def fit_rep_reader(self, train_data):
        dataset = self.ai_human_function_dataset(train_data, self.tokenizer)

        self.rep_reader = self.rep_reading_pipeline.get_directions(
            dataset['train']['data'],
            rep_token=self.rep_token,
            hidden_layers=self.hidden_layers,
            n_difference=self.n_difference,
            train_labels=dataset['train']['labels'],
            direction_method=self.direction_method,
            batch_size=self.batch_size,
            ai_weight=self.ai_weight,
            human_weight=self.human_weight,
        )

    def score_text(self, text: str) -> float:
        if self.rep_reader is None:
            raise RuntimeError(
                "rep_reader 未初始化，请先调用 fit_rep_reader，或通过 "
                "REPRE_GUARD_READER_PATH 加载已保存的方向向量。"
            )

        H_test_token = self.rep_reading_pipeline(
            [text],
            rep_reader=self.rep_reader,
            rep_token=0,
            hidden_layers=self.hidden_layers,
        )
        all_token_scores = []
        num_tokens = len(H_test_token[0][-1][0])

        for token_idx in range(1, num_tokens, 1):
            token_scores = []
            for layer in self.hidden_layers:
                token_score_in_layer = (
                    H_test_token[0][layer][0][token_idx]
                    * self.rep_reader.direction_signs[layer][0]
                )
                token_scores.append(token_score_in_layer)
            all_token_scores.append(token_scores)

        return float(np.mean(all_token_scores))

    def process_test_data(self,test_data):
        # logging.info(f"Test in {test_data_path}")
        # test_data = json.load(open(test_data_path, "r"))

        test_json_data = self.process_data(test_data, mode="test")

        # test_file_name = f"{os.path.basename(self.test_data_path.split('.json')[0])}_BY_{os.path.basename(self.train_data_path.split('.json')[0])}_ntrain_{self.ntrain}_reptoken_{self.rep_token}"
        # self.save_json(test_json_data, f'results/{test_file_name}.json')
        return test_json_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument('--train_data_path', type=str, required=True)
    parser.add_argument('--test_data_path', type=str, required=True)
    parser.add_argument('--ntrain', default=128, type=int)
    parser.add_argument('--rep_token', default=-1, type=float)
    parser.add_argument('--batch_size', default=16, type=int)
    # parser.add_argument('--mode',default='test',type=str)
    args = parser.parse_args()
    # entrance(args)
