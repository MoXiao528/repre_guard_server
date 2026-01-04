import argparse
import logging
from repreGuard_detector import AIHumanFunctionModel
import numpy as np
import logging
import json
from metrics import get_roc_by_threshold,get_roc_metrics
import os
import random
from collections import defaultdict


def bootstrap_sample_by_domain(data, ntrain,random_seed=2025,domain_key="domain"):
    random.seed(random_seed)
    domain_groups = defaultdict(list)
    for item in data:
        domain_groups[item[domain_key]].append(item)

    
    ordered_domains = [(domain, domain_groups[domain]) for domain in sorted(domain_groups.keys())]

    sampled_data = []
    while len(sampled_data) < ntrain:
        for domain, items in ordered_domains:
            if len(sampled_data) >= ntrain:
                break
            # Randomly select a sample from the current field (with put back sampling)
            sampled_data.append(random.choice(items))

    return sampled_data

def _load_json_data(path: str, dataset_name: str):
    try:
        with open(path, "r", encoding="utf-8") as json_file:
            data = json.load(json_file)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"未找到 {dataset_name} 文件: {path}，请确认路径或文件名是否正确。"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"{dataset_name} JSON 解析失败: {path}，请检查是否为合法 JSON。"
        ) from exc
    if not isinstance(data, list):
        raise ValueError(f"{dataset_name} 数据格式错误，期望 list，实际为 {type(data)}。")
    if not data:
        raise ValueError(f"{dataset_name} 数据为空，无法继续计算。")
    return data


def process_eval(args,train_json_data, test_json_data,test_data_path):
    print(f"Eval in {args.train_data_path}")
    # with open(train_filepath, 'r') as json_file:
    #     train_dataset_result = json.load(json_file)
    real_preds = []
    sample_preds = []

    for item in train_json_data:
        if item["train_input_label"] == 0:
            real_preds.append(np.mean(item['rep_reader_scores_dict']))
        elif item["train_input_label"] == 1:
            sample_preds.append(np.mean((item['rep_reader_scores_dict'])))
        
    roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy,tpr_at_fpr_0_01 = get_roc_metrics(real_preds,sample_preds)

    train_result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tpr_at_fpr_0_01": tpr_at_fpr_0_01
        }
    print(f"Train result: {train_result}")
    print(
        f"Train samples: {len(train_json_data)} "
        f"(AI={sum(1 for item in train_json_data if item['train_input_label'] == 1)}, "
        f"HUMAN={sum(1 for item in train_json_data if item['train_input_label'] == 0)})"
    )

    print(f"Eval in {test_data_path}")
    real_preds = []
    sample_preds = []

    for item in test_json_data:
        if item["test_input_label"] == 0:
            real_preds.append(np.mean(item['rep_reader_scores_dict']))
        elif item["test_input_label"] == 1:
            sample_preds.append(np.mean((item['rep_reader_scores_dict'])))
        
    roc_auc, optimal_threshold, conf_matrix, precision, recall, f1, accuracy,tpr_at_fpr_0_01 = get_roc_by_threshold(real_preds,
                                                                                            sample_preds,threshold=optimal_threshold)
    test_result = {
            "roc_auc": roc_auc,
            "optimal_threshold": optimal_threshold,
            "conf_matrix": conf_matrix,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "tpr_at_fpr_0_01": tpr_at_fpr_0_01
        }
    print(f"Test result: {test_result}")
    print(
        f"Test samples: {len(test_json_data)} "
        f"(AI={sum(1 for item in test_json_data if item['test_input_label'] == 1)}, "
        f"HUMAN={sum(1 for item in test_json_data if item['test_input_label'] == 0)})"
    )

    return train_result,test_result

def entrance(args):
    model = AIHumanFunctionModel(model_name_or_path=args.model_name_or_path, 
                            ntrain=args.ntrain,
                            rep_token=args.rep_token,
                            batch_size=args.batch_size,
                            random_seed=args.random_seed)
    if args.bootstrap_iter == -1:
        logging.info(f"Train in {args.train_data_path}")
        train_data_path = args.train_data_path.strip()
        train_data = _load_json_data(train_data_path, "训练集")[:args.ntrain]

        train_json_data = model.process_train_data(train_data=train_data)

        test_data_paths = args.test_data_paths.split(",")
        print(f"Test data paths: {test_data_paths}")
        for test_data_path in test_data_paths:
            test_data_path = test_data_path.strip()
            logging.info(f"Test in {test_data_path}")
            test_data = _load_json_data(test_data_path, "测试集")

            test_json_data = model.process_test_data(test_data=test_data)
            
            train_result,test_result = process_eval(args,train_json_data, test_json_data,test_data_path)
            result = {"train_result": train_result, "test_result": test_result}
            # result_file_name = f"{os.path.basename(test_data_path.split('.json')[0])}_BY_{os.path.basename(args.train_data_path.split('.json')[0])}_ntrain_{model.ntrain}_reptoken_{model.rep_token}"
            result_file_name = f"{os.path.basename(args.model_name_or_path.split('/')[-1])}_{os.path.basename(test_data_path.split('.json')[0])}_BY_{os.path.basename(args.train_data_path.split('.json')[0])}_ntrain_{model.ntrain}_reptoken_{model.rep_token}"
            os.makedirs('results/score_results', exist_ok=True)
            with open(f'results/{result_file_name}.json', 'w') as json_file:
                json.dump(result, json_file, indent=4)
            with open(f'results/score_results/{result_file_name}_test_score.json', 'w') as json_file:
                json.dump(test_json_data, json_file, indent=4)
            with open(f'results/score_results/{result_file_name}_train_score.json', 'w') as json_file:
                json.dump(train_json_data, json_file, indent=4)
    elif args.bootstrap_iter > 0:
        logging.info(f"Train in {args.train_data_path} using bootstrap")
        train_data_path = args.train_data_path.strip()
        ori_train_data = _load_json_data(train_data_path, "训练集")
        random.seed(args.random_seed)
        random_seeds = [random.randint(1, 100) for _ in range(args.bootstrap_iter)]
        print(f"Random seeds: {random_seeds}")
        for random_seed in random_seeds:
            logging.info(f"Bootstrap iter. random seed: {random_seed}")
            random.seed(random_seed)
            train_data = bootstrap_sample_by_domain(ori_train_data, args.ntrain,random_seed=random_seed)
            train_json_data = model.process_train_data(train_data=train_data)
            test_data_paths = args.test_data_paths.split(",")
            print(f"Test data paths: {test_data_paths}")
            for test_data_path in test_data_paths:
                test_data_path = test_data_path.strip()
                logging.info(f"Test in {test_data_path}")
                test_data = _load_json_data(test_data_path, "测试集")
                test_json_data = model.process_test_data(test_data=test_data)
                train_result,test_result = process_eval(args,train_json_data, test_json_data,test_data_path)
                result = {"train_result": train_result, "test_result": test_result}
                # result_file_name = f"{os.path.basename(test_data_path.split('.json')[0])}_BY_{os.path.basename(args.train_data_path.split('.json')[0])}_ntrain_{model.ntrain}_reptoken_{model.rep_token}_bootstrap_seed_{random_seed}"
                result_file_name = f"{os.path.basename(args.model_name_or_path.split('/')[-1])}_{os.path.basename(test_data_path.split('.json')[0])}_BY_{os.path.basename(args.train_data_path.split('.json')[0])}_ntrain_{model.ntrain}_reptoken_{model.rep_token}_bootstrap_seed_{random_seed}"
                os.makedirs('results/score_results', exist_ok=True)
                with open(f'results/{result_file_name}.json', 'w') as json_file:
                    json.dump(result, json_file, indent=4)
                with open(f'results/score_results/{result_file_name}_test_score.json', 'w') as json_file:
                    json.dump(test_json_data, json_file, indent=4)
                with open(f'results/score_results/{result_file_name}_train_score.json', 'w') as json_file:
                    json.dump(train_json_data, json_file, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name_or_path', type=str, required=True)
    parser.add_argument(
        '--train_data_path',
        type=str,
        required=False,
        default="direct_prompt_train.json",
        help="训练集路径，默认读取当前目录下的 direct_prompt_train.json",
    )
    parser.add_argument(
        '--test_data_paths',
        type=str,
        required=False,
        default="direct_prompt_test.json",
        help="测试集路径，支持多个文件用英文逗号分隔。默认 direct_prompt_test.json",
    )
    parser.add_argument('--ntrain', default=128, type=int,required=False)
    parser.add_argument('--bootstrap_iter', default=-1, type=int,required=False)
    parser.add_argument('--rep_token', default=-1, type=float,required=False)
    parser.add_argument('--batch_size', default=16, type=int,required=False)
    parser.add_argument('--random_seed', default=2025, type=int, required=False)
    args = parser.parse_args()
    entrance(args)
