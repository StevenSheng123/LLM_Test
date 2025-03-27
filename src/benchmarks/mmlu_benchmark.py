import os
import json
import pandas as pd
from typing import Dict, List, Any, Optional
from src.benchmarks.base_benchmark import BaseBenchmark

class MMLUBenchmark(BaseBenchmark):
    def __init__(
        self,
        data_path: str = "data/MMLU",
        output_path: str = "results/mmlu_results.json",
        few_shot: int = 5,
        max_samples: Optional[int] = None,
        subjects: Optional[List[str]] = None,
    ):
        super().__init__(
            name="MMLU",
            data_path=data_path,
            output_path=output_path,
            few_shot=few_shot,
            max_samples=max_samples,
        )
        self.subjects = subjects
        
    def load_data(self) -> List[Dict[str, Any]]:
        """加载MMLU数据"""
        examples = []
        
        # 如果是直接加载JSON文件
        if self.data_path.endswith(".json"):
            with open(self.data_path, "r") as f:
                return json.load(f)
                
        # 否则假设是MMLU目录结构
        dev_path = os.path.join(self.data_path, "dev")
        test_path = os.path.join(self.data_path, "test")
        
        subject_dirs = []
        if self.subjects:
            subject_dirs = self.subjects
        else:
            # 获取所有主题
            if os.path.exists(test_path):
                subject_dirs = [f.replace("_test.csv", "") for f in os.listdir(test_path) if f.endswith("_test.csv")]
        
        for subject in subject_dirs:
            # 加载few-shot示例
            few_shot_examples = []
            if self.few_shot > 0 and os.path.exists(dev_path):
                dev_file = os.path.join(dev_path, f"{subject}_dev.csv")
                if os.path.exists(dev_file):
                    dev_df = pd.read_csv(dev_file)
                    for _, row in dev_df.iloc[:self.few_shot].iterrows():
                        few_shot_examples.append({
                            "question": row["question"],
                            "choices": [row["A"], row["B"], row["C"], row["D"]],
                            "answer": row["answer"]
                        })
            
            # 加载测试集
            test_file = os.path.join(test_path, f"{subject}_test.csv")
            if os.path.exists(test_file):
                test_df = pd.read_csv(test_file)
                for _, row in test_df.iterrows():
                    examples.append({
                        "subject": subject,
                        "question": row["question"],
                        "choices": [row["A"], row["B"], row["C"], row["D"]],
                        "answer": row["answer"],
                        "few_shot_examples": few_shot_examples
                    })
        
        if self.max_samples and len(examples) > self.max_samples:
            examples = examples[:self.max_samples]
            
        return examples
    
    def prepare_prompts(self, examples: List[Dict[str, Any]]) -> List[str]:
        """准备MMLU提示"""
        prompts = []
        
        for ex in examples:
            prompt = f"主题: {ex['subject']}\n\n"
            
            # 添加few-shot示例
            if "few_shot_examples" in ex and ex["few_shot_examples"]:
                prompt += "以下是一些例子：\n\n"
                for i, fs_ex in enumerate(ex["few_shot_examples"]):
                    prompt += f"问题 {i+1}: {fs_ex['question']}\n"
                    prompt += f"A. {fs_ex['choices'][0]}\n"
                    prompt += f"B. {fs_ex['choices'][1]}\n"
                    prompt += f"C. {fs_ex['choices'][2]}\n"
                    prompt += f"D. {fs_ex['choices'][3]}\n"
                    prompt += f"答案: {fs_ex['answer']}\n\n"
            
            # 添加当前问题
            prompt += f"问题: {ex['question']}\n"
            prompt += f"A. {ex['choices'][0]}\n"
            prompt += f"B. {ex['choices'][1]}\n"
            prompt += f"C. {ex['choices'][2]}\n"
            prompt += f"D. {ex['choices'][3]}\n"
            prompt += "答案: "
            
            prompts.append(prompt)
        
        return prompts
    
    def evaluate(self, examples: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, float]:
        """评估MMLU结果"""
        correct = 0
        total = len(predictions)
        
        subject_metrics = {}
        
        for ex, pred in zip(examples, predictions):
            # 提取预测的答案（A、B、C、D）
            pred = pred.strip().upper()
            if pred and pred[0] in ["A", "B", "C", "D"]:
                pred_answer = pred[0]
            else:
                pred_answer = "X"  # 标记无效答案
            
            correct_answer = ex["answer"]
            is_correct = pred_answer == correct_answer
            
            if is_correct:
                correct += 1
            
            # 按主题统计
            subject = ex["subject"]
            if subject not in subject_metrics:
                subject_metrics[subject] = {"correct": 0, "total": 0}
            
            subject_metrics[subject]["total"] += 1
            if is_correct:
                subject_metrics[subject]["correct"] += 1
        
        # 计算整体准确率
        accuracy = correct / total if total > 0 else 0.0
        
        # 计算每个主题的准确率
        subject_accuracies = {}
        for subject, counts in subject_metrics.items():
            subject_accuracies[f"{subject}_accuracy"] = counts["correct"] / counts["total"] if counts["total"] > 0 else 0.0
        
        # 合并结果
        metrics = {
            "accuracy": accuracy,
            "correct": correct,
            "total": total,
            **subject_accuracies
        }
        
        return metrics 