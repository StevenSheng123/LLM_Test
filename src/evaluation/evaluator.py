import os
import json
import time
from typing import Dict, List, Any, Optional
import pandas as pd

from src.models.vllm_model import VLLMModel
from src.benchmarks.base_benchmark import BaseBenchmark
from src.config import EvalConfig

class Evaluator:
    def __init__(self, config: EvalConfig):
        self.config = config
        self.results = {}
        
        # 初始化模型
        self.model = VLLMModel(
            model_name_or_path=config.model.model_path or config.model.model_name,
            tokenizer_path=config.model.tokenizer_path,
            max_tokens=config.model.max_tokens,
            temperature=config.model.temperature,
            top_p=config.model.top_p,
            tensor_parallel_size=config.model.tensor_parallel_size,
        )
        
        # 确保输出目录存在
        os.makedirs(config.output_dir, exist_ok=True)
    
    def evaluate_benchmark(self, benchmark: BaseBenchmark) -> Dict[str, float]:
        """评估单个基准测试"""
        print(f"开始评估基准: {benchmark.name}")
        
        # 加载数据
        examples = benchmark.load_data()
        if benchmark.max_samples and len(examples) > benchmark.max_samples:
            examples = examples[:benchmark.max_samples]
        
        # 准备提示
        prompts = benchmark.prepare_prompts(examples)
        
        # 生成回复
        start_time = time.time()
        predictions = self.model.batch_generate(prompts, self.config.batch_size)
        end_time = time.time()
        
        # 评估结果
        metrics = benchmark.evaluate(examples, predictions)
        metrics["time_seconds"] = end_time - start_time
        metrics["samples"] = len(examples)
        
        # 保存结果
        benchmark.save_results(examples, predictions, metrics)
        
        return metrics
    
    def run_evaluation(self, benchmark_instances: List[BaseBenchmark]) -> Dict[str, Dict[str, float]]:
        """运行所有基准测试的评估"""
        results = {}
        
        for benchmark in benchmark_instances:
            metrics = self.evaluate_benchmark(benchmark)
            results[benchmark.name] = metrics
        
        # 保存汇总结果
        summary_path = os.path.join(self.config.output_dir, "summary.json")
        with open(summary_path, "w") as f:
            json.dump(results, f, indent=2)
        
        # 创建可读性更强的表格
        summary_data = []
        for bench_name, metrics in results.items():
            row = {"benchmark": bench_name}
            row.update(metrics)
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_csv = os.path.join(self.config.output_dir, "summary.csv")
        summary_df.to_csv(summary_csv, index=False)
        
        print(f"评估完成。汇总结果已保存到 {summary_path} 和 {summary_csv}")
        
        return results 