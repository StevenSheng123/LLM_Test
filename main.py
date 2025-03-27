import argparse
import os
import json
from src.config import EvalConfig, ModelConfig, BenchmarkConfig
from src.models.vllm_model import VLLMModel
from src.evaluation.evaluator import Evaluator
from src.benchmarks.mmlu_benchmark import MMLUBenchmark

def parse_args():
    parser = argparse.ArgumentParser(description="评估大语言模型在多个基准测试上的性能")
    
    parser.add_argument("--model_path", type=str, required=True, help="模型路径或名称")
    parser.add_argument("--tokenizer_path", type=str, default=None, help="词元化器路径（可选）")
    parser.add_argument("--output_dir", type=str, default="results", help="结果输出目录")
    parser.add_argument("--batch_size", type=int, default=1, help="批处理大小")
    parser.add_argument("--max_tokens", type=int, default=2048, help="生成的最大token数")
    parser.add_argument("--tensor_parallel_size", type=int, default=1, help="张量并行大小")
    parser.add_argument("--benchmarks", type=str, nargs="+", 
                        choices=["MMLU", "MATH-500", "AIME", "Codeforces", "GPQA", "SWE-bench"],
                        default=["MMLU"], help="要运行的基准测试")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    
    return parser.parse_args()

def create_benchmark(benchmark_name, config):
    if benchmark_name == "MMLU":
        return MMLUBenchmark(
            data_path=f"data/MMLU",
            output_path=f"{config.output_dir}/mmlu_results.json",
            few_shot=5,
        )
    # 其他基准测试的实现将在后续添加
    else:
        raise ValueError(f"未实现的基准测试: {benchmark_name}")

def main():
    args = parse_args()
    
    # 如果提供了配置文件，从中加载配置
    if args.config and os.path.exists(args.config):
        with open(args.config, "r") as f:
            config_dict = json.load(f)
        config = EvalConfig(**config_dict)
    else:
        # 否则从命令行参数创建配置
        model_config = ModelConfig(
            model_name=os.path.basename(args.model_path),
            model_path=args.model_path,
            tokenizer_path=args.tokenizer_path,
            max_tokens=args.max_tokens,
            tensor_parallel_size=args.tensor_parallel_size,
        )
        
        benchmark_configs = [
            BenchmarkConfig(
                name=bench_name,
                data_path=f"data/{bench_name}",
                output_path=f"{args.output_dir}/{bench_name.lower()}_results.json",
            )
            for bench_name in args.benchmarks
        ]
        
        config = EvalConfig(
            model=model_config,
            benchmarks=benchmark_configs,
            output_dir=args.output_dir,
            batch_size=args.batch_size,
        )
    
    # 创建评估器
    evaluator = Evaluator(config)
    
    # 创建基准测试实例
    benchmark_instances = [
        create_benchmark(bench_config.name, config)
        for bench_config in config.benchmarks
    ]
    
    # 运行评估
    results = evaluator.run_evaluation(benchmark_instances)
    
    # 打印结果摘要
    print("\n=== 评估结果摘要 ===")
    for bench_name, metrics in results.items():
        print(f"{bench_name}: 准确率 = {metrics.get('accuracy', 0):.4f}")

if __name__ == "__main__":
    main() 