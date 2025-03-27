from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import json
import os
import pandas as pd

class BaseBenchmark(ABC):
    def __init__(
        self,
        name: str,
        data_path: str,
        output_path: str,
        few_shot: int = 0,
        max_samples: Optional[int] = None,
    ):
        self.name = name
        self.data_path = data_path
        self.output_path = output_path
        self.few_shot = few_shot
        self.max_samples = max_samples
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    @abstractmethod
    def load_data(self) -> List[Dict[str, Any]]:
        """加载基准测试数据"""
        pass
    
    @abstractmethod
    def prepare_prompts(self, examples: List[Dict[str, Any]]) -> List[str]:
        """准备输入提示"""
        pass
    
    @abstractmethod
    def evaluate(self, examples: List[Dict[str, Any]], predictions: List[str]) -> Dict[str, float]:
        """评估模型性能"""
        pass
    
    def save_results(self, examples: List[Dict[str, Any]], predictions: List[str], metrics: Dict[str, float]):
        """保存结果和指标"""
        results = []
        for ex, pred in zip(examples, predictions):
            item = {
                "input": ex.get("input", ""),
                "reference": ex.get("reference", ""),
                "prediction": pred,
            }
            results.append(item)
        
        # 保存详细结果
        with open(self.output_path, "w") as f:
            json.dump({
                "benchmark": self.name,
                "metrics": metrics,
                "results": results,
            }, f, indent=2)
        
        # 也保存为CSV格式
        csv_path = self.output_path.replace(".json", ".csv")
        pd.DataFrame(results).to_csv(csv_path, index=False)
        
        print(f"结果已保存到 {self.output_path} 和 {csv_path}") 