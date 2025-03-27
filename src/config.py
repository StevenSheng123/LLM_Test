from pydantic import BaseModel
from typing import List, Optional, Dict, Any, Union

class ModelConfig(BaseModel):
    model_name: str
    model_path: Optional[str] = None
    tokenizer_path: Optional[str] = None
    max_tokens: int = 2048
    temperature: float = 0.0
    top_p: float = 1.0
    vllm: bool = True
    tensor_parallel_size: int = 1
    
class BenchmarkConfig(BaseModel):
    name: str
    data_path: str
    output_path: str
    metric: str = "accuracy"
    few_shot: int = 0
    max_samples: Optional[int] = None
    
class EvalConfig(BaseModel):
    model: ModelConfig
    benchmarks: List[BenchmarkConfig]
    output_dir: str = "results"
    batch_size: int = 1
    seed: int = 42 