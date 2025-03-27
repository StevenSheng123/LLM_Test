from vllm import LLM, SamplingParams
import os
from typing import List, Dict, Any, Optional

class VLLMModel:
    def __init__(
        self,
        model_name_or_path: str,
        tokenizer_path: Optional[str] = None,
        max_tokens: int = 2048,
        temperature: float = 0.0,
        top_p: float = 1.0,
        tensor_parallel_size: int = 1,
    ):
        """
        初始化vLLM模型
        
        Args:
            model_name_or_path: 模型名称或路径
            tokenizer_path: 词元化器路径，如果不提供则使用model_name_or_path
            max_tokens: 生成的最大token数
            temperature: 采样温度
            top_p: 核采样参数
            tensor_parallel_size: 张量并行大小
        """
        self.model_name = os.path.basename(model_name_or_path)
        tokenizer_path = tokenizer_path or model_name_or_path
        
        self.llm = LLM(
            model=model_name_or_path,
            tokenizer=tokenizer_path,
            tensor_parallel_size=tensor_parallel_size,
        )
        
        self.sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )
    
    def generate(self, prompts: List[str]) -> List[str]:
        """生成回复"""
        outputs = self.llm.generate(prompts, self.sampling_params)
        return [output.outputs[0].text for output in outputs]
    
    def batch_generate(self, prompts: List[str], batch_size: int = 1) -> List[str]:
        """批量生成回复"""
        results = []
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_results = self.generate(batch_prompts)
            results.extend(batch_results)
        return results 