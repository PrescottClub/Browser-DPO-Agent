# src/agent/inference_module.py

from typing import Optional, Dict, Any
import torch

from .base_model import BaseModel


class InferenceModule(BaseModel):
    """
    推理模块，专门负责模型的推理生成功能。
    
    继承自BaseModel，专注于推理生成的具体实现，
    遵循单一职责原则。
    """

    def __init__(
        self, 
        model_name: str, 
        quantization_config=None,
        device_map: str = "auto"
    ):
        """
        初始化推理模块。

        Args:
            model_name (str): Hugging Face上的模型名称
            quantization_config: 量化配置
            device_map (str): 设备映射策略
        """
        super().__init__(model_name, quantization_config, device_map)
        
        # 推理配置
        self.default_generation_config = {
            "max_new_tokens": 256,
            "do_sample": False,  # 默认使用贪心解码
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": 50,
            "repetition_penalty": 1.1,
        }
    
    def update_generation_config(self, **kwargs):
        """
        更新生成配置。
        
        Args:
            **kwargs: 生成配置参数
        """
        self.default_generation_config.update(kwargs)
    
    def format_prompt(self, instruction: str, response_prefix: str = "") -> str:
        """
        格式化输入prompt。
        
        Args:
            instruction (str): 任务指令
            response_prefix (str): 响应前缀（可选）
            
        Returns:
            str: 格式化后的prompt
        """
        formatted = f"### Instruction:\n{instruction}\n\n### Response:\n{response_prefix}"
        return formatted
    
    def generate_completion(
        self, 
        prompt: str, 
        generation_config: Optional[Dict[str, Any]] = None,
        return_full_text: bool = False
    ) -> str:
        """
        根据给定的prompt生成模型响应。

        Args:
            prompt (str): 输入的任务指令
            generation_config (dict, optional): 生成配置，覆盖默认配置
            return_full_text (bool): 是否返回包含输入的完整文本
            
        Returns:
            str: 模型生成的响应文本
        """
        # 格式化输入
        if not prompt.startswith("### Instruction:"):
            formatted_prompt = self.format_prompt(prompt)
        else:
            formatted_prompt = prompt
        
        # 合并生成配置
        config = self.default_generation_config.copy()
        if generation_config:
            config.update(generation_config)
        
        # 分词
        inputs = self.tokenizer(
            formatted_prompt, 
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.get_device())
        
        # 生成响应
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=config["max_new_tokens"],
                do_sample=config["do_sample"],
                temperature=config["temperature"] if config["do_sample"] else None,
                top_p=config["top_p"] if config["do_sample"] else None,
                top_k=config["top_k"] if config["do_sample"] else None,
                repetition_penalty=config["repetition_penalty"],
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        # 解码输出
        if return_full_text:
            # 返回完整文本（包含输入）
            completion = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        else:
            # 只返回生成的部分
            response_ids = outputs[0][inputs.input_ids.shape[1]:]
            completion = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        
        return completion.strip()
    
    def generate_batch(
        self, 
        prompts: list, 
        generation_config: Optional[Dict[str, Any]] = None,
        batch_size: int = 4
    ) -> list:
        """
        批量生成响应。
        
        Args:
            prompts (list): 输入prompt列表
            generation_config (dict, optional): 生成配置
            batch_size (int): 批次大小
            
        Returns:
            list: 生成的响应列表
        """
        results = []
        
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            
            # 格式化batch
            formatted_batch = [
                self.format_prompt(p) if not p.startswith("### Instruction:") else p
                for p in batch_prompts
            ]
            
            # 分词
            inputs = self.tokenizer(
                formatted_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=1024
            ).to(self.get_device())
            
            # 合并配置
            config = self.default_generation_config.copy()
            if generation_config:
                config.update(generation_config)
            
            # 批量生成
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=config["max_new_tokens"],
                    do_sample=config["do_sample"],
                    temperature=config["temperature"] if config["do_sample"] else None,
                    top_p=config["top_p"] if config["do_sample"] else None,
                    top_k=config["top_k"] if config["do_sample"] else None,
                    repetition_penalty=config["repetition_penalty"],
                    eos_token_id=self.tokenizer.eos_token_id,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            # 解码批量输出
            for j, output in enumerate(outputs):
                input_length = inputs.input_ids[j].shape[0]
                response_ids = output[input_length:]
                completion = self.tokenizer.decode(response_ids, skip_special_tokens=True)
                results.append(completion.strip())
        
        return results
    
    def parse_response(self, response: str) -> Dict[str, str]:
        """
        解析模型响应，提取thought和action。
        
        Args:
            response (str): 模型生成的响应
            
        Returns:
            dict: 包含thought和action的字典
        """
        result = {"thought": "", "action": ""}
        
        # 简单的解析逻辑，可以根据需要改进
        lines = response.strip().split('\n')
        
        current_section = None
        for line in lines:
            line = line.strip()
            
            if line.lower().startswith('thought:') or line.lower().startswith('thinking:'):
                current_section = 'thought'
                content = line.split(':', 1)[1].strip() if ':' in line else ''
                result['thought'] = content
            elif line.lower().startswith('action:'):
                current_section = 'action'
                content = line.split(':', 1)[1].strip() if ':' in line else ''
                result['action'] = content
            elif current_section and line:
                # 继续当前section的内容
                if result[current_section]:
                    result[current_section] += ' ' + line
                else:
                    result[current_section] = line
        
        return result
    
    def predict(self, instruction: str) -> Dict[str, str]:
        """
        预测方法，结合生成和解析。
        
        Args:
            instruction (str): 任务指令
            
        Returns:
            dict: 包含thought和action的预测结果
        """
        response = self.generate_completion(instruction)
        return self.parse_response(response)
    
    def get_generation_stats(self) -> Dict[str, Any]:
        """
        获取生成统计信息。
        
        Returns:
            dict: 生成统计信息
        """
        return {
            "model_info": self.get_model_info(),
            "current_config": self.default_generation_config.copy(),
            "device": str(self.get_device()),
        }
    
    def set_eval_mode(self):
        """设置模型为评估模式"""
        if self._model is not None:
            self.model.eval()
    
    def set_train_mode(self):
        """设置模型为训练模式"""
        if self._model is not None:
            self.model.train() 