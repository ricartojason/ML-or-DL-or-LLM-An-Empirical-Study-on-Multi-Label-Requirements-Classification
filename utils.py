import json
import re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

def load_json(path):
    """
    加载JSON文件
    
    Args:
        path: JSON文件路径
        
    Returns:
        加载的JSON数据
    """
    with open(path, 'r') as file:
        data = json.load(file)
        return data

def safe_loads(text, default_value={"classification": []}):
    """
    安全解析可能格式不正确的JSON字符串
    
    Args:
        text: 要解析的JSON字符串
        default_value: 解析失败时返回的默认值
        
    Returns:
        解析后的Python对象
    """
    # 移除可能的markdown格式
    json_string = re.sub(r'^```json\n(.*)\n```$', r'\1', text.strip(), flags=re.DOTALL)
    json_string = re.sub(r'^```\n(.*)\n```$', r'\1', json_string, flags=re.DOTALL)
    
    try:
        # 尝试直接解析JSON
        return json.loads(json_string)
    except json.JSONDecodeError:
        # 如果失败，尝试提取格式中的分类信息
        try:
            # 尝试匹配"classification": [...] 模式
            match = re.search(r'"classification"\s*:\s*\[(.*?)\]', json_string, re.DOTALL)
            if match:
                # 提取分类数组内容
                classifications_str = match.group(1)
                # 提取所有引号中的内容作为分类项
                classifications = re.findall(r'"([^"]*)"', classifications_str)
                return {"classification": classifications}
            
            # 尝试从任意JSON格式中提取标签
            per = "Per" in json_string and ("true" in json_string.lower() or "per" in json_string.lower())
            sup = "Sup" in json_string and ("true" in json_string.lower() or "sup" in json_string.lower())
            rel = "Rel" in json_string and ("true" in json_string.lower() or "rel" in json_string.lower())
            usa = "Usa" in json_string and ("true" in json_string.lower() or "usa" in json_string.lower())
            mis = "Mis" in json_string and ("true" in json_string.lower() or "mis" in json_string.lower())
            
            result = []
            if per: result.append("Per")
            if sup: result.append("Sup")
            if rel: result.append("Rel")
            if usa: result.append("Usa")
            if mis: result.append("Mis")
            
            if result:
                return {"classification": result}
            
            print(f"无法解析JSON: {json_string}")
            return default_value
        except Exception as e:
            print(f"处理JSON时出错: {str(e)}, 原始JSON: {json_string}")
            return default_value

def load_model(model_path, checkpoint_path='', device='cuda:0'):
    """
    加载模型和分词器
    
    Args:
        model_path: 模型路径
        checkpoint_path: 检查点路径，可选
        device: 运行设备
        
    Returns:
        加载的模型和分词器
    """
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.bfloat16, trust_remote_code=True).eval().to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    # 设置pad_token
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    if checkpoint_path:
        model = PeftModel.from_pretrained(model, checkpoint_path).to(device)

    return model, tokenizer

def convert_to_serializable(obj):
    """
    将numpy值等转换为Python原生类型，以便JSON序列化
    
    Args:
        obj: 要转换的对象
        
    Returns:
        可序列化的对象
    """
    import numpy as np
    
    if isinstance(obj, np.float32) or isinstance(obj, np.float64):
        return float(obj)
    elif isinstance(obj, np.int32) or isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {k: convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(i) for i in obj]
    else:
        return obj 