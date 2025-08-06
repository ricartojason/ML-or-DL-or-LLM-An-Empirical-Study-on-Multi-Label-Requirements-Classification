import torch
from typing import List, Dict
from tqdm import tqdm

from .utils import load_json, safe_loads, load_model
from .metrics import calculate_metrics

def predict(model, tokenizer, text, device='cuda:0'):
    """
    使用模型预测单个文本的分类结果
    
    Args:
        model: 模型
        tokenizer: 分词器
        text: 要预测的文本
        device: 运行设备
        
    Returns:
        预测结果
    """
    prompt = f"""The following is a piece of requirement text, please analyze its content belongs to which kind of non-functional requirements.

                There are only five tags: Per (Performance), Usa (Usability), Sup (Supportability), Rel (Reliability), Mis (Miscellaneous).
                A statement can belong to more than one category at the same time.

                Please provide your answer in the following JSON format exactly:
                {{"classification": ["Tag1", "Tag2", ...]}}

                Where Tag1, Tag2, etc. are the tags you've identified from the list above.
                Example: {{"classification": ["Per", "Usa"]}}

    The requirement text is: {text}"""
    
    # 根据输入类型调整处理方式
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, 
                                          add_generation_prompt=True,
                                          tokenize=True,
                                          return_tensors='pt').to(device)
    
    # 生成参数
    gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "top_k": 1}
    
    with torch.no_grad():
        if isinstance(inputs, dict):
            # 如果是字典，使用**inputs展开
            outputs = model.generate(**inputs, **gen_kwargs)
            input_length = inputs['input_ids'].shape[1]
        else:
            # 如果是Tensor，直接作为input_ids参数传递
            outputs = model.generate(inputs, **gen_kwargs)
            input_length = inputs.shape[1]
        
        outputs = outputs[:, input_length:]
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        parsed_result = safe_loads(response)
        
        # 确保结果格式正确
        if parsed_result is None or "classification" not in parsed_result:
            return {"classification": []}
            
        return parsed_result

def build_prompt(text):
    """
    构建提示文本
    
    Args:
        text: 输入文本
        
    Returns:
        构建好的提示列表
    """
    prompt = f"""
            The following is a piece of requirement text, please analyze its content belongs to which kind of non-functional requirements.

            There are only five tags: Per (Performance), Usa (Usability), Sup (Supportability), Rel (Reliability), Mis (Miscellaneous).
            A text can belong to more than one category at the same time.

            Please provide your answer in the following JSON format exactly:
            {{"classification": ["Tag1", "Tag2", ...]}}

            Where Tag1, Tag2, etc. are the tags you've identified from the list above.
            Example: {{"classification": ["Per", "Usa"]}}

            The requirement text is: {text}    
            """
    return [{"role": "user", "content": prompt}]

def predict_batch(model, tokenizer, contents: List[str], device='cuda', debug=False):
    """
    批量预测文本分类
    
    Args:
        model: 模型
        tokenizer: 分词器
        contents: 文本列表
        device: 运行设备
        debug: 是否打印调试信息
        
    Returns:
        预测结果列表
    """
    prompts = [build_prompt(content) for content in contents]
    inputs = tokenizer(
        tokenizer.apply_chat_template(prompts, add_generation_prompt=True, tokenize=False),
        padding=True,
        return_tensors="pt"
    ).to(device)
    
    default_response = {'classification': []}
    gen_kwargs = {"max_new_tokens": 512, "do_sample": True, "top_k": 1}
    
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        responses = []
        for i in range(outputs.size(0)):
            # 输出生成标记之后所有生成的内容
            output = outputs[i, inputs['input_ids'].shape[1]:]
            response = tokenizer.decode(output, skip_special_tokens=True)
            responses.append(safe_loads(response, default_response))
        return responses

def run_test(model, tokenizer, test_data, device='cuda:0', debug=False):
    """
    对测试数据运行评估（单样本模式）
    
    Args:
        model: 模型
        tokenizer: 分词器
        test_data: 测试数据
        device: 运行设备
        debug: 是否打印调试信息
        
    Returns:
        评估结果
    """
    results = []
    
    # 用于跟踪真实和预测标签
    true_labels_list = []
    pred_labels_list = []
    
    for i, item in enumerate(test_data):
        input_text = item['input']
        true_labels = item['label']
        
        prediction = predict(model, tokenizer, input_text, device)
        pred_labels = prediction.get('classification', [])
        
        true_labels_list.append(true_labels)
        pred_labels_list.append(pred_labels)
        
        # 存储每个样本的结果
        is_correct = set(true_labels) == set(pred_labels)
        results.append({
            'input': input_text,
            'true_labels': true_labels,
            'pred_labels': pred_labels,
            'is_correct': is_correct
        })
        
        if debug and i % (max(1, len(test_data) // 20)) == 0:
            print(f"进度: {(i+1)*100/len(test_data):.2f}%")
            print(f"样本: '{input_text[:50]}...'")
            print(f"真实标签: {true_labels}")
            print(f"预测标签: {pred_labels}")
            print("-" * 50)
    
    # 计算评估指标
    metrics = calculate_metrics(true_labels_list, pred_labels_list)
    
    return {
        'results': results,
        'metrics': metrics
    }

def run_test_batch(model, tokenizer, test_data: List[Dict], batch_size: int = 8, device='cuda', debug=False):
    """
    对测试数据运行评估（批处理模式）
    
    Args:
        model: 模型
        tokenizer: 分词器
        test_data: 测试数据
        batch_size: 批大小
        device: 运行设备
        debug: 是否打印调试信息
        
    Returns:
        评估结果
    """
    print(f"以批处理模式运行，batch_size={batch_size}")
    true_labels_list = []
    pred_labels_list = []
    pbar = tqdm(total=len(test_data), desc=f'进度')
    
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i + batch_size]
        dialog_inputs = [item['input'] for item in batch_data]
        true_batch_labels = [item['label'] for item in batch_data]
        
        predictions = predict_batch(model, tokenizer, dialog_inputs, device)
        pred_batch_labels = [prediction['classification'] for prediction in predictions]
        
        true_labels_list.extend(true_batch_labels)
        pred_labels_list.extend(pred_batch_labels)
        
        pbar.update(len(batch_data))
    
    # 计算评估指标
    metrics = calculate_metrics(true_labels_list, pred_labels_list)
    
    return {
        'metrics': metrics
    }

def evaluate_with_model(model, tokenizer, testdata_path, device='cuda', batch=False, debug=False):
    """
    使用给定模型评估测试数据
    
    Args:
        model: 模型
        tokenizer: 分词器
        testdata_path: 测试数据路径
        device: 运行设备
        batch: 是否使用批处理模式
        debug: 是否打印调试信息
        
    Returns:
        评估结果
    """
    dataset = load_json(testdata_path)
    
    if batch:
        # 批处理模式
        return run_test_batch(model, tokenizer, dataset, device=device, debug=debug)
    else:
        # 非批处理模式
        return run_test(model, tokenizer, dataset, device=device, debug=debug)

def evaluate(model_path, checkpoint_path, testdata_path, device='cuda', batch=False, debug=False):
    """
    评估模型在测试数据上的性能
    
    Args:
        model_path: 模型路径
        checkpoint_path: 检查点路径
        testdata_path: 测试数据路径
        device: 运行设备
        batch: 是否使用批处理模式
        debug: 是否打印调试信息
        
    Returns:
        评估结果
    """
    model, tokenizer = load_model(model_path, checkpoint_path, device)
    return evaluate_with_model(model, tokenizer, testdata_path, device, batch, debug) 
