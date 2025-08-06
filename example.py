"""
使用拆分后的代码示例
"""
import os
import json
from utils import load_model, convert_to_serializable
from inference import evaluate

# 参数配置
device = 'cuda:0'
evaldata_path = '/home/wangwenhao/WorkSpace/LLaMA-Factory/data/split/test.json'
model_path = '/home/wangwenhao/WorkSpace/Llama-3.2-3B-Instruct'
checkpoint_path = '/home/wangwenhao/WorkSpace/saves/LLaMA3-3B/lora/sft/checkpoint-170'

# 执行评估
evaluation_results = evaluate(model_path, checkpoint_path, evaldata_path, device=device, batch=True, debug=False)

# 将评估结果保存到文件
output_dir = os.path.dirname(evaldata_path)
results_file = os.path.join(output_dir, 'evaluation_results.json')

# 将结果序列化为JSON格式
serializable_results = convert_to_serializable(evaluation_results)

# 保存结果
with open(results_file, 'w', encoding='utf-8') as f:
    json.dump(serializable_results, f, ensure_ascii=False, indent=2)

print(f"\n结果已保存至: {results_file}")

# 也可以使用更低层级的API
if __name__ == "__main__":
    # 示例：使用较低层级API进行自定义评估
    from utils import load_json
    from inference import predict, predict_batch
    from metrics import calculate_metrics
    
    # 加载模型和数据
    model, tokenizer = load_model(model_path, checkpoint_path, device)
    test_data = load_json(evaldata_path)
    
    # 单个样本预测示例
    sample = test_data[0]
    result = predict(model, tokenizer, sample['input'], device)
    print(f"样本: {sample['input'][:50]}...")
    print(f"真实标签: {sample['label']}")
    print(f"预测标签: {result.get('classification', [])}")
    
    # 批量预测示例
    batch = [item['input'] for item in test_data[:5]]
    batch_results = predict_batch(model, tokenizer, batch, device)
    for i, (text, result) in enumerate(zip(batch, batch_results)):
        print(f"批处理样本 {i+1}: {text[:30]}... => {result.get('classification', [])}") 