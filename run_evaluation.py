import os
import json
import argparse
from .utils import convert_to_serializable
from .inference import evaluate

def main():
    """
    主函数，解析命令行参数并执行评估
    """
    output_dir = "/home/wangwenhao/WorkSpace/LLaMA-Factory/outputs"
    
    parser = argparse.ArgumentParser(description='评估模型在需求分类任务上的性能')
    
    # parser.add_argument('--model_path', type=str, required=True, 
    #                     help='模型路径')
    # parser.add_argument('--checkpoint_path', type=str, default='', 
    #                     help='模型检查点路径，如有LoRA等微调权重')
    # parser.add_argument('--evaldata_path', type=str, required=True, 
    #                     help='评估数据的路径')
    parser.add_argument('--device', type=str, default='cuda:0', 
                        help='运行设备')
    parser.add_argument('--batch', action='store_true', 
                        help='是否使用批处理模式')
    parser.add_argument('--debug', action='store_true', 
                        help='是否打印调试信息')
    parser.add_argument('--output_file', type=str, default='evaluation_results.json', 
                        help='结果输出文件路径')
    
    args = parser.parse_args()
    
    # # 执行评估
    # evaluation_results = evaluate(
    #     args.model_path, 
    #     args.checkpoint_path, 
    #     args.evaldata_path, 
    #     device=args.device, 
    #     batch=args.batch, 
    #     debug=args.debug
    # )

    evaluation_results = evaluate(
        model_path="/home/wangwenhao/WorkSpace/saves/Llama-8B/lora/emse",
        checkpoint_path='',
        testdata_path="/home/wangwenhao/WorkSpace/LLaMA-Factory/data/split-emse/test.json",
        device="cuda:0",
        batch=True,
        debug=False
)
    
    # sft_promise_results.json
    # zero-shot-promise_results.json
    
    # 处理结果保存路径
    if args.output_file:
        results_file = os.path.join(output_dir, args.output_file)
    else:
        output_dir = os.path.dirname(args.evaldata_path)
        results_file = os.path.join(output_dir, 'evaluation_results.json')
    
    # 将结果序列化为JSON格式
    serializable_results = convert_to_serializable(evaluation_results)
    
    # 保存结果
    with open(results_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n结果已保存至: {results_file}")

if __name__ == "__main__":
    main() 