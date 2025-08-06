import json
import random
import os
from sklearn.model_selection import train_test_split

# 定义文件路径
input_file = '/home/wangwenhao/WorkSpace/LLaMA-Factory/data/combined_data.json'
output_dir = '/home/wangwenhao/WorkSpace/LLaMA-Factory/data/split-emse'

# 创建输出目录（如果不存在）
os.makedirs(output_dir, exist_ok=True)

# 加载JSON数据
def load_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

# 保存JSON数据
def save_json(data, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

# 主函数
def main():
    # 加载数据
    print(f"Loading data from {input_file}...")
    data = load_json(input_file)
    
    # 设置随机种子确保可重复性
    random.seed(42)
    
    # 首先分出训练集 (80%)
    train_data, val_test_data = train_test_split(
        data, 
        train_size=0.8, 
        random_state=42
    )
    
    # 然后从剩余数据中分出验证集和测试集 (各占总数据的10%)
    val_data, test_data = train_test_split(
        val_test_data, 
        test_size=0.5, 
        random_state=42
    )
    
    # 输出分割结果
    print(f"Split complete: Train {len(train_data)} ({len(train_data)/len(data)*100:.2f}%), "
          f"Validation {len(val_data)} ({len(val_data)/len(data)*100:.2f}%), "
          f"Test {len(test_data)} ({len(test_data)/len(data)*100:.2f}%)")
    
    # 保存分割后的数据集
    train_file = os.path.join(output_dir, 'train.json')
    val_file = os.path.join(output_dir, 'val.json')
    test_file = os.path.join(output_dir, 'test.json')
    
    save_json(train_data, train_file)
    save_json(val_data, val_file)
    save_json(test_data, test_file)
    
    print(f"Data saved to:")
    print(f"  Train: {train_file}")
    print(f"  Validation: {val_file}")
    print(f"  Test: {test_file}")
    
    # 检查分割后各子集的类别分布
    print("\nClass distribution in splits:")
    for name, dataset in [("Train", train_data), ("Validation", val_data), ("Test", test_data)]:
        split_labels = {}
        for item in dataset:
            if 'label' in item and isinstance(item['label'], list):
                for label in item['label']:
                    if label not in split_labels:
                        split_labels[label] = 0
                    split_labels[label] += 1
            elif 'label' in item:
                label = item['label']
                if label not in split_labels:
                    split_labels[label] = 0
                split_labels[label] += 1
        
        print(f"\n{name} set:")
        for label, count in split_labels.items():
            print(f"  {label}: {count}")

if __name__ == "__main__":
    main()