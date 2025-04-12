from pathlib import Path
BASE_DIR = Path('pybert')
config = {
    'raw_data_path': BASE_DIR/r'C:\Users\wyf\Desktop\research\GPT-BERT\GPT-BERT-个人实验\pybert\dataset\emse.csv',
    'test_path': BASE_DIR/r'dataset/mnr_test.csv',
    'aug_data_path': BASE_DIR/r'dataset/TTA_Per.csv',
    'aug_test_path': BASE_DIR/r'dataset/TTA_Test.csv',

    'data_dir': BASE_DIR/r"dataset",
    'log_dir': BASE_DIR/r'output/log',
    'writer_dir': BASE_DIR/r"output/TSboard",
    'figure_dir': BASE_DIR/r"output/figure",
    'checkpoint_dir': BASE_DIR/r"output/checkpoints",
    'cache_dir': BASE_DIR /r'model/',
    'result': BASE_DIR /r"output/result",

    'bert_vocab_path': BASE_DIR/r'pretrain/bert/base-uncased/vocab.txt',
    'bert_config_file': BASE_DIR/r'pretrain/bert/base-uncased/config.json',
    'bert_model_dir': BASE_DIR/r'pretrain/bert/base-uncased',

    'xlnet_vocab_path': BASE_DIR / 'pretrain/xlnet/base-cased/spiece.model',
    'xlnet_config_file': BASE_DIR / 'pretrain/xlnet/base-cased/config.json',
    'xlnet_model_dir': BASE_DIR / 'pretrain/xlnet/base-cased',

    'albert_vocab_path': BASE_DIR / 'pretrain/albert/albert-base/30k-clean.model',
    'albert_config_file': BASE_DIR / 'pretrain/albert/albert-base/config.json',
    'albert_model_dir': BASE_DIR / 'pretrain/albert/albert-base',
    
    'glove_path': BASE_DIR / 'pretrain/glove/glove.6B.300d.txt',
    'glove_model_dir': BASE_DIR / 'pretrain/glove',
    'embedding_dim': 300,  # GloVe维度
    'hidden_size': 512,  # BiLSTM隐藏层大小
    'num_layers': 2,  # BiLSTM层数
}
