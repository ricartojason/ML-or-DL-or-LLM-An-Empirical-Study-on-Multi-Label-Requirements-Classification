## Bert multi-label text classification by PyTorch

This repo contains a PyTorch implementation of the pretrained BERT and glove model for multi-label text classification.

###  Structure of the code

At the root of the project, you will see:

```text
├── pybert
|  └── callback
|  |  └── lrscheduler.py　　
|  |  └── trainingmonitor.py　
|  |  └── ...
|  └── config
|  |  └── basic_config.py #a configuration file for storing model parameters
|  └── dataset　　　
|  └── io　　　　
|  |  └── dataset.py　　
|  |  └── data_transformer.py　　
|  └── model
|  |  └── nn　
|  |  └── pretrain　
|  └── output #save the ouput of model
|  └── preprocessing #text preprocessing 
|  └── train #used for training a model
|  |  └── trainer.py 
|  |  └── ...
|  └── common # a set of utility functions
├── run_bert.py
```
### Dependencies

- csv
- tqdm
- numpy
- pickle
- scikit-learn
- PyTorch 1.1+
- matplotlib
- pandas
- transformers=2.5.1

### How to use the code

you need download pretrained bert model and xlnet model.

<div class="note info"><p> BERT:  bert-base-uncased</p></div>

1. Download the Bert pretrained model from [s3](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-pytorch_model.bin) 
2. Download the Bert config file from [s3](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-config.json) 
3. Download the Bert vocab file from [s3](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased-vocab.txt) 
4. Rename:

    - `bert-base-uncased-pytorch_model.bin` to `pytorch_model.bin`
    - `bert-base-uncased-config.json` to `config.json`
    - `bert-base-uncased-vocab.txt` to `bert_vocab.txt`
5. Place `model` ,`config` and `vocab` file into  the `/pybert/pretrain/bert/base-uncased` directory.
6. `pip install pytorch-transformers` from [github](https://github.com/huggingface/pytorch-transformers).
7. Put emse.csvs in `pybert/dataset`.
    -  you can modify the `io.task_data.py` to adapt your data.
8. Modify configuration information in `pybert/configs/basic_config.py`(the path of data,...).
9. Run `python run_bert.py --do_data` to preprocess data.
10. Run `python run_bert.py --do_train --save_best --do_lower_case"` to fine tuning bert model.
11. Run `python run_bert.py --do_test --do_lower_case"` to predict new data.


### training 

```text
[training] 8511/8511 [>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>] -0.8s/step- loss: 0.0640
training result:
[2019-01-14 04:01:05]: bert-multi-label trainer.py[line:176] INFO  
Epoch: 2 - loss: 0.0338 - val_loss: 0.0373 - val_auc: 0.9922
```



