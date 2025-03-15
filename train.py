from nlp2 import set_seed

from code_qwen import QWENSeq2Seq
import pandas as pd
from sklearn.model_selection import train_test_split
import json
set_seed(42)
# # 指定CSV文件的路径
# csv_file_path = '../dataset/buffer/buffer_detect_mod_slice.csv'
#
# # 读取数据
# data = pd.read_csv(csv_file_path)
#
# # 使用 sklearn 的 train_test_split 按照 8:1:1 划分
# train_data, temp_data = train_test_split(data, test_size=0.2, random_state=42)  # 先取 20% 的数据用于验证和测试
# val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)  # 再将 20% 分为验证和测试各 10%
#
# # 打印数据集大小
# print(f"Total data: {len(data)}")
# print(f"Training data: {len(train_data)}")
# print(f"Validation data: {len(val_data)}")
# print(f"Test data: {len(test_data)}")
#
# # 保存划分后的数据集到文件
# train_data.to_csv('../dataset/buffer/detect_train.csv', index=False)
# val_data.to_csv('../dataset/buffer/detect_val.csv', index=False)
# test_data.to_csv('../dataset/buffer/detect_test.csv', index=False)
#
# print("Data split completed and saved to files!")

model = QWENSeq2Seq(base_model_path="../model/Deepseek-Coder-1.3B",
                    add_eos_token=False,
                    adapter="lora",
                    load_adapter_path="None",
                    source_len=512,
                     cutoff_len=256)

model.train(train_filename="../dataset/buffer/buffer_detect_train_add_mod_blank_sliced_deepseek.csv",
            train_batch_size=4,
            learning_rate=2e-5,
            num_train_epochs=25,
            early_stop=3,
            do_eval=True,
            eval_filename="",
            eval_batch_size=4, output_dir='save_model/', do_eval_bleu=True)