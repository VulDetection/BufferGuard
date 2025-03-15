import torch
from torch.utils.data import Dataset
import pandas as pd
device = "cuda" if torch.cuda.is_available() else "cpu"
from tqdm import tqdm
import copy


label_map = {1: "vulnerable", 0: "safe"}


class GPTDatasetForSequenceClassification(Dataset):
    def __init__(self, datafile, tokenizer, source_len, cutoff_len, label_map=None):
        """
        用于分类任务的自定义数据集
        Args:
            datafile (str): 数据文件路径，支持 CSV 格式。
            tokenizer: 分词器（如 Hugging Face Tokenizer）。
            max_length (int): 文本的最大长度。
            label_map (dict): 标签映射字典，将字符串标签映射到整数（如 {"negative": 0, "neutral": 1, "positive": 2}）。
        """
        self.max_length = source_len
        self.tokenizer = tokenizer
        self.label_map = label_map

        self.texts = []
        self.des = []
        self.labels = []
        self.api = []
        self.boundary = []
        self.match = []

        # 加载数据文件
        data = pd.read_csv(datafile)

        for idx in tqdm(range(len(data))):
            text = data["slice_code"][idx]  # 假设数据集有 "text" 列
            des = data["description"][idx]
            api = data["API"][idx]
            boundary = data["boundary"][idx]
            match = data["match"][idx]
            text = prompt_pre(src=text, des=des, api=api, boundary=boundary, match=match)
            label = data["label"][idx]  # 假设数据集有 "label" 列

            # 映射标签为整数
            if self.label_map:
                label = self.label_map[label]

            self.texts.append(text)
            self.labels.append(label)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        """
        返回单条样本的输入和标签张量。
        Args:
            idx (int): 样本索引。
        Returns:
            dict: 包含 `input_ids`, `attention_mask`, `label` 的字典。
        """
        text = self.texts[idx]
        label = self.labels[idx]

        # 对文本进行分词
        tokenized = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding="max_length",
            return_tensors="pt",
        )

        input_ids = tokenized["input_ids"].squeeze(0)  # 去掉多余维度
        attention_mask = tokenized["attention_mask"].squeeze(0)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "label": torch.tensor(label, dtype=torch.long),
        }