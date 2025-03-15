import math
import os
import sys
import csv
import pickle
import matplotlib.pyplot as plt
# 假设您已经有了一个用于存储嵌入向量的字典
class_embeddings = {0: [], 1: []}
import pandas as pd
import numpy as np
import torch
from peft import TaskType, LoraConfig, AdaLoraConfig, PrefixTuningConfig, \
    PromptEncoderConfig, PromptTuningConfig, get_peft_model, PeftModel
from torch import nn
from torch.utils.data import RandomSampler, DataLoader, SequentialSampler
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, Qwen2Tokenizer
from dataset import GPTDatasetForSequenceClassification, cot_prompt_pre
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score,roc_auc_score, matthews_corrcoef
device = "cuda" if torch.cuda.is_available() else "cpu"

a = 0.05
b = 3
savefile = 'slice_code+des+api+bound+match+diffloss(0.05_3)_save'
# savefile = 'slice_code+des+api+bound+match'
def find_all_linear_names(model):
    """
    找出所有全连接层，为所有全连接添加adapter
    """
    # cls = bnb.nn.Linear4bit
    cls = nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names:  # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)



class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)).to(device)
        self.weights = torch.ones(num_classes).to(device)  # 初始化权重为1
        self.weights[1] = b  # 增加标签为1的样本的权重

    def forward(self, x, labels):
        # 计算特征与其对应类别中心的距离
        diff = x - self.centers.index_select(0, labels)

        # 计算加权损失
        weighted_diff = diff * self.weights[labels].view(-1, 1)
        loss = (weighted_diff ** 2).sum(dim=1).mean()

        return loss
class CenterLossMean(nn.Module):
    def __init__(self, num_classes, feat_dim, device):
        super(CenterLoss, self).__init__()
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim)).to(device)

    def forward(self, x, labels):
        # 计算特征与其对应类别中心的距离
        diff = x - self.centers.index_select(0, labels)
        loss = (diff ** 2).sum(dim=1).mean()
        return loss
# 训练结束后，绘制散点图
def plot_embeddings(class_embeddings, epoch):
    # 仅在绘图时将列表转换为numpy数组
    class_0_embeddings = np.array(class_embeddings[0])
    class_1_embeddings = np.array(class_embeddings[1])

    plt.figure(figsize=(10, 10))
    plt.scatter(class_0_embeddings[:, 0], class_0_embeddings[:, 1], c='blue', label='Class 0')
    plt.scatter(class_1_embeddings[:, 0], class_1_embeddings[:, 1], c='red', label='Class 1')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    plt.title(f'Scatter Plot with Center Loss - Epoch {epoch}')
    # 存储图表为 PDF 文件
    plt.savefig(f'result/DeepSeek-Coder-1.3b/gpt/{savefile}/fig/fig_{epoch}.pdf', bbox_inches='tight')
    plt.close()  # 关闭图形，避免显示
class QWENSeq2Seq():

    def __init__(self, base_model_path, add_eos_token=False, adapter="lora", load_adapter_path="None", source_len=300, cutoff_len=512):
        print("***** CUDA.empty_cache() *****")
        torch.cuda.empty_cache()
        self.base_model = base_model_path
        self.add_eos_token = add_eos_token
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.adapter = adapter
        self.load_adapter_path = load_adapter_path
        self.cutoff_len = cutoff_len
        self.source_len = source_len
        # 初始化LLM模型
        self.model, self.tokenizer = self.get_model_tokenizer()
        self.center_loss = CenterLoss(num_classes=2, feat_dim=self.model.config.hidden_size, device=self.device)
        self.inverse_label_map = {"vulnerable": 1, "safe": 0}

        # 初始化adapter
        if self.load_adapter_path == "None":
            self.model = self.load_adapter_config(self.model)

        # 加载训练好的adapter
        if self.load_adapter_path != "None":
            self.model = PeftModel.from_pretrained(
                self.model,
                self.load_adapter_path
            )

        if torch.__version__ >= "2" and sys.platform != "win32":
            self.model = torch.compile(self.model)

        self.model.to(self.device)

    def get_model_tokenizer(self):

        model = AutoModelForSequenceClassification.from_pretrained(
            self.base_model,
            # quantization_config=q_config,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            num_labels=2,
            output_attentions=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(
            self.base_model,
            trust_remote_code=True,
            add_eos_token=self.add_eos_token,
            pad_token = '<PAD>'
        )  # default add_eos_token=False
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

    def load_adapter_config(self, model):
        t_type = TaskType.SEQ_CLS

        if self.adapter == "lora":
            print(model)
            target_modules = find_all_linear_names(model)
            print(target_modules)
            config = LoraConfig(
                task_type=t_type,
                inference_mode=False,
                lora_dropout=0.05,
                r=8,
                lora_alpha=16,
                target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                # target_modules = ["c_proj","c_attn","c_fc"]
            )
        elif self.adapter == 'adalora':
            config = AdaLoraConfig(
                task_type=t_type,
                inference_mode=False,
            )
        elif self.adapter == "prefix":
            config = PrefixTuningConfig(
                task_type=t_type,
                prefix_projection=True
            )
        elif self.adapter == "p_tuning":
            config = PromptEncoderConfig(
                task_type=t_type
            )
        elif self.adapter == "prompt":
            config = PromptTuningConfig(
                task_type=t_type
            )
        else:
            raise KeyError("Unknow adapter: {}".format(self.adapter))

        model = get_peft_model(model, config)
        model.print_trainable_parameters()

        return model

    def train(self, train_filename, train_batch_size, learning_rate, num_train_epochs, early_stop,
              do_eval, eval_filename, eval_batch_size, output_dir, do_eval_bleu):

        train_data = GPTDatasetForSequenceClassification(train_filename, tokenizer=self.tokenizer, source_len=self.source_len, cutoff_len=self.cutoff_len)
        print(train_data)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler,
                                      batch_size=train_batch_size)

        # Prepare optimizer and schedule (linear warmup and decay)
        t_total = len(train_dataloader) // num_train_epochs
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=learning_rate)
        num_train_optimization_steps = num_train_epochs * len(train_dataloader)
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=int(t_total * 0.1),
                                                    num_training_steps=num_train_optimization_steps)

        # Start training
        train_example_num = len(train_data)
        print("***** Running training *****")
        print("  Num examples = %d", train_example_num)
        print("  Batch size = %d", train_batch_size)
        print("  Batch num = %d", math.ceil(train_example_num / train_batch_size))
        print("  Num epoch = %d", num_train_epochs)

        global_step, best_bleu, best_loss = 0, -1, 1e6
        count = 0

        for cur_epoch in range(int(num_train_epochs)):
            bar = tqdm(train_dataloader, total=len(train_dataloader), desc="Training")
            nb_tr_examples, nb_tr_steps, tr_loss = 0, 0, 0
            self.model.train()
            for step, batch in enumerate(bar):
                # 取出 batch 数据
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["label"].to(device)
                # 梯度清零
                optimizer.zero_grad()
                # 前向传播，计算损失
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, output_attentions=True, output_hidden_states=True)
                last_hidden_state = outputs.hidden_states[-1] # 获取嵌入
                loss = outputs.loss  # 分类任务的损失

                # 计算中心损失
                cls_embeddings = last_hidden_state[:, 0, :]  # 获取CLS标记的嵌入
                center_loss = self.center_loss(cls_embeddings, labels)/train_example_num
                loss = loss + a * center_loss  # 将中心损失加到原始损失上

                # 画图
                cls_embeddings = last_hidden_state[:, 0, :]  # 获取CLS标记的嵌入
                # center_loss = self.center_loss(cls_embeddings, labels)
                # total_loss = loss + center_loss  # 将中心损失加到原始损失上

                # 收集嵌入向量
                for i, label in enumerate(labels.cpu()):
                    # 将张量转换为float32类型，然后再转换为NumPy数组
                    class_embeddings[label.item()].append(cls_embeddings[i].detach().cpu().to(torch.float32).numpy())

                tr_loss += loss.item()
                nb_tr_steps += 1

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()
                scheduler.step()

                global_step += 1
                train_loss = round(tr_loss * 1 / (nb_tr_steps + 1), 4)
                bar.set_description("[{}] Train loss {}".format(cur_epoch, round(train_loss, 3)))
            # # 将列表转换为numpy数组
            # class_embeddings[0] = np.array(class_embeddings[0])
            # class_embeddings[1] = np.array(class_embeddings[1])


            if do_eval:
                # Eval model with dev dataset
                eval_data = GPTDatasetForSequenceClassification(eval_filename, tokenizer=self.tokenizer, source_len=self.source_len, cutoff_len=self.cutoff_len)
                eval_sampler = SequentialSampler(eval_data)
                eval_dataloader = DataLoader(eval_data, sampler=eval_sampler, batch_size=eval_batch_size)

                print("***** Running evaluation  *****")
                print("  Num examples = %d", eval_data.__len__())
                print("  Batch size = %d", eval_batch_size)
                print("  Num epoch = %d", cur_epoch)
                self.model.eval()
                all_preds = []
                all_labels = []
                eval_loss, batch_num = 0, 0
                for batch in tqdm(eval_dataloader, desc="Evaluating"):
                    input_ids = batch["input_ids"].to(device)
                    attention_mask = batch["attention_mask"].to(device)
                    labels = batch["label"].to(device)

                    with torch.no_grad():
                        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                        loss = outputs.loss
                        logits = outputs.logits

                    eval_loss += loss.item()
                    batch_num += 1

                    # 获取预测结果
                    preds = torch.argmax(logits, dim=-1).cpu().numpy()
                    all_preds.extend(preds)
                    all_labels.extend(labels.cpu().numpy())
                    # 打开一个新的 CSV 文件用于写入
                    with open('result/DeepSeek-Coder-1.3b/gpt/{}/pred/pred_{}.csv'.format(savefile, cur_epoch), mode='w', newline='', encoding='utf-8') as file:
                        writer = csv.writer(file)
                        # 写入列名
                        writer.writerow(['pred', 'label'])
                        # 写入预测结果和标签
                        for pred, label in zip(all_preds, all_labels):
                            writer.writerow([pred, label])
                self.model.train()
                eval_loss = eval_loss / batch_num
                result = {'eval_loss': round(eval_loss, 5),
                          'global_step': global_step + 1,
                          'train_loss': round(train_loss, 5)}
                for key in sorted(result.keys()):
                    print("  %s = %s", key, str(result[key]))
                print("  " + "*" * 20)

                # 计算评估指标
                accuracy = accuracy_score(all_labels, all_preds)
                precision = precision_score(all_labels, all_preds)
                recall = recall_score(all_labels, all_preds)
                try:
                    f1 = 2 * precision * recall / (precision + recall)
                except:
                    f1 = 0.0
                # f1 = f1_score(all_labels, all_preds, average="weighted")
                auc = roc_auc_score(all_labels, all_preds)
                mcc = matthews_corrcoef(all_labels, all_preds)
                # 打印评估结果
                print(f"Eval loss: {eval_loss:.4f}")
                print(f"Accuracy: {accuracy:.4f}")
                print(f"Precision: {precision:.4f}")
                print(f"Recall: {recall:.4f}")
                print(f"F1 Score: {f1:.4f}")
                print(f"AUC: {auc:.4f}")
                print(f"MCC: {mcc:.4f}")

                print("***** CUDA.empty_cache() *****")
                torch.cuda.empty_cache()
            # # 绘制散点图
            # plt.figure(figsize=(10, 10))
            # plt.scatter(class_embeddings[0][:, 0], class_embeddings[0][:, 1], c='blue', label='Class 0')
            # plt.scatter(class_embeddings[1][:, 0], class_embeddings[1][:, 1], c='red', label='Class 1')
            # plt.xlabel('Feature 1')
            # plt.ylabel('Feature 2')
            # plt.legend()
            # plt.title('Scatter Plot with Center Loss')
            # # 存储图表为 PDF 文件
            # plt.savefig('result/Qwen2.5-coder-1.5b/code/fig/fig_{}.pdf'.format(cur_epoch), bbox_inches='tight')
            # # plt.show()
            plot_embeddings(class_embeddings, cur_epoch)
        with open('result/DeepSeek-Coder-1.3b/gpt/model/best_model.pkl','wb') as f:
            pickle.dump(self.model, f)

    def test(self, filename, output_dir, decoding='greedy'):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        hyp_list = []
        datas = pd.read_csv(filename)
        ref_list = datas['tgt'].tolist()
        src_list = datas['src'].tolist()

        for i in tqdm(range(len(src_list))):
            src = src_list[i]
            hyp_list.append(self.predict(src, decoding))

        assert len(ref_list) == len(hyp_list)
        df = pd.DataFrame(ref_list)
        df.to_csv(output_dir + "/gold.csv", index=False, header=None)
        df = pd.DataFrame(hyp_list)
        df.to_csv(output_dir + "/codellama.csv", index=False, header=None)
        score = nlgeval.compute_metrics(ref_list=[ref_list], hyp_list=hyp_list)
        print(score)

    def predict(self, src, decoding='greedy'):
        src = cot_prompt_pre(src)
        encoding = self.tokenizer([src], return_tensors="pt", truncation=True, max_length=256).to(self.device)

        with torch.no_grad():
            if decoding == 'greedy':
                gen_tokens = self.model.generate(**encoding,
                                                 do_sample=False,
                                                 num_beams=1,
                                                 temperature=0.2,
                                                 max_new_tokens=256,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.eos_token_id,
                                                 top_p=0.95)
            elif decoding == 'beam':
                gen_tokens = self.model.generate(**encoding,
                                                 do_sample=False,
                                                 num_beams=5,
                                                 temperature=0.2,
                                                 max_new_tokens=256,
                                                 num_return_sequences=1,
                                                 eos_token_id=self.tokenizer.eos_token_id,
                                                 top_p=0.95)
            elif decoding == 'multinomial':
                gen_tokens = self.model.generate(**encoding,
                                        do_sample=True,
                                        num_beams=1,
                                        temperature=0.2,
                                        max_new_tokens=256,
                                        num_return_sequences=1,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        top_p=0.95)
            elif decoding == 'contrastive':
                gen_tokens = self.model.generate(**encoding,
                                        penalty_alpha=0.6,
                                        top_k=4,
                                        temperature=0.2,
                                        max_new_tokens=256,
                                        num_return_sequences=1,
                                        eos_token_id=self.tokenizer.eos_token_id,
                                        top_p=0.95)
        gen_tokens = gen_tokens[:, encoding['input_ids'].shape[-1]:]
        gen_seqs = self.tokenizer.batch_decode(gen_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        completion_seqs = []
        for gen_seq in gen_seqs:
            if self.tokenizer.eos_token in gen_seq:
                gen_seq = gen_seq[:gen_seq.index(self.tokenizer.eos_token)]
            completion_seqs.append(gen_seq)
        return completion_seqs[0]

