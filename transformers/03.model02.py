from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
from torch.utils.data import Dataset, DataLoader, random_split
import torch
import pandas as pd
from torch.optim import Adam

tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
device = torch.device('mps')
model = AutoModelForSequenceClassification.from_pretrained("hfl/rbt3").to(device)


class MyDataset(Dataset):

    def __init__(self) -> None:
        super().__init__()
        self.data = pd.read_csv("./ChnSentiCorp_htl_all.csv")
        # dropna()去掉空数据
        self.data = self.data.dropna()

    def __getitem__(self, index):
        return self.data.iloc[index]["review"], self.data.iloc[index]["label"]

    def __len__(self):
        return len(self.data)


dataset = MyDataset()
# 划分数据集
trainset, validset = random_split(dataset, lengths=[0.9, 0.1])


def collate_func(batch):
    texts, labels = [], []
    for item in batch:
        texts.append(item[0])
        labels.append(item[1])
    inputs = tokenizer(texts, max_length=128, padding="max_length", truncation=True, return_tensors="pt")
    inputs["labels"] = torch.tensor(labels, dtype=torch.long)
    return inputs


# Dataloader中的collate_fn参数是batch级加工，Dataset中使用transform函数的样本加工不同。
# 由于tokenizer可以进行batch处理，所以放在dataloader。
trainloader = DataLoader(trainset, batch_size=32, shuffle=True, collate_fn=collate_func)
validloader = DataLoader(validset, batch_size=64, shuffle=False, collate_fn=collate_func)

optimizer = Adam(model.parameters(), lr=2e-5)


def evaluate():
    model.eval()
    acc_num = 0
    with torch.inference_mode():
        for batch in validloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            output = model(**batch)
            pred = torch.argmax(output.logits, dim=-1)
            acc_num += (pred.long() == batch["labels"].long()).float().sum()
    return acc_num / len(validset)


def train(epoch=3, log_step=100):
    global_step = 0
    model.to(device)
    for ep in range(epoch):
        model.train()
        for batch in trainloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()
            output = model(**batch)
            output.loss.backward()
            optimizer.step()
            if global_step % log_step == 0:
                print(f"ep: {ep}, global_step: {global_step}, loss: {output.loss.item()}")
            global_step += 1
        acc = evaluate()
        print(f"ep: {ep}, acc: {acc}")


train()

# 预测
sen = "我觉得这家酒店不错，饭很好吃！"
id2_label = {0: "差评！", 1: "好评！"}
model.config.id2label = id2_label
pipe = pipeline("text-classification", model=model, tokenizer=tokenizer, device=-1)
print(pipe(sen))
