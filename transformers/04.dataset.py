import datasets as ds
import torch
import os

# 加载本地简单数据集
# path：HF上的在线数据集，或者本地数据类型（csv/json/text/parquet）
# name：有些在线数据集有多个任务，HF 页面有 Configs 就要用
# split：数据集切分。split="train[:1%]“，小样本debug。
# data_files：本地数据集。支持多文件：data_files={"train": "train.json", "validation": "val.json"}
#   也可以直接传入文件夹。
# column_names：csv没有header时自定义用。
# keep_in_memory=True，直接把数据放内存里，只能小数据集用。
# streaming=True，在线大数据集不下载至本地。
dataset1 = ds.load_dataset("csv", data_files="./ChnSentiCorp_htl_all.csv", split="train[:1%]")

# 数据过滤，配合lambda表达式
dataset1 = dataset1.filter(
    lambda x: x["review"] is not None and x["label"] is not None
)

# 一个标准手写 Collator 骨架，实例化之后传入dataloader的collate_fn。
class MyCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        # 1. 解包 batch
        texts = [x["text"] for x in batch]
        labels = [x["label"] for x in batch]

        # 2. batch 级处理
        enc = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            return_tensors="pt"
        )

        # 3. labels 对齐 batch
        enc["labels"] = torch.tensor(labels, dtype=torch.long)

        return enc


# 用HF的datasets自定义数据集
class MyDataset(ds.GeneratorBasedBuilder):

    def _info(self):
        # 在这里定义数据集的信息,这里要对数据的字段进行定义。
        # 要求是人类可读，不需要对齐模型的输入。
        # 例如不同模态的数据如下：
        return ds.DatasetInfo(
            features=ds.Features({
                "image": ds.Image(),
                # "text": ds.Value("string"),
                # "label": ds.ClassLabel(names=["cat", "dog"]),   # 分类标签
                # "audio": ds.Audio(sampling_rate=16000),
                # "video": ds.Video(),
                # "objects": ds.Sequence({
                #     "bbox": ds.Sequence(ds.Value("float32"), length=4), # 语义边缘
                #     "label": ds.ClassLabel(names=["person", "car"])
                # })
            })
        )

    def _split_generators(self, dl_manager):
        # name: 指定数据集的划分
        # gen_kwargs: 指定要读取的文件的相对路径
        data_dir = dl_manager.base_path  # dataset 脚本所在目录
        return [
            ds.SplitGenerator(
                name=ds.Split.TRAIN,
                gen_kwargs={
                    "image_dir": os.path.join(data_dir, "train")
                }
            ),
            ds.SplitGenerator(
                name=ds.Split.VALIDATION,
                gen_kwargs={
                    "image_dir": os.path.join(data_dir, "val")
                }
            ),
        ]

    def _generate_examples(self, image_dir):
        # 逐个生成数据样本，需要自己临场写
        # 生成的每一条样本，字段必须是 _info().features 的子集，或完全一致
        idx = 0
        for fname in os.listdir(image_dir):
            if not fname.lower().endswith((".jpg", ".png", ".jpeg")):
                continue

            path = os.path.join(image_dir, fname)

            yield idx, {
                "image": path  # 或 Image.open(path)
            }
            idx += 1


# 加载本地复杂数据集（用类写的）
dataset2 = ds.load_dataset("mydata", split="train[:1%]")
