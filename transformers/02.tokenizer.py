from transformers import AutoTokenizer
from transformers import GPT2TokenizerFast

# # 加载训练好的tokenizer
# tokenizer = AutoTokenizer.from_pretrained("uer/roberta-base-finetuned-dianping-chinese")
# # tokenizer 保存到本地，并从本地加载tokenizer的方法
# tokenizer.save_pretrained("./roberta_tokenizer")
# tokenizer = AutoTokenizer.from_pretrained("./roberta_tokenizer/")

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

tokenizer.add_special_tokens({'pad_token': '[PAD]'})

sens = ["弱小的我也有大梦想",
        "有梦想谁都了不起",
        "追逐梦想的心，比梦想本身，更可贵"]

# 常用参数：
# add_special_tokens参数是句子的开始和结束位置的提示词，具体看模型的设计。
# max_length是句子的最大长度，一般结合truncation=True使用
# truncation=True启用序列截断，如果文本长度超过 max_length，自动截断
# offset_mapping 会返回 每个 token 在原始字符串中的起始和结束字符位置。在英语中体现比较重要
res = tokenizer(sens, add_special_tokens=True, padding="max_length", max_length=15, truncation=True)
print(tokenizer)
print(res)
