from transformers import AutoConfig, AutoModel, AutoTokenizer

model = AutoModel.from_pretrained("hfl/rbt3", force_download=True)
config = AutoConfig.from_pretrained("hfl/rbt3")

# config       → 控制模型结构（层数、维度、头数）
# | 参数                       | 用途               | 工业场景             |
# | ------------------------- | ----------------- | ---------------- |
# | `hidden_size`             | Transformer 隐藏层维度 | 小显存训练/裁剪模型       |
# | `num_hidden_layers`       | Transformer 层数    | 小模型/加速训练         |
# | `num_attention_heads`     | 注意力头数             | 调整性能/显存          |
# | `intermediate_size`       | FFN 内部维度          | 控制容量/显存          |
# | `max_position_embeddings` | 最大序列长度            | OCR/长文本任务，扩展输入长度 |
# | `initializer_range`       | 权重初始化范围           | 微调时控制稳定性         |
# 工业实践中：很少大幅改，通常只是 max_position_embeddings 或少量裁剪层数/hidden_size。


# model.config → 控制任务行为（输入输出、标签、生成策略）
# | 参数                                               | 用途                | 工业场景                 |
# | ------------------------------------------------ | ----------------- | -------------------- |
# | `num_labels`                                     | 分类任务类别数           | 下游分类任务               |
# | `id2label` / `label2id`                          | 标签映射              | 分类任务标签对应             |
# | `pad_token_id` / `bos_token_id` / `eos_token_id` | 输入/输出 token 设置    | 微调 & 推理              |
# | `hidden_dropout_prob`                            | Dropout 概率        | 微调防过拟合               |
# | `attention_probs_dropout_prob`                   | Attention Dropout | 微调防过拟合               |
# | `max_position_embeddings`（一般不减小）                 | 输入序列最大长度          | 推理/生成任务              |
# | `vocab_size`                                     | 词表大小（可选）          | 自定义 tokenizer 或特殊字符集 |
# 工业实践中：几乎每个微调项目都会改 num_labels、pad_token_id、dropout，生成任务会改 max_length/bos/eos。
print(config,model.config)

sen = "弱小的我也有大梦想！"
tokenizer = AutoTokenizer.from_pretrained("hfl/rbt3")
# tokenizer的最大长度不能超过模型的max_position_embeddings
inputs = tokenizer(sen, return_tensors="pt", add_special_tokens=True, padding="max_length", max_length=12, truncation=True)
# 输出维度[batch, token, output_dim]
print(model(**inputs)['last_hidden_state'].shape)