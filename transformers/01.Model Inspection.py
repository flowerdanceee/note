from transformers import AutoModel
import inspect

model = AutoModel.from_pretrained("hfl/rbt3")

# 在这检查模型，跑一次forward，然后问自己如下问题：
# 1.input的字典中，哪个key的第 0 维表示着batch？
# 2.最终输出的张量维度是什么？
# 3.模型需不需要传入labels？
# 4.打算怎么改config？
print(inspect.signature(model.forward))

