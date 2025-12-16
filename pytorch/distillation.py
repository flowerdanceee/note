# 不同场景下如何蒸馏：
# 只想快速压模型：CE + KD
# student 很浅：CE + KD + hidden
# 没标签：      KD + hidden
# 表征迁移：    hidden 为主
import torch.nn.functional as F
import torch

# 超参数设置：
# alpha (KD)     = 0.5 ~ 0.9
# beta (hidden)  = 0.1 ~ 0.5
# T              = 2 ~ 4
alpha = 0.7
beta = 0.3
T = 2.0

# 前提：已经有了 logits
student_logits = student(input_ids)  # [B, C] or [B, T, C]
teacher_logits = teacher(input_ids)  # 同 shape
labels = labels  # [B] or [B, T]

# 最普通的分类任务中的交叉熵
ce_loss = F.cross_entropy(
    student_logits.view(-1, student_logits.size(-1)),
    labels.view(-1)
)

# KL Distillation 损失

with torch.no_grad():
    teacher_probs = F.softmax(teacher_logits / T, dim=-1)
student_log_probs = F.log_softmax(student_logits / T, dim=-1)
kd_loss = F.kl_div(
    student_log_probs,
    teacher_probs,
    # KL 是分布 → batchmean
    reduction="batchmean") * (T * T)

# hidden MSE loss
# 可以每 N step 算一次 hidden MSE， 其余 step 只算 KL
hidden_loss = 0.0
num_layers = len(student_hidden_states)

for i in range(num_layers):
    s_h = student_hidden_states[i]
    t_h = teacher_hidden_states[2 * i + 1]  # 每2层对1层
    # 向量对齐 → mean
    hidden_loss += F.mse_loss(s_h, t_h, reduction="mean")

hidden_loss = hidden_loss / num_layers

# 蒸馏损失
loss = alpha * kd_loss + beta * hidden_loss + ce_loss
# 如果梯度爆炸就上梯度裁剪


