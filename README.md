# Math-Solver-FDU-AI

基于通义千问 Qwen3-0.6B 模型微调的小学数学题解答系统，支持对一至六年级题目的自动解题与解析生成。项目结合强化学习（GRPO）进行优化，采用自定义奖励函数引导模型输出更具逻辑性的思维链（Chain-of-Thought）。

## 项目亮点

- 基于 Qwen3-0.6B 模型进行全量微调
- 集成 GRPO（Group Relative Policy Optimization）强化学习算法
- 自定义奖励函数：结合语义相似度（CoT）和答案正确性
- 提供可视化训练指标，支持 reward 收敛情况追踪与评估
- 支持小学阶段多类型数学题目的自动推理与结构化解题

## 模型与训练流程

1. **基础模型**：使用 [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) 作为初始模型  
2. **数据构造**：采集并格式化一至六年级小学数学题，包括题干、参考答案与思维链  
3. **奖励设计**：
   - 语义相似度（CoT Similarity）：采用 sentence-transformers 计算模型解题思路与参考解的相似度
   - 答案匹配：使用字符串比较与规则模板判断模型答案是否与参考答案一致
   - 综合上述两项，按设定权重加权生成 reward 分值
4. **训练方式**：通过 GRPO 算法结合奖励值对模型进行优化训练

## 技术栈

- 模型：Qwen3ForCausalLM（transformers）
- 强化学习：GRPO 自定义策略优化器
- 评估：sentence-transformers, matplotlib, tqdm
- 环境管理：conda + PyTorch 2.x + CUDA

