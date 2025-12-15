
# 工作记忆（Working Memory）的检索逻辑

这段代码实现了 **工作记忆（Working Memory）的检索逻辑**。它的核心目标是从当前活动的短期记忆中，找出与用户查询（Query）最相关的记忆项。

由于工作记忆通常存储在内存中（容量较小），它没有依赖外部复杂的向量数据库，而是实现了一个 **轻量级的混合检索算法**。

下面详细讲解这段代码的 **7 个关键步骤** 及其原理：

### 1. 清理与预过滤 (Line 64-78)
```python
self._expire_old_memories()  # 清理过期的记忆
# ...
active_memories = [m for m in self.memories if not m.metadata.get("forgotten", False)]
# ...
filtered_memories = [m for m in active_memories if m.user_id == user_id]
```
*   **原理**：在检索前，先剔除无效数据。
    *   `_expire_old_memories()`：根据 TTL（生存时间）自动删除太旧的记忆。
    *   `forgotten` 标记：排除被显式标记为“遗忘”的记忆。
    *   `user_id` 过滤：确保多用户环境下，只检索当前用户的记忆，防止数据泄露。

### 2. 语义向量检索 (TF-IDF) (Line 80-106)
```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# ...
documents = [query] + [m.content for m in filtered_memories]
tfidf_matrix = vectorizer.fit_transform(documents)
```
*   **原理**：使用 **TF-IDF (Term Frequency-Inverse Document Frequency)** 将文本转换为向量。
    *   **TF-IDF** 是一种统计方法，用来评估一个词对一个文件集或一个语料库中的其中一份文件的重要程度。
    *   代码将 `query`（查询）和所有记忆内容（`memories`）放在一起进行向量化。
*   **相似度计算**：
    *   使用 **余弦相似度 (Cosine Similarity)** 计算查询向量与每条记忆向量之间的夹角。
    *   结果 `vector_score` 表示语义上的相似程度（0到1之间）。
    *   *注意：这里使用了 `try-except` 块，如果 `sklearn` 库未安装或出错，会平滑降级，不中断程序。*

### 3. 关键词匹配 (Line 118-128)
```python
if query_lower in content_lower:
    keyword_score = len(query_lower) / len(content_lower)
else:
    # 分词匹配 (Jaccard Similarity 变体)
    intersection = query_words.intersection(content_words)
    keyword_score = len(intersection) / len(query_words.union(content_words)) * 0.8
```
*   **原理**：作为向量检索的补充，处理精确匹配的情况。
    *   **包含匹配**：如果查询字符串直接包含在记忆中，得分基于长度比例。
    *   **集合匹配**：如果不是直接包含，计算查询词和记忆词的 **交集 (Intersection)** 与 **并集 (Union)** 的比率（类似 Jaccard 相似系数）。

### 4. 混合打分 (Hybrid Scoring) (Line 130-134)
```python
if vector_score > 0:
    base_relevance = vector_score * 0.7 + keyword_score * 0.3
else:
    base_relevance = keyword_score
```
*   **原理**：结合两种检索方式的优点。
    *   **向量检索 (70%)**：擅长捕捉语义关联（例如 "apple" 和 "fruit"）。
    *   **关键词匹配 (30%)**：擅长捕捉精确名词或术语。
    *   这种加权机制（0.7 vs 0.3）表明系统更倾向于语义理解，但也不忽略字面匹配。

### 5. 时间衰减 (Time Decay) (Line 136-138)
```python
time_decay = self._calculate_time_decay(memory.timestamp)
base_relevance *= time_decay
```
*   **原理**：模拟人类记忆的遗忘曲线。
    *   越新的记忆，`time_decay` 越接近 1。
    *   越旧的记忆，`time_decay` 越小。
    *   这确保了系统优先回忆起“最近发生”的事情。

### 6. 重要性加权 (Importance Weighting) (Line 140-142)
```python
importance_weight = 0.8 + (memory.importance * 0.4)
final_score = base_relevance * importance_weight
```
*   **原理**：某些记忆比其他记忆更重要（例如用户的名字 vs 用户随口说的一句话）。
    *   `memory.importance` 是一个 0-1 的属性。
    *   重要的记忆会获得额外的分数加成，使其更容易被检索到，即使它稍微旧一点。

### 7. 排序与截断 (Line 147-149)
```python
scored_memories.sort(key=lambda x: x[0], reverse=True)
return [memory for _, memory in scored_memories[:limit]]
```
*   **原理**：
    *   按最终计算的 `final_score` 从高到低排序。
    *   只返回前 `limit` 个结果（通常是 5 个），避免上下文窗口过大。

### 总结
这段代码实现了一个 **即插即用、无需外部依赖** 的智能检索器。它通过 **TF-IDF 语义向量 + 关键词匹配 + 时间衰减 + 重要性加权** 四重机制，精准地从工作记忆中提取与用户当前意图最相关的信息。