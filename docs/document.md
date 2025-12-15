
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

# `EpisodicMemory`（情景记忆）
          
`EpisodicMemory`（情景记忆）是模仿人类记忆系统中存储具体生活事件的部分。在 `HelloAgents` 中，它被设计用来记录 Agent 与用户交互的详细历史。

以下是对这四个特点的具体实现原理的详细解读：

### 1. 存储具体的交互事件
**实现原理**：
*   **Episode 对象**：代码中定义了一个 `Episode` 类（Line 23-45），它就是“交互事件”的载体。
*   **权威存储 (SQLite)**：使用 `SQLiteDocumentStore` (Line 71) 将每一个事件持久化到本地磁盘的 `memory.db` 数据库中。
*   **双重存储架构**：
    *   **热数据（内存）**：`self.episodes` 列表维护了当前活跃的记忆对象，用于快速访问。
    *   **冷数据（磁盘）**：所有事件都被完整写入 SQLite 数据库，确保掉电不丢失，并作为数据的权威来源 (Line 115)。

### 2. 包含丰富的上下文信息
**实现原理**：
*   **Metadata 提取**：在 `add` 方法中 (Line 90-95)，系统不仅仅存储聊天的文本内容 (`content`)，还从元数据中提取了大量的上下文信息：
    *   `session_id`：所属会话 ID。
    *   `participants`：参与者。
    *   `outcome`：事件的结果/产出。
    *   `context` 字典：任意扩展的键值对上下文。
*   **结构化存储**：这些上下文信息被存储在 SQLite 的 `properties` JSON 字段中 (Line 122-128) 和 `Episode` 对象的属性中。这使得记忆不再是孤立的文本，而是带有“当时环境”的立体数据。

### 3. 按时间序列组织
**实现原理**：
*   **时间戳索引**：每个 `Episode` 都有 `timestamp` 属性。
*   **时间线检索 (`get_timeline`)**：专门提供了一个 `get_timeline` 方法 (Line 528)，它会对 `episodes` 按时间戳倒序排列 (`sort(key=lambda x: x.timestamp, reverse=True)`)，从而生成一条清晰的时间线。
*   **基于时间的过滤**：检索函数 `retrieve` 支持 `time_range` 参数 (Line 158)，利用 SQLite 的时间索引快速筛选特定时间段内的记忆 (Line 169-170)。
*   **近因效应 (`recency_score`)**：在检索打分时 (Line 219-220)，计算 `age_days`（距今天数），并给予最近发生的事件更高的权重 (`recency_score = 1.0 / (1.0 + age_days)`)，模仿人类“对最近发生的事记得更清楚”的特性。

### 4. 支持模式识别和回溯
**实现原理**：
*   **模式识别 (`find_patterns`)**：实现了一个简单的分析引擎 (Line 470)，它遍历用户的历史记忆，统计高频特征：
    *   **关键词频率**：统计用户经常提到的词汇 (Line 486-491)。
    *   **上下文模式**：统计重复出现的上下文键值对（例如 `mood:happy` 出现了几次）(Line 493-496)。
    *   **置信度计算**：计算频率占比 (`frequency / len(episodes)`)，筛选出显著的行为模式 (Line 501-517)。
*   **回溯能力**：
    *   通过 `session_id` 将离散的 `Episode` 串联成完整的会话 (Line 61 `self.sessions`)。
    *   `get_session_episodes` 方法 (Line 462) 允许 Agent 回溯并重现某一次特定会话的完整上下文，这对于处理“接着上次的话题说”这种请求至关重要。

### 总结
`EpisodicMemory` 通过 **SQLite + 内存对象 + 统计分析** 的组合，实现了一个既能像日志一样精准记录（Event & Time），又能像分析师一样挖掘规律（Pattern & Context）的记忆系统。