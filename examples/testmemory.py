import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from hello_agents import SimpleAgent, HelloAgentsLLM, ToolRegistry
from hello_agents.tools import MemoryTool

# 创建具有记忆能力的Agent
llm = HelloAgentsLLM()
agent = SimpleAgent(name="记忆助手", llm=llm)

# 创建记忆工具
memory_tool = MemoryTool(user_id="user123")
tool_registry = ToolRegistry()
tool_registry.register_tool(memory_tool)
agent.tool_registry = tool_registry
 
# 体验记忆功能
print("=== 添加多个记忆 ===")

params = {
    "action": "add",
    "content": "用户张三是一名Python开发者，专注于机器学习和数据分析",
    "memory_type": "semantic",
    "importance": 0.8
}

# 添加第一个记忆
result1 = memory_tool.run(params)
print(f"记忆1: {result1}")

# 添加第二个记忆
params["content"] = "李四是前端工程师，擅长React和Vue.js开发"
params["importance"] = 0.7
result2 = memory_tool.run(params)   
print(f"记忆2: {result2}")

# 添加第三个记忆
params["content"] = "王五是产品经理，负责用户体验设计和需求分析"
params["importance"] = 0.6
result3 = memory_tool.run(params)   
print(f"记忆3: {result3}")

print("\n=== 搜索特定记忆 ===")
# 搜索前端相关的记忆
print("🔍 搜索 '前端工程师':")
params["action"] = "search"
params["query"] = "前端工程师"
params["limit"] = 3
result = memory_tool.run(params)
print(result)

print("\n=== 记忆摘要 ===")
params["action"] = "summary"
result = memory_tool.run(params)
print(result)
