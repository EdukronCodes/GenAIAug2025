# Module 12: End-to-End Agentic AI System

**Course:** Generative AI & Agentic AI  
**Module Duration:** 1 week  
**Class:** 20

---

## Class 20: Agentic AI Systems – Advanced

### Topics Covered

- Agent types: Reactive, Planning, Multi-Agent Systems
- Memory management: Long-term memory, episodic recall
- Tool use and reasoning chains
- Building a simple multi-agent system using LangGraph or LangChain Agents

### Learning Objectives

By the end of this class, students will be able to:
- Understand different types of AI agents
- Implement memory systems for agents
- Enable tool use in agents
- Build multi-agent systems
- Deploy agentic AI applications

### Core Concepts

#### Agent Types

**1. Reactive Agents:**
- React to current state
- No memory or planning
- Simple decision-making
- Fast responses

**Example:**
- Chatbot responding to current message
- Rule-based systems

**2. Planning Agents:**
- Create plans before acting
- Multi-step reasoning
- Goal-oriented behavior
- More sophisticated

**Example:**
- Task decomposition
- Multi-step problem solving

**3. Multi-Agent Systems:**
- Multiple agents working together
- Communication between agents
- Specialized roles
- Complex coordination

**Example:**
- Agent teams
- Collaborative problem solving

#### Memory Management

**Types of Memory:**

**1. Short-term Memory:**
- Current conversation context
- Recent interactions
- Limited capacity
- Fast access

**2. Long-term Memory:**
- Persistent knowledge
- User preferences
- Historical data
- Slower access

**3. Episodic Memory:**
- Specific events
- Experiences
- Contextual information
- Temporal ordering

**Implementation:**

**Conversation Memory:**
```python
from langchain.memory import ConversationBufferMemory

memory = ConversationBufferMemory()
memory.save_context({"input": "Hello"}, {"output": "Hi there!"})
```

**Long-term Memory:**
```python
# Store in vector database
vectorstore.add_texts([memory_text], metadatas=[{"timestamp": now()}])
```

**Episodic Memory:**
```python
# Store events with context
episodes = [
    {"event": "user_query", "context": {...}, "timestamp": ...}
]
```

#### Tool Use

**What are Tools?**
- External functions agents can use
- Extend agent capabilities
- Examples: web search, calculator, API calls

**Tool Types:**
- **Search:** Web search, database queries
- **Computation:** Calculator, code execution
- **APIs:** External service calls
- **Custom:** Domain-specific tools

**Implementation with LangChain:**
```python
from langchain.tools import Tool
from langchain.agents import initialize_agent

def search_tool(query: str) -> str:
    # Web search implementation
    return results

tools = [
    Tool(
        name="Search",
        func=search_tool,
        description="Search the web for information"
    )
]

agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="zero-shot-react-description"
)
```

**Reasoning Chains:**
- Agent reasons about tool use
- Plans sequence of actions
- Executes tools
- Synthesizes results

#### Building Multi-Agent Systems

**Architecture:**
```
User Query
    ↓
Orchestrator Agent
    ↓
    ├─→ Research Agent → Tool: Search
    ├─→ Analysis Agent → Tool: Analysis
    └─→ Synthesis Agent → Tool: LLM
    ↓
Final Response
```

**LangGraph Implementation:**
```python
from langgraph.graph import StateGraph, END

def research_node(state):
    # Research agent logic
    return {"research": results}

def analysis_node(state):
    # Analysis agent logic
    return {"analysis": results}

def synthesis_node(state):
    # Synthesis agent logic
    return {"response": final_answer}

# Build graph
workflow = StateGraph()
workflow.add_node("research", research_node)
workflow.add_node("analysis", analysis_node)
workflow.add_node("synthesis", synthesis_node)
workflow.set_entry_point("research")
workflow.add_edge("research", "analysis")
workflow.add_edge("analysis", "synthesis")
workflow.add_edge("synthesis", END)
```

**Coordination:**
- Message passing
- Shared state
- Task delegation
- Result aggregation

#### Agent Capabilities

**Reasoning:**
- Chain-of-thought
- Multi-step thinking
- Problem decomposition

**Planning:**
- Goal setting
- Action sequences
- Resource allocation

**Tool Use:**
- Function calling
- API integration
- External resources

**Memory:**
- Context retention
- Learning from experience
- User preferences

**Communication:**
- Inter-agent messaging
- User interaction
- Status reporting

### Advanced Topics

#### Agent Orchestration

**Patterns:**
- **Sequential:** Agents work in sequence
- **Parallel:** Agents work simultaneously
- **Hierarchical:** Agents organized in hierarchy
- **Market-based:** Agents bid for tasks

#### Agent Evaluation

**Metrics:**
- Task completion rate
- Response quality
- Tool usage efficiency
- User satisfaction

**Evaluation Framework:**
- Define success criteria
- Test scenarios
- Measure performance
- Iterate improvements

### Readings

- Recent papers on agentic AI:
  - "ReAct: Synergizing Reasoning and Acting" (Yao et al., 2022)
  - "AutoGPT: Autonomous Agents" (research)
  - "LangGraph: Multi-Agent Workflows" (documentation)

- LangGraph documentation:
  - [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)

- Multi-agent system research:
  - Survey papers on multi-agent systems
  - Recent agentic AI papers

 

### Additional Resources

- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangChain Agents](https://python.langchain.com/docs/modules/agents/)
- [AutoGPT](https://github.com/Significant-Gravitas/AutoGPT)
- [Agentic AI Research](https://arxiv.org/search/?query=agentic+AI)

### Key Takeaways

1. Agentic AI enables autonomous, goal-oriented behavior
2. Different agent types suit different tasks
3. Memory management is crucial for context-aware agents
4. Tool use extends agent capabilities beyond language
5. Multi-agent systems enable complex collaborative tasks
6. Proper orchestration ensures effective agent coordination
7. Evaluation frameworks help improve agent performance
8. Agentic AI represents the future of autonomous AI systems

---

**Previous Module:** [Module 11: Frameworks, Libraries & Platforms Overview](../module_11.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)

