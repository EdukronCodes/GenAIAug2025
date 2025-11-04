# Module 1: Foundations of Generative & Agentic AI

**Course:** Generative AI & Agentic AI  

**Module Duration:** 1 week  

**Class:** 1

---

## Class 1: Introduction to Generative AI & Agentic AI

### Topics Covered

- What is Generative AI?

- Evolution from Traditional AI → Generative AI → Agentic AI

- Core components: Foundation Models, Embeddings, Context Windows

- Agentic AI concepts: Reasoning, Planning, Tool Use, Memory

- Real-world use cases and architecture overview

---

### Learning Objectives

By the end of this module, students will be able to:

- Define Generative AI and distinguish it from traditional AI approaches

- Understand the evolution and progression toward Agentic AI

- Identify core components of generative AI systems

- Explain key concepts in Agentic AI: reasoning, planning, tool use, and memory

- Recognize real-world applications and use cases

- Understand the theoretical foundations underlying generative models

- Analyze the architectural differences between traditional, generative, and agentic AI systems

---

## Core Concepts

### 1. Generative AI Fundamentals

#### What is Generative AI?

Generative AI represents a fundamental paradigm shift in artificial intelligence, moving from systems that primarily classify, predict, or make decisions to systems that can create new, original content across multiple modalities including text, images, audio, video, and code.

Traditional AI systems are discriminative in nature—they learn to distinguish between different classes or predict outcomes based on input data.

These systems excel at tasks like classification (e.g., identifying spam emails), regression (e.g., predicting house prices), and decision-making (e.g., recommending products).

However, they are fundamentally limited in their ability to create new content that didn't exist in their training data.

Generative AI, in contrast, learns the underlying probability distribution of data and can sample from this distribution to generate new instances that are similar to but distinct from the training data.

This capability enables the creation of novel content that follows the patterns, styles, and structures learned during training.

**Pro Tip:** When evaluating generative AI for your use case, consider whether you truly need content generation. If you only need classification or prediction, traditional discriminative models may be more efficient and reliable.

**Common Pitfall:** Assuming generative AI is always better than traditional AI. Many tasks don't require generation capabilities, and using generative models for simple classification can introduce unnecessary complexity and cost.

**Check Your Understanding:**
1. What is the fundamental difference between discriminative and generative AI models?
2. How does generative AI learn to create new content?
3. What are the key characteristics that distinguish generative AI from traditional AI?

#### Key Characteristics of Generative AI

**Content Creation Capability:**

The primary distinguishing feature of generative AI is its ability to create new content rather than just analyze existing data.

This content can be highly creative, such as writing poetry, generating artwork, composing music, or writing computer code.

The generated content is not simply retrieved or copied from training data but is synthesized through the model's understanding of patterns, relationships, and structures.

**Diversity and Variability:**

Generative models can produce highly diverse outputs for the same input, demonstrating the probabilistic nature of generation.

This diversity is achieved through sampling techniques that introduce randomness into the generation process.

Each generation can explore different creative paths, leading to unique outputs even when given identical prompts.

**Context Understanding:**

Advanced generative models demonstrate sophisticated understanding of context, allowing them to maintain coherence across long sequences of text, understand nuanced instructions, and adapt their output style to match desired formats or tones.

This contextual awareness enables applications like conversational AI, where the model must maintain conversation history and respond appropriately to evolving dialogue.

**Multimodal Capabilities:**

Modern generative AI systems can work across multiple data modalities simultaneously, understanding relationships between text, images, audio, and other forms of data.

This multimodal understanding enables applications like image captioning, text-to-image generation, video generation, and cross-modal retrieval.

#### Theoretical Foundations

**Probabilistic Models:**

At the core of generative AI are probabilistic models that learn probability distributions over data spaces.

These models attempt to capture P(X), the probability distribution of the data, enabling them to sample new instances from this learned distribution.

Different approaches include autoregressive models, variational autoencoders, generative adversarial networks, and diffusion models.

**Autoregressive Generation:**

Many text generation models (like GPT) use autoregressive approaches where each token is generated based on all previous tokens.

This creates a sequence generation process: P(x₁, x₂, ..., xₙ) = P(x₁) × P(x₂|x₁) × ... × P(xₙ|x₁, ..., xₙ₋₁)

The model learns conditional probabilities at each step, enabling coherent long-form generation.

**Transfer Learning:**

Foundation models leverage transfer learning, where models pre-trained on vast, diverse datasets can be adapted to specific tasks with minimal additional training.

This paradigm shift means that instead of training models from scratch for each task, we can fine-tune pre-trained models, dramatically reducing computational requirements and improving performance.

---

### 2. Evolution Timeline: From Traditional AI to Agentic AI

#### Traditional AI (1950s - 1980s)

**Rule-Based Systems:**

The earliest AI systems were rule-based, relying on explicitly programmed if-then rules to make decisions.

These systems were deterministic and worked well for well-defined problems with clear logical structures.

However, they were brittle and couldn't handle ambiguity, exceptions, or learning from data.

**Expert Systems:**

Expert systems attempted to capture human expertise in specific domains through knowledge bases and inference engines.

While successful in narrow domains (like medical diagnosis systems), they required extensive manual knowledge engineering and couldn't scale or adapt easily.

**Limitations:**

Traditional AI systems were fundamentally limited by their inability to learn from data, handle uncertainty, or generalize beyond their explicitly programmed rules.

They required extensive human expertise to encode domain knowledge and couldn't improve their performance automatically.

#### Machine Learning Era (1990s - 2010s)

**Statistical Learning:**

Machine learning introduced the ability to learn patterns from data rather than requiring explicit programming of all behaviors.

Algorithms could identify statistical patterns in data and make predictions based on these learned patterns.

**Supervised Learning:**

Supervised learning algorithms learn mappings from inputs to outputs using labeled training data.

This enabled applications like spam detection, image classification, and recommendation systems.

**Unsupervised Learning:**

Unsupervised learning algorithms discover hidden patterns in data without labeled examples.

This enabled clustering, dimensionality reduction, and anomaly detection.

**Key Limitations:**

While machine learning enabled significant advances, models were still primarily discriminative, focusing on classification and prediction rather than generation.

Most models were task-specific and required retraining for new applications.

#### Deep Learning Revolution (2010s - 2020)

**Neural Networks:**

Deep learning leveraged neural networks with multiple layers to learn hierarchical representations of data.

These networks could automatically discover relevant features, reducing the need for manual feature engineering.

**Convolutional Neural Networks (CNNs):**

CNNs revolutionized computer vision by learning spatial hierarchies of features.

They enabled breakthroughs in image classification, object detection, and image segmentation.

**Recurrent Neural Networks (RNNs):**

RNNs enabled processing of sequential data, making advances in natural language processing, speech recognition, and time series prediction.

Long Short-Term Memory (LSTM) and Gated Recurrent Units (GRU) addressed vanishing gradient problems, enabling longer sequence processing.

**Transfer Learning:**

Pre-trained models could be fine-tuned for specific tasks, reducing training requirements and improving performance.

This marked the beginning of the foundation model paradigm.

#### Generative AI Emergence (2020 - Present)

**Transformer Architecture:**

The introduction of the Transformer architecture in 2017 revolutionized natural language processing.

Self-attention mechanisms enabled parallel processing of sequences and better capture of long-range dependencies.

**Large Language Models (LLMs):**

Models like GPT-3, GPT-4, PaLM, and LLaMA demonstrated emergent capabilities at scale.

These models showed that scaling model size, data, and compute led to qualitatively new behaviors not present in smaller models.

**Multimodal Models:**

Models like CLIP, DALL-E, and GPT-4V demonstrated the ability to understand and generate across multiple modalities.

This enabled new applications combining vision, language, and other data types.

**Key Characteristics:**

Generative AI systems can create novel content, understand context, follow instructions, and adapt to diverse tasks with minimal task-specific training.

#### Agentic AI (2023 - Present)

**Autonomous Agents:**

Agentic AI represents the next evolution, where AI systems can act autonomously, make decisions, and use tools to accomplish goals.

These systems combine generative capabilities with reasoning, planning, and action execution.

**Key Capabilities:**

Agentic systems can break down complex goals into sub-tasks, plan sequences of actions, use external tools and APIs, maintain memory across interactions, and adapt their behavior based on feedback.

**Multi-Agent Systems:**

Advanced agentic systems involve multiple agents working together, each with specialized roles and capabilities.

These systems can solve complex problems through collaboration and coordination.

**Examples:**

Systems like AutoGPT, LangChain Agents, and Claude's tool use capabilities demonstrate early agentic AI implementations.

These systems can search the web, use calculators, call APIs, and perform complex multi-step tasks autonomously.

**Code Example: Simple Agentic AI Pattern**

```python
from langchain.agents import initialize_agent, AgentType
from langchain.llms import OpenAI
from langchain.tools import Tool
import requests

# Define tools the agent can use
def search_web(query: str) -> str:
    """Search the web for information"""
    # Simplified example - in production, use proper search API
    return f"Search results for: {query}"

def calculate(expression: str) -> str:
    """Evaluate mathematical expression"""
    try:
        result = eval(expression)
        return str(result)
    except:
        return "Error: Invalid expression"

# Create tools
tools = [
    Tool(
        name="Web Search",
        func=search_web,
        description="Search the web for current information"
    ),
    Tool(
        name="Calculator",
        func=calculate,
        description="Perform mathematical calculations"
    )
]

# Initialize agent
llm = OpenAI(temperature=0)
agent = initialize_agent(
    tools,
    llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Use the agent
result = agent.run("What is the square root of 144? Then search for recent AI news.")
print(result)
```

**Pro Tip:** Agentic AI systems require careful prompt engineering and tool design. Ensure tools have clear descriptions and error handling to prevent agent confusion.

**Common Pitfall:** Allowing agents unlimited autonomy can lead to unexpected behaviors or high costs. Always implement timeouts, token limits, and safety checks.

---

### Historical Context and Milestones

The origins of AI date to the Dartmouth Workshop (1956), which launched symbolic, logic-driven approaches and expert systems. Through the 1980s, hand-authored rules dominated narrow domains, but brittleness, maintenance overhead, and data scarcity limited generalization and scale.

The 1990s ushered in statistical machine learning, emphasizing data-driven modeling and generalization. The 2010s deep learning wave (ImageNet 2012, sequence models) automated feature learning, culminating in the Transformer (2017) and the rise of foundation and agentic models.

Notable milestones include ImageNet-fueled vision breakthroughs, attention mechanisms and sequence-to-sequence learning, BERT for language understanding, and GPT-series for generation. Instruction tuning and RLHF improved controllability; multimodality and long-context models expanded capability envelopes substantially.

Summary:
- Symbolic AI → statistical ML → deep learning → Transformers → foundation models.
- Instruction tuning, RLHF, and tool use improved reliability and actionability.
- Multimodal and long-context capabilities enabled new end-to-end workflows.

---

### 3. Foundation Models: The Building Blocks of Modern AI

#### Definition and Characteristics

**What are Foundation Models?**

Foundation models are large-scale machine learning models pre-trained on vast, diverse datasets using self-supervised learning.

They serve as the foundation for a wide range of downstream applications through transfer learning, fine-tuning, and prompt engineering.

The term was coined by researchers at Stanford to describe models that are trained on broad data at scale and can be adapted to a wide range of downstream tasks.

**Scale Characteristics:**

Foundation models are characterized by their massive scale: billions or trillions of parameters, trained on petabytes of data, using massive computational resources.

This scale enables emergent capabilities—abilities that emerge at scale but aren't explicitly programmed.

**Diverse Data Training:**

Unlike task-specific models, foundation models are trained on diverse, multimodal data.

This diversity enables them to understand and work across many domains, languages, and modalities.

**Transfer Learning:**

Foundation models demonstrate powerful transfer learning capabilities.

They can be adapted to specific tasks through fine-tuning, prompt engineering, or few-shot learning with minimal task-specific data.

#### Types of Foundation Models

**Language Models:**

Models like GPT, BERT, T5, PaLM, and LLaMA are primarily language-focused, trained on text data.

They understand and generate human language, enabling applications in natural language understanding, generation, translation, summarization, and more.

**Vision Models:**

Models like CLIP, DALL-E, and Stable Diffusion work with images and visual data.

CLIP learns joint representations of images and text, enabling cross-modal understanding and retrieval.

**Multimodal Models:**

Models like GPT-4V, Gemini, and PaLM-E integrate multiple modalities.

They can understand and generate across text, images, audio, and other data types simultaneously.

**Code Models:**

Models like Codex, GitHub Copilot, and StarCoder are trained on code repositories.

They understand programming languages, can generate code, debug, and explain code functionality.

#### Key Examples of Foundation Models

**GPT (Generative Pre-trained Transformer) Series:**

GPT-3 demonstrated that large language models could perform many tasks with few or no examples.

GPT-4 introduced improved reasoning, safety, and multimodal capabilities.

GPT-4 Turbo and GPT-4o further improved performance and efficiency.

**BERT (Bidirectional Encoder Representations from Transformers):**

BERT revolutionized natural language understanding through bidirectional context encoding.

It enabled state-of-the-art performance on many NLP tasks through fine-tuning.

**T5 (Text-to-Text Transfer Transformer):**

T5 framed all NLP tasks as text-to-text problems, creating a unified framework.

This enabled a single model to handle diverse tasks through task-specific prompts.

**CLIP (Contrastive Language-Image Pre-training):**

CLIP learned joint representations of images and text through contrastive learning.

It enables zero-shot image classification, text-to-image retrieval, and semantic image search.

**LLaMA (Large Language Model Meta AI):**

Meta's open-source language models made foundation models accessible to researchers and developers.

LLaMA 2 and LLaMA 3 improved performance and safety while remaining open-source.

#### Capabilities and Limitations

**Capabilities:**

Foundation models demonstrate remarkable capabilities including few-shot learning, in-context learning, reasoning, code generation, and creative tasks.

They can adapt to new tasks with minimal examples, understand context, and generate coherent, relevant outputs.

**Limitations:**

Despite their capabilities, foundation models have limitations including potential for generating incorrect information (hallucinations), lack of real-time knowledge updates, biases in training data, high computational requirements, and challenges with long-term memory.

**Emergent Behaviors:**

At scale, foundation models exhibit emergent behaviors not present in smaller models.

These include chain-of-thought reasoning, instruction following, and tool use capabilities.

---

## Theoretical Deep Dive

### Scaling Laws and Capacity

Empirical scaling laws show performance improves predictably with model size, dataset size, and compute, under suitable optimization. Larger models with more diverse data manifest emergent abilities, but diminishing returns and data quality constraints necessitate careful scaling strategies.

Token budgets, deduplication, and curriculum effects alter scaling behavior; high-quality, diverse corpora often outperform indiscriminate growth. Practical trade-offs balance parameter count, context window, latency, and inference cost for target applications.

Summary:
- Accuracy correlates with model/data/compute scale under proper training.
- Data quality and diversity are as critical as sheer quantity.
- Real-world systems balance accuracy with cost, latency, and context.

### Generalization and In-Context Learning

Large language models learn latent procedures that support in-context learning: they can condition on demonstrations and adapt behavior without parameter updates. This ability emerges from pretraining objectives that encourage pattern induction and representation reuse.

Prompt format, ordering, and inductive biases materially affect few-shot performance. Retrieval augmentation increases effective generalization by grounding outputs in external knowledge, reducing hallucination and improving factuality.

Summary:
- LLMs can learn from examples at inference time (few-shot ICL).
- Prompt design and retrieval strongly influence generalization.
- Grounding reduces hallucinations and improves reliability.

### Optimization Dynamics and Inductive Biases

Transformer training uses stochastic gradient descent variants with schedules, normalization, and regularization shaping learned representations. Architectural choices (depth, width, heads), initialization, and tokenization impose inductive biases that guide what is easy to learn.

Stability techniques—gradient clipping, mixed precision, and optimizer tuning—enable efficient large-scale training. Quantization, distillation, and PEFT approaches (LoRA/QLoRA) preserve accuracy while reducing cost for deployment.

Summary:
- Architecture and optimization choices create useful inductive biases.
- Stability and efficiency techniques are essential at scale.
- PEFT, quantization, and distillation enable practical deployment.

### Alignment, Safety, and Evaluation

Instruction tuning and RLHF align model behavior with human preferences, improving helpfulness and reducing harmful outputs. Safety layers combine policy constraints, refusal behavior, and post-generation filtering to mitigate misuse and bias.

Evaluation blends automatic metrics with human assessment, measuring factuality, robustness, faithfulness to context, and harmful content. Continuous red-teaming and domain-specific guardrails are necessary for production readiness.

Summary:
- Alignment methods (SFT, RLHF) improve controllability and safety.
- Evaluation must include human judgment and domain checks.
- Guardrails and governance are core to responsible deployment.

---

## Practical Code Examples

### Example 1: Basic Text Generation with OpenAI

```python
import os
from openai import OpenAI

# Initialize client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def generate_text(prompt, model="gpt-4", temperature=0.7, max_tokens=500):
    """Generate text using OpenAI API"""
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=temperature,
            max_tokens=max_tokens
        )
        return response.choices[0].message.content
    except Exception as e:
        print(f"Error: {e}")
        return None

# Usage
result = generate_text("Explain quantum computing in simple terms")
print(result)
```

### Example 2: Embedding Generation

```python
from openai import OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def get_embedding(text, model="text-embedding-3-small"):
    """Generate embedding for text"""
    response = client.embeddings.create(
        model=model,
        input=text
    )
    return response.data[0].embedding

# Usage
embedding = get_embedding("Generative AI is transforming technology")
print(f"Embedding dimension: {len(embedding)}")
```

### Example 3: Context Window Management

```python
def manage_context_window(messages, max_tokens=4096, system_prompt="You are helpful."):
    """Manage context window by truncating oldest messages"""
    token_count = len(system_prompt.split())
    
    # Simple token estimation (1 token ≈ 4 characters)
    for msg in messages:
        token_count += len(msg["content"]) // 4
    
    if token_count > max_tokens:
        # Remove oldest user/assistant pairs, keep system
        while token_count > max_tokens * 0.8 and len(messages) > 1:
            removed = messages.pop(1)
            token_count -= len(removed["content"]) // 4
    
    return messages

# Usage
messages = [
    {"role": "system", "content": "You are helpful."},
    {"role": "user", "content": "Long conversation history..."},
    # ... more messages
]
managed = manage_context_window(messages, max_tokens=4096)
```

---

## Quick Reference Guide

### AI Evolution Timeline

| Era | Period | Key Characteristics | Limitations |
|-----|--------|---------------------|-------------|
| Traditional AI | 1950s-1980s | Rule-based, expert systems | Brittle, no learning |
| Machine Learning | 1990s-2010s | Statistical learning, pattern recognition | Task-specific, limited generalization |
| Deep Learning | 2010s-2020 | Neural networks, feature learning | Still discriminative, requires retraining |
| Generative AI | 2020-Present | Content creation, multimodal | Hallucinations, high compute |
| Agentic AI | 2023-Present | Autonomous action, tool use | Complexity, safety concerns |

### Generative vs. Discriminative AI

| Aspect | Discriminative | Generative |
|--------|----------------|------------|
| **Purpose** | Classify/Predict | Create/Generate |
| **Output** | Label/Value | New content |
| **Training** | P(Y\|X) | P(X) or P(X,Y) |
| **Use Cases** | Classification, regression | Text/image generation |
| **Examples** | Logistic regression, CNNs | GPT, DALL-E, Stable Diffusion |

### Foundation Models Comparison

| Model | Type | Capabilities | Use Cases |
|-------|------|--------------|-----------|
| GPT-4 | Text | Generation, reasoning, code | Chatbots, writing, coding |
| CLIP | Vision | Image-text understanding | Image search, captioning |
| DALL-E | Image | Text-to-image generation | Art creation, design |
| Codex | Code | Code generation | Programming assistance |

---

## Case Studies

### Case Study 1: Customer Service Chatbot Transformation

**Challenge:** A major e-commerce company needed to handle 10,000+ customer inquiries daily with 24/7 availability.

**Traditional Approach:**
- Rule-based chatbot with fixed responses
- 30% resolution rate
- Required constant manual updates
- Poor customer satisfaction (2.5/5)

**Generative AI Solution:**
- Implemented GPT-4 based conversational agent
- Trained on company knowledge base
- Integrated with order management system

**Results:**
- 85% first-contact resolution
- 4.2/5 customer satisfaction
- 60% cost reduction
- Reduced response time from 5 minutes to 30 seconds

**Lessons Learned:**
- Hallucination issues required RAG implementation
- Context management crucial for long conversations
- Human oversight needed for sensitive issues

### Case Study 2: Content Generation for Marketing

**Challenge:** Marketing team needed to generate 100+ product descriptions weekly.

**Traditional Approach:**
- Manual writing by content team
- 2 hours per description
- Inconsistent quality and tone
- High cost ($50,000/month)

**Generative AI Solution:**
- Fine-tuned GPT-3.5 for brand voice
- Automated pipeline with human review
- Template-based generation with variations

**Results:**
- 10 minutes per description (90% time savings)
- Consistent brand voice
- 80% cost reduction
- Improved SEO through keyword optimization

**Lessons Learned:**
- Fine-tuning essential for brand consistency
- Human review critical for quality control
- Version control important for iterations

---

## Hands-On Lab: Your First Generative AI Application

**Lab Objective:** Build a simple text generation application with error handling and best practices.

**Steps:**

1. **Setup Environment**
```bash
pip install openai python-dotenv
```

2. **Create Configuration**
```python
# .env file
OPENAI_API_KEY=your_key_here
```

3. **Implement Basic Generator**
```python
import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

class TextGenerator:
    def __init__(self, model="gpt-3.5-turbo"):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
    
    def generate(self, prompt, max_tokens=200, temperature=0.7):
        """Generate text with error handling"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature
            )
            return response.choices[0].message.content
        except Exception as e:
            return f"Error: {str(e)}"

# Usage
generator = TextGenerator()
result = generator.generate("Write a haiku about AI")
print(result)
```

4. **Add Features**
- Context management
- Error handling
- Logging
- Rate limiting

**Expected Outcomes:**
- Working text generation system
- Understanding of API usage
- Error handling patterns
- Best practices implementation

---

## Testing Examples

### Unit Test Template

```python
import unittest
from unittest.mock import Mock, patch
from your_module import TextGenerator

class TestTextGenerator(unittest.TestCase):
    def setUp(self):
        self.generator = TextGenerator()
    
    @patch('openai.OpenAI')
    def test_generate_success(self, mock_openai):
        """Test successful text generation"""
        mock_response = Mock()
        mock_response.choices[0].message.content = "Generated text"
        
        mock_client = Mock()
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client
        
        result = self.generator.generate("test prompt")
        self.assertEqual(result, "Generated text")
    
    def test_generate_error_handling(self):
        """Test error handling"""
        with patch('openai.OpenAI') as mock_openai:
            mock_client = Mock()
            mock_client.chat.completions.create.side_effect = Exception("API Error")
            mock_openai.return_value = mock_client
            
            generator = TextGenerator()
            result = generator.generate("test")
            self.assertIn("Error", result)

if __name__ == "__main__":
    unittest.main()
```

---

## Troubleshooting Guide

| Issue | Symptoms | Solutions |
|-------|----------|-----------|
| **API Key Errors** | Authentication failures | Check environment variables, verify key format |
| **Rate Limiting** | 429 errors, throttling | Implement exponential backoff, add queuing |
| **Hallucinations** | Incorrect or fabricated information | Use RAG, add fact-checking, lower temperature |
| **Context Overflow** | Token limit exceeded | Implement chunking, summarization, context management |
| **Inconsistent Outputs** | Different results for same input | Lower temperature, set seed if available |
| **High Costs** | Unexpected API charges | Monitor usage, implement caching, optimize prompts |
| **Slow Responses** | High latency | Use faster models, optimize prompts, add caching |

**Common Pitfalls:**
- **Pitfall:** Not validating API responses before use
  - **Solution:** Always check response structure and handle errors gracefully
- **Pitfall:** Exceeding context windows without management
  - **Solution:** Implement token counting and truncation strategies
- **Pitfall:** Ignoring rate limits leads to failures
  - **Solution:** Implement retry logic with exponential backoff

---

## Glossary

**Generative AI:** AI systems that can create new content (text, images, code) rather than just classify or predict.

**Discriminative Model:** AI models that learn to distinguish between different classes or predict outcomes based on input data.

**Foundation Model:** Large-scale AI models pre-trained on diverse data that can be adapted to many tasks.

**Emergent Behavior:** Capabilities that appear in large models but not in smaller versions, such as reasoning or tool use.

**Context Window:** The maximum number of tokens a model can process in a single input/output sequence.

**Embedding:** A dense vector representation that captures semantic meaning of text or other data in high-dimensional space.

**Agentic AI:** AI systems that can act autonomously, make decisions, and use tools to accomplish goals.

**Hallucination:** When a generative model produces information that is not present in its training data or context.

**Prompt Engineering:** The practice of designing effective prompts to guide model behavior and improve output quality.

**Fine-tuning:** Adapting a pre-trained model to a specific task using task-specific data.

**Transfer Learning:** Using knowledge learned from one task to improve performance on related tasks.

**RLHF (Reinforcement Learning from Human Feedback):** Training method that aligns model behavior with human preferences.

**RAG (Retrieval-Augmented Generation):** Technique combining information retrieval with generation to produce grounded answers.

---

## Check Your Understanding

1. **Fundamental Concepts:**
   - What distinguishes generative AI from traditional discriminative AI?
   - How do foundation models enable transfer learning?
   - What are the key components of agentic AI systems?

2. **Evolution and History:**
   - What were the main limitations of rule-based AI systems?
   - How did the transformer architecture revolutionize NLP?
   - What emergent behaviors appear in large language models?

3. **Practical Application:**
   - When should you use generative AI vs. traditional AI?
   - How do you manage context windows effectively?
   - What are the main risks and how do you mitigate them?

4. **Technical Details:**
   - How does autoregressive generation work?
   - What is the role of embeddings in semantic understanding?
   - How do scaling laws affect model performance?

---

**Previous Module:** N/A (First Module)  
**Next Module:** [Module 2: GenAI Project Architecture & Flow](../module_02.md)  
**Back to Syllabus:** [Course Syllabus](../gen_ai_syllabus.md)
