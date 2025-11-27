
# 🧬 Gene Expression Explanation – README
You can get LLM explanations from our  `.explain` step by setting `llm_explain=True`. Therefore you need to setup LLM Clients

You can use either:

* **Mistral API** (cloud-based)
* **Ollama** (local models running on your machine)



## Requirements

Depending on if you want to use Mistral or Ollama you need to have:
- a Mistral API key
- Ollam installed and at least one model served

---

# ⚙️ Environment Setup

## 1. Using Mistral API

Set an API key in your `.env` file in the root of the repository

```
MISTRAL_API_KEY=your_api_key_here
```

### Models

You may use any model served by Mistral, for example:

* `mistral-small-latest`
* `mistral-medium-latest`
* `mistral-large-latest`

Make sure the `model_name` you pass to `.expalain` matches an available Mistral model.

---

## 2. Using Ollama

Ollama runs models **locally**.
You must first install Ollama and pull the model you want:

```bash
ollama pull <model-name>
```


Your `model_name` must match exactly the name of the model served by Ollama:

* `qwen2.5:0.5b`
* `deepseek-r1:8b`

---

# 🧬 Using `.explain()` for Gene Expression Interpretation

The `.explain()` method can generate a short biological explanation and hypothesis about what is happening in disease vs healthy samples given a list of altered genes.

### Example
```python
varix.explain(explainer="", llm_explain=True, llm_client="ollama", llm_model="qwen2.5:0.5b")
```

or 
```python
varix.explain(explainer="", llm_explain=True, llm_client="mistral", llm_model="mistral-small-latest")
```