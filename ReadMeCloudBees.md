# Short Explanations
## What you chose to implement and why:
  - Implementations:
    - Code: functions_mlflow.py
    - Data: train.spacy, dev.spacy
    - Config: config.cfg
  - I generated three MLOps components (mlflow_after_creation.v1, mlflow_before_to_disk.v1, and mlflow_logger.v1) that can be integrated into the spaCy training pipeline. The reason why I did this is that SpaCy is a widely used NLP toolkit, and while large language models can handle many NLP tasks, they are often slower compared to classical tools. Thanks to SpaCy’s structured workflow, it can help reduce the instability of LLM-based approaches, making it valuable to understand and leverage traditional NLP pipelines. In addition, SpaCy offers a well-organized training process where a CLI and configuration file can define the entire workflow, but it lacks monitoring and model version control. My MLflow components address this gap by extending a regular training pipeline into a more complete MLOps workflow, making training and experiment tracking easier and more reproducible. Finally, by integrating SpaCy with MLflow, I can also scale training efficiently on GPU clusters through SpaCy’s CLI, ensuring both performance and scalability.
## How to run your code:
```bash
python -m venv .env
source .env/bin/activate
pip install -r requirements.txt
python -m spacy train config.cfg --paths.train ./train.spacy --paths.dev ./dev.spacy --output ./output --code functions_mlflow.py
# Optional
mlflow ui --backend-store-uri ./mlruns --port 5000
```
## Any assumptions or limitations:
  - Assumptions: I assumed that GPU resources may be available, but the training workflow should still be able to run on CPU without issues. I also assumed that the training and development data would already be labeled and provided in spaCy’s .spacy format, so no additional data preprocessing would be required. Finally, I assumed that the scope of this task was to demonstrate MLOps integration for a single component (NER) as a proof of concept, rather than to build and deliver a full production-ready pipeline.
  - Limitations: There are a few known limitations. On the spaCy side, multi-GPU training requires the use of spacy-ray and may involve additional cluster setup, which is not included by default in this implementation. On the MLflow side, the logging currently covers metrics, configs, and final artifacts, but it does not yet include advanced capabilities such as model registry stage transitions or deployment-ready serving APIs.

# Reflection
## Did coding assistant help you move faster? 
- Yes, ChatGPT helped me brainstorm ideas and draft the initial code quickly. It provided clear explanations and examples that made it easier to get started. Cline assisted by automatically testing the code, reading errors, and suggesting fixes. Together, these tools saved me significant time compared to working manually. Without coding assistants, exploring five different open-source projects would have taken much longer.
## Did coding assistant generate incorrect or surprising suggestions? 
- My original thought was to have the AI generate a Language class component, but instead it produced three registry functions, which turned out better than I expected. However, the generated code did not always follow PEP8 formatting perfectly, and some snippets used deprecated spaCy settings that required adjustment to match the current version. In a few cases, the assistant also suggested overly complex solutions, such as custom readers or advanced configurations, when simpler approaches would have been sufficient.
## Where was coding assistant most/least useful? 
- The coding assistant was most useful in its agentic coding loop. It generated code and fixed errors automatically. When it worked with searching MCP, the answer can be even better. The least useful moments were when it produced overly complicated snippets. Sometimes these mixed with my code and reduced readability.