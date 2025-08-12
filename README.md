# Ai-tools
Primary differences between TensorFlow and PyTorch & when to choose

Programming style:

TensorFlow: Uses static computation graphs (TF1.x) and now also supports eager execution (TF2.x). More production-focused with deployment tools like TensorFlow Serving and TensorFlow Lite.

PyTorch: Uses dynamic computation graphs (define-by-run), making it more intuitive and Pythonic for experimentation and debugging.

Ecosystem:

TensorFlow: Strong integration with Google’s tools, TPU support, and mature production pipelines.

PyTorch: Favored in research, fast prototyping, and academic work; widely adopted for flexibility.

When to choose:

TensorFlow: When production deployment, scalability, and cross-platform support are priorities.

PyTorch: When doing rapid prototyping, research, or needing more intuitive debugging.

Q2: Two use cases for Jupyter Notebooks in AI development

Interactive prototyping and data exploration — Run code cells incrementally, visualize datasets, and adjust models without restarting the whole program.

Model training demonstrations and documentation — Combine code, text, and visual outputs in one place for tutorials, reports, and reproducible research.

Q3: How spaCy enhances NLP compared to basic Python string operations

spaCy provides advanced, efficient NLP pipelines (tokenization, part-of-speech tagging, named entity recognition, dependency parsing) that are language-aware and optimized for speed.

Basic Python string ops (like .split() or .replace()) treat text as raw sequences of characters without linguistic context, making them inadequate for complex NLP tasks.

In short: spaCy understands language structure, not just raw text patterns.

Comparative Analysis — Scikit-learn vs TensorFlow

Feature	Scikit-learn	TensorFlow
Target applications	Classical ML (linear regression, decision trees, clustering, etc.)	Deep learning (neural networks, CNNs, RNNs, transformers)
Ease of use	Very beginner-friendly; consistent APIs; minimal setup needed	Steeper learning curve; more complex model setup
Community support	Large, mature, well-documented; strong for traditional ML	Large and active, especially in DL/AI production environments
