We release EarlyBird and Task2Model, two datasets derived from scientific publications that document how NLP models are mentioned and linked to scientific tasks in the research literature.

EarlyBird (EarlyBird Model Mention Dataset) is an intermediate dataset that captures all detected mentions of NLP models within individual scientific papers. Each entry is accompanied by rich metadata, including paper identifiers, section indicators, and model mention counts, enabling fine-grained analyses of model usage, reporting practices, and mention patterns across the scientific literature.

Building on this resource, we further release Task2Model, a large-scale dataset of scientific problem descriptions or tasks linked to NLP models.Task2Model emphasizes hierarchical and family-level evaluation, systematic quality audits, and explicit leakage-prevention measures, providing a reliable framework for benchmarking task-to-model mapping approaches. The dataset comprises more than 99,000 scientific problem descriptions associated with over 500 unique NLP models.

Each row in the Task2Model dataset contains a cleaned and filtered text snippet with the model name masked, capturing a scientific task or problem context, along with associated metadata.

Columns:

Scientific_problem_descriptions: Cleaned and filtered text snippets in which the model name is masked, capturing a scientific task or problem context.
Full_ID: Canonical full identifier of the NLP model associated with the task (e.g., openai-community/gpt2).
Short_ID: Normalized short identifier of the model, used as a collapsed label for analysis (e.g., gpt2).
paper_id: Identifier linking the task description to the original scientific publication.
family_label: Higher-level model family corresponding to the model (e.g., BERT, GPT, LLaMA), enabling hierarchical and family-level evaluation.

Purpose and Use Cases:

These datasets are designed to support research on how NLP models are mentioned, reported, and applied in scientific literature. They can be used for:

Analyzing trends and common usage patterns of NLP models in research.

Studying reporting practices and model adoption across scientific domains.

Grouping models by families for aggregated and hierarchical analyses.

Training and evaluating systems for model mention recognition and task-to-model mapping.

Linking model mentions to their scientific task contexts for downstream empirical studies.

Overall, these resources provide a foundation for systematic analysis of NLP model usage in scientific publications and for the development and evaluation of task-to-model recommendation and benchmarking methods.

