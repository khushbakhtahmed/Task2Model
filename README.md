📦 Released Datasets: EarlyBird & Task2Model

We release EarlyBird and Task2Model, two datasets derived from scientific publications that document how NLP models are mentioned and how they are linked to scientific tasks in the research literature.

🐦 EarlyBird: Model Mention Dataset

<strong>EarlyBird (EarlyBird Model Mention Dataset)</strong> is an intermediate dataset that captures all detected mentions of NLP models within individual scientific papers.

Each entry is enriched with structured metadata, including:

<ul> <li><strong>Paper identifiers</strong></li> <li><strong>Section indicators</strong></li> <li><strong>Model mention counts</strong></li> </ul>

This enables fine-grained analyses of <em>model usage patterns</em>, <em>reporting practices</em>, and <em>mention behavior</em> across the scientific literature.

🔗 Task2Model: Scientific Tasks to Models

Building on EarlyBird, we further release <strong>Task2Model</strong>, a large-scale dataset of scientific problem descriptions (tasks) linked to NLP models.

Task2Model emphasizes:

<ul> <li><strong>Hierarchical and family-level evaluation</strong></li> <li><strong>Systematic quality audits</strong></li> <li><strong>Explicit leakage-prevention measures</strong></li> </ul>

Together, these design choices provide a reliable framework for benchmarking <em>task-to-model mapping</em> approaches.

<p> The dataset comprises <strong>more than 99,000</strong> scientific problem descriptions associated with <strong>over 500</strong> unique NLP models. </p>

Each row in Task2Model contains a <strong>cleaned and filtered text snippet</strong>, with the model name <em>masked</em>, capturing a scientific task or problem context along with associated metadata.

📊 Dataset Schema (Task2Model)
<ul> <li> <strong><code>Scientific_problem_descriptions</code></strong><br/> Cleaned and filtered text snippets with the model name masked, capturing a scientific task or problem context. </li> <li> <strong><code>Full_ID</code></strong><br/> Canonical full identifier of the NLP model (e.g., <code>openai-community/gpt2</code>). </li> <li> <strong><code>Short_ID</code></strong><br/> Normalized short identifier used as a collapsed label (e.g., <code>gpt2</code>). </li> <li> <strong><code>paper_id</code></strong><br/> Identifier linking the task description to the original scientific publication. </li> <li> <strong><code>family_label</code></strong><br/> Higher-level model family (e.g., <code>BERT</code>, <code>GPT</code>, <code>LLaMA</code>), enabling hierarchical and family-level evaluation. </li> </ul>
🎯 Purpose and Use Cases

These datasets are designed to support research on how NLP models are mentioned, reported, and applied in scientific literature. They can be used for:

<ul> <li>Analyzing trends and common usage patterns of NLP models</li> <li>Studying reporting practices and model adoption across scientific domains</li> <li>Grouping models by families for aggregated and hierarchical analyses</li> <li>Training and evaluating systems for model mention recognition</li> <li>Benchmarking task-to-model mapping methods</li> <li>Linking model mentions to scientific task contexts for downstream empirical studies</li> </ul>

Overall, these resources provide a foundation for <strong>systematic analysis of NLP model usage</strong> and for the development and evaluation of <strong>task-to-model recommendation and benchmarking methods</strong>.

⚙️ Reproducibility & Repository Structure

In addition to the datasets, we provide <strong>scripts to recreate the datasets</strong> following the same construction procedure.

<ul> <li> The <code>dataset construction/</code> directory contains <strong>14 sequential pipeline scripts</strong> covering: <ul> <li>Model mention extraction</li> <li>Context construction</li> <li>Masking</li> <li>Filtering</li> <li>Label normalization</li> </ul> </li> <li> The <code>Datasets+Splits/</code> directory contains the datasets and splits required to initialize the construction process and reproduce the experimental results. </li> </ul>
Available Splits
<ul> <li> <strong><code>train_group</code> / <code>test_group</code></strong><br/> Group-based (paper-level) splits designed to prevent leakage. </li> <li> <strong><code>train_stratified</code> / <code>test_stratified</code></strong><br/> Random stratified splits used for comparative evaluation. </li> </ul>

To reproduce the results reported in the paper, the repository also includes:

<ul> <li><code>Initial-Classification/</code></li> <li><code>TF-IDFandSVM_Classification/</code></li> <li><code>Model Retrieval/</code></li> </ul>

These directories provide the corresponding baseline implementations and evaluation code.
