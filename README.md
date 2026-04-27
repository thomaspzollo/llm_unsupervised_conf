
## Unsupervised Confidence Calibration for Reasoning LLMs from a Single Generation

Code for the paper *Unsupervised Confidence Calibration for Reasoning LLMs from a Single Generation*

Paper: [[Link]](https://arxiv.org/abs/2604.19444)

### Setup and installation

To get setup, run:

    git clone <repo-url>
    cd llm_unsupervised_conf

Then, once you've setup a virtual environment for the project

    pip install -e .


### Producing LLM outputs and embeddings

To reproduce our experiments, first run the following bash scripts to produce LLM outputs, embeddings, and verbalized confidence:

    cd scripts/
    bash run_produce_data.sh
    bash run_embeddings.sh
    bash run_verbalized_confidence.sh
    bash run_base.sh

### Paper experiments

Then, run notebooks in `experiments/` in numerical order

 - `00*` run initial setup
 - `01*` run main experiments
 - `02*` run distribution shift experiments
 - `03*` run selective prediction experiment
 - `04*` run linguistic calibration experiment