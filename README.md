
## Unsupervised Confidence Calibration for Reasoning LLMs from a Single Generation

Code for the paper *Unsupervised Confidence Calibration for Reasoning LLMs from a Single Generation*

To get setup, run:

    git clone <repo-url>
    cd llm_unsupervised_conf

Then, once you've setup a virtual environment for the project

    pip install -e .

To reproduce our experiments, first run the following bash scripts to produce LLM outputs, embeddings, and verbalized confidence:

    cd scripts/
    bash run_produce_data.sh
    bash run_embeddings.sh
    bash run_verbalized_confidence.sh
    bash run_base.sh

Then, run notebooks in `experiments/` in numerical order

 - Notebooks numbered 00X run initial setup
 - Notebooks numbered 01X run main experiments
 - Notebooks numbered 02X run distribution shift experiments
 - Notebooks numbered 03X run selective prediction experiment
 - Notebooks numbered 04X run linguistic calibration experiment