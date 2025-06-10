# Verigen

This repository contains utilities for translating natural language requirements to Verilog designs.

The `nl2spec` directory bundles the original *nl2spec* tool which converts natural language into Linear Temporal Logic (LTL).  We build on top of this by providing a small script, `nl2verilog.py`, that uses `nl2spec` together with synthesis tools such as `syfco`, `ltlsynt`, and `abc` to generate Verilog code.

## Requirements
- Python 3.12
- Packages listed in `nl2spec/requirements.txt`
- External tools: `syfco`, `ltlsynt`, `aigtoaig` from the [AIGER](https://github.com/arminbiere/aiger) suite, and `abc` from [Berkeley ABC](https://github.com/berkeley-abc/abc)

The external tools are not included and must be installed separately.

## Usage
1. Install the Python dependencies:
   ```bash
   python3 -m pip install -r nl2spec/requirements.txt
   ```
2. Provide an API key for the LLM in a file. For OpenAI models this is usually `oai_key.txt`.
3. Run the pipeline:
   ```bash
   python3 nl2verilog.py --nl "Every request is eventually granted" \
       --keyfile PATH/TO/oai_key.txt
   ```
   By default all signals found in the LTL formula are treated as outputs.  You
   can still provide `--inputs` or `--outputs` to override the automatic
   detection.  The script writes `controller.v` containing the synthesized
   Verilog controller.

The script assumes that `syfco`, `ltlsynt`, `aigtoaig`, and `abc` are available in the system `PATH`.
