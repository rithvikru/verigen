#!/usr/bin/env python3
import argparse
import subprocess
import os
import sys
import tempfile
import re

# Ensure nl2spec modules are importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "nl2spec", "src"))
from nl2spec.src import backend


def nl_to_ltl(nl, model, prompt):
    args = argparse.Namespace(
        model=model,
        nl=nl,
        fewshots="",
        keyfile="",
        keydir="",
        prompt=prompt,
        maxtokens=64,
        given_translations="",
        num_tries=1,
        temperature=0.2,
    )
    formula, _ = backend.call(args)
    return formula.strip()


def extract_signals(formula):
    tokens = set(re.findall(r"[a-zA-Z_][a-zA-Z0-9_]*", formula))
    reserved = {"G", "F", "X", "U", "R", "W", "M", "true", "false"}
    return sorted(t for t in tokens if t not in reserved)


def write_tlsf(inputs, outputs, formula, path):
    with open(path, "w") as f:
        f.write("INFO {\n")
        f.write('  TITLE: "auto"\n')
        f.write('  SEMANTICS: Mealy\n')
        f.write('  TARGET: Mealy\n')
        f.write("}\n\n")
        f.write("MAIN {\n")
        f.write("  INPUTS { %s; }\n" % ", ".join(inputs))
        f.write("  OUTPUTS { %s; }\n" % ", ".join(outputs))
        f.write("  GUARANTEES { %s; }\n" % formula)
        f.write("}\n")


def run(cmd):
    print("+", " ".join(cmd))
    return subprocess.run(cmd, check=True)


def ltl_to_verilog(tlsf, prefix):
    ltl = subprocess.check_output(["syfco", tlsf, "-f", "ltlxba", "-m", "fully"]).decode().strip()
    ins = subprocess.check_output(["syfco", tlsf, "--print-input-signals"]).decode().strip()
    outs = subprocess.check_output(["syfco", tlsf, "--print-output-signals"]).decode().strip()
    run(["ltlsynt", f"--formula={ltl}", f"--ins={ins}", f"--outs={outs}", f"--aiger={prefix}.aag", "--simplify=bwoa-sat"])
    run(["aigtoaig", f"{prefix}.aag", f"{prefix}.aig"])
    run(["abc", "-c", f"read_aiger {prefix}.aig; write_verilog {prefix}.v"])


def main():
    parser = argparse.ArgumentParser(description="Natural language to Verilog")
    parser.add_argument("--nl", required=True, help="requirement in natural language")
    
    parser.add_argument("--inputs", help="comma separated inputs")
    parser.add_argument("--outputs", help="comma separated outputs")
    
    parser.add_argument("--model", default="gpt-3.5-turbo", help="LLM model")
    parser.add_argument("--prompt", default="minimal", help="prompt file name in nl2spec/prompts")
    parser.add_argument("--prefix", default="controller", help="output file prefix")
    args = parser.parse_args()

    formula = nl_to_ltl(args.nl, args.model, args.prompt)
    print("LTL:", formula)

    signals = extract_signals(formula)
    inputs = args.inputs.split(",") if args.inputs else []
    outputs = args.outputs.split(",") if args.outputs else signals

    tlsf_file = args.prefix + ".tlsf"
    write_tlsf(inputs, outputs, formula, tlsf_file)

    ltl_to_verilog(tlsf_file, args.prefix)


if __name__ == "__main__":
    main()
