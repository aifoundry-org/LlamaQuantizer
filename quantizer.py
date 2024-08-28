#!/usr/bin/env python3
import argparse
import os
import subprocess
import time

import colorama
import psutil
from colorama import Fore, Style
from colorama import init as colorama_init

# Defualt list ist of quantizations to run (w/o imatrix support)
SIMPLE_QUANT_KEY_LIST = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]

# Defualt list ist of quantizations to run (with imatrix support)
IMATRIX_QUANT_KEY_LIST = [
    "IQ1_S",
    "IQ2_S",
    "IQ1_M",
    "IQ2_M",
]

# Relative path to HF 2 GGUF converter in llamacpp repo
HF_2_GGUF_PATH = "convert_hf_to_gguf.py"
# Relative path to llama-imatrix binary in llamacpp repo
IMATRIX_PATH = "build/bin/llama-imatrix"
# Relative path to llama-quantize binary in llamacpp repo
QUANTIZER_PATH = "build/bin/llama-quantize"
# Relative path to sd binary in stable-diffusion.cpp repo
SDCPP_PATH = "build/bin/sd"


def monitor_process(pid):
    try:
        p = psutil.Process(pid)
        with p.oneshot():
            memory_info = p.memory_info()
            return memory_info.rss / (1024**3)  # bytes to GB
    except psutil.NoSuchProcess:
        return None


def run_binary(cmd_list):
    colorama_init()
    results = []
    for command in cmd_list:
        print(f"Running command: {' '.join(command)}")

        # Start the process
        start_time = time.time()
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        mem_usage = []
        stdout_lines = []
        stderr_lines = []

        # Read stdout and stderr in real-time
        while True:
            # Monitor process memory usage
            mem = monitor_process(process.pid)
            if mem is not None:
                mem_usage.append(mem)

            # Read output from stdout and stderr
            stdout_line = process.stdout.readline()
            stderr_line = process.stderr.readline()

            if stdout_line:
                print(stdout_line, end="")
                stdout_lines.append(stdout_line)
            if stderr_line:
                print(stderr_line, end="")
                stderr_lines.append(stderr_line)

            # Check if the process has finished
            if process.poll() is not None and not stdout_line and not stderr_line:
                break

            time.sleep(0.1)

        # Store the results
        result = {
            "stdout": "".join(stdout_lines),
            "stderr": "".join(stderr_lines),
            "mem_usage": mem_usage,
        }
        results.append(result)

        mem_usage_avg = sum(mem_usage) / len(mem_usage) if mem_usage else 0.0
        mem_usage_min = min(mem_usage) if mem_usage else 0.0
        mem_usage_max = max(mem_usage) if mem_usage else 0.0

        print(colorama.Fore.YELLOW + f"Completed command: {' '.join(command)}")
        print(
            "Memory usage, GB (avg/min/max): {:.3f} {:.3f} {:.3f}".format(
                mem_usage_avg, mem_usage_min, mem_usage_max
            )
        )
        print("Execution time, s:  {:.3f}".format(time.time() - start_time))
        print(
            "————————————————————————————————————————————————————————————"
            + colorama.Style.RESET_ALL
        )

    return results


def compose_llamacpp_cmd_list(args):
    if not os.path.exists(args.engine_path):
        raise FileNotFoundError("No llamacpp repo found at {}".format(args.engine_path))

    converter_path = os.path.join(args.engine_path, args.hf_to_gguf_path)
    imatrix_calc_path = os.path.join(args.engine_path, args.llama_imatrix_path)
    quantizer_path = os.path.join(args.engine_path, args.llama_quantize_path)
    if not (
        os.path.exists(converter_path)
        and os.path.exists(imatrix_calc_path)
        and os.path.exists(quantizer_path)
    ):
        raise FileNotFoundError(
            "No binaries found at {}".format(
                [converter_path, imatrix_calc_path, quantizer_path]
            )
        )

    cmd_list = []

    f16_model_path = os.path.join(args.model_dir, "{}-F16.gguf".format(args.model_name))

    if not os.path.exists(f16_model_path):
        gguf_conversion_cmd = ["python3", converter_path, args.model_dir]
        cmd_list.append(gguf_conversion_cmd)
    else:
        colorama_init()
        print(
            colorama.Fore.YELLOW
            + "File {} already exists, convert_hf_to_gguf is skipped".format(
                f16_model_path
            )
            + colorama.Style.RESET_ALL
        )

    for q in args.quant_keys:
        quant_model_path = os.path.join(
            args.model_dir, "{}-{}.gguf".format(args.model_name, q)
        )

        command = [
            quantizer_path,
            f16_model_path,
            quant_model_path,
            q,
        ]
        cmd_list.append(command)

    imatrix_path = os.path.join(
        args.model_dir, "{}-{}.dat".format(args.model_name, args.imatrix_text_name)
    )

    if args.use_imatrix:
        print(
            colorama.Fore.YELLOW
            + "Imatrix quantizations enabled"
            + colorama.Style.RESET_ALL
        )

        assert args.imatrix_text_name is not None and args.imatrix_data_path is not None

        if not os.path.exists(imatrix_path):
            imatrix_gen_cmd = [
                imatrix_calc_path,
                "-m",
                f16_model_path,
                "-f",
                args.imatrix_data_path,
                "-o",
                imatrix_path,
            ]
            cmd_list.append(imatrix_gen_cmd)
        else:
            print(
                colorama.Fore.YELLOW
                + "File {} already exists, imatrix calculation path is skipped".format(
                    imatrix_path
                )
                + colorama.Style.RESET_ALL
            )

        for q in args.imatrix_quant_keys:
            quant_model_path = os.path.join(
                args.model_dir, "{}-{}.gguf".format(args.model_name, q)
            )

            command = [
                args.quantizer_path,
                "--imatrix",
                imatrix_path,
                f16_model_path,
                quant_model_path,
                q,
            ]
            cmd_list.append(command)
    return cmd_list


def compose_sdcpp_cmd_list(args):
    cmd_list = []

    sdcpp_path = os.path.join(args.engine_path, args.scdpp_path)
    if not os.path.exists(sdcpp_path):
        raise FileNotFoundError("No binary found at {}".format(sdcpp_path))

    input_path = os.path.join(args.model_dir, "{}.safetensors".format(args.model_name))
    if not os.path.exists(input_path):
        raise FileNotFoundError("No files found at {}".format(input_path))

    for q in args.quant_keys:
        output_path = os.path.join(
            args.model_dir, "{}-{}.gguf".format(args.model_name, q)
        )
        output_type = q.lower()
        command = [
            sdcpp_path,
            "-M",
            "convert",
            "-m",
            input_path,
            "-o",
            output_path,
            "--type",
            output_type,
            "-v",
        ]
        cmd_list.append(command)

    return cmd_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM quantizations")

    parser.add_argument(
        "--engine",
        choices=["llamacpp", "sdcpp"],
        required=True,
        default="llamacpp",
        help="Choose the engine to use: 'llamacpp' or 'sdcpp'.",
    )
    parser.add_argument(
        "--use_imatrix",
        action="store_true",
        help="Use this flag to run imatrix calculation and related quantizations.",
    )
    parser.add_argument(
        "--engine_path", type=str, help="Path to built llamacpp or sdcpp repo"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="OLMo-7B-SFT-hf-0724",
        help="Name of the model.",
    )
    parser.add_argument(
        "--model_dir",
        type=str,
        help="Directory containing the model in format of HF repo.",
    )
    parser.add_argument(
        "--imatrix_text_name",
        type=str,
        default=None,
        help="Name of the text for imatrix calculation.",
    )
    parser.add_argument(
        "--imatrix_data_path",
        type=str,
        default=None,
        help="Path to the .txt file for imatrix calculation.",
    )

    parser.add_argument(
        "--quant_keys",
        type=str,
        nargs="+",
        default=SIMPLE_QUANT_KEY_LIST,
        help="List of quantization keys (without imatrix).",
    )

    parser.add_argument(
        "--imatrix_quant_keys",
        type=str,
        nargs="+",
        default=IMATRIX_QUANT_KEY_LIST,
        help="List of quantization keys (with imatrix).",
    )

    parser.add_argument(
        "--hf_to_gguf_path",
        type=str,
        default=HF_2_GGUF_PATH,
        help="Relative path to HF 2 GGUF converter in llamacpp repo.",
    )

    parser.add_argument(
        "--llama_imatrix_path",
        type=str,
        default=IMATRIX_PATH,
        help="Relative path to llama-imatrix binary in llamacpp repo.",
    )

    parser.add_argument(
        "--llama_quantize_path",
        type=str,
        default=QUANTIZER_PATH,
        help="Relative path to llama-quantize binary in llamacpp repo.",
    )

    parser.add_argument(
        "--scdpp_path",
        type=str,
        default=SDCPP_PATH,
        help="Relative path to sd binary in stable-diffusion.cpp repo.",
    )

    args = parser.parse_args()

    if args.engine == "llamacpp":
        cmd_list = compose_llamacpp_cmd_list(args)
    elif args.engine == "sdcpp":
        cmd_list = compose_sdcpp_cmd_list(args)
    else:
        raise NotImplementedError("No implemetation for {}".format(args.engine))

    _ = run_binary(cmd_list)
