#!/usr/bin/env python3
import os
import subprocess
import time

import colorama
import psutil
from colorama import Fore, Style
from colorama import init as colorama_init


def monitor_process(pid):
    try:
        p = psutil.Process(pid)
        with p.oneshot():
            cpu_usage = p.cpu_percent(interval=1)
            memory_info = p.memory_info()
            return cpu_usage, memory_info.rss / (1024**3)  # bytes to GB
    except psutil.NoSuchProcess:
        return None, None


def run_binary(cmd_list):
    colorama_init()
    results = []
    for command in cmd_list:
        print(f"Running command: {' '.join(command)}")

        # Start the process
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
        )

        cpu_usage, mem_usage = [], []
        stdout_lines = []
        stderr_lines = []

        # Read stdout and stderr in real-time
        while True:
            # Monitor process CPU and memory usage
            cpu, mem = monitor_process(process.pid)
            if cpu is not None:
                cpu_usage.append(cpu)
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
            "cpu_usage": cpu_usage,
            "mem_usage": mem_usage,
        }
        results.append(result)

        cpu_usage_avg = sum(cpu_usage) / len(cpu_usage) if cpu_usage else 0.0
        cpu_usage_min = min(cpu_usage) if cpu_usage else 0.0
        cpu_usage_max = max(cpu_usage) if cpu_usage else 0.0

        mem_usage_avg = sum(mem_usage) / len(mem_usage) if mem_usage else 0.0
        mem_usage_min = min(mem_usage) if mem_usage else 0.0
        mem_usage_max = max(mem_usage) if mem_usage else 0.0

        print(colorama.Fore.YELLOW + f"Completed command: {' '.join(command)}")
        print(
            "CPU usage (avg/min/max): {:.2f} {:.2f} {:.2f}".format(
                cpu_usage_avg, cpu_usage_min, cpu_usage_max
            )
        )
        print(
            "Memory usage, GB (avg/min/max): {:.3f} {:.3f} {:.3f}".format(
                mem_usage_avg, mem_usage_min, mem_usage_max
            )
        )
        print(
            "————————————————————————————————————————————————————————————"
            + colorama.Style.RESET_ALL
        )

    return results


if __name__ == "__main__":

    # args
    convertion_path = "./llama.cpp/convert_hf_to_gguf.py"
    binary_path = "./llama.cpp/build/bin/llama-quantize"
    model_name = "OLMo-1B-hf-0724"
    model_dir = "./OLMo-1B-0724-hf/"

    f16_model_path = os.path.join(model_dir, "{}-F16.gguf".format(model_name))

    cmd_list = [["python3", convertion_path, model_dir]]

    quantization_keys = ["Q2_K", "Q3_K", "Q4_K", "Q5_K", "Q6_K", "Q8_0"]

    for q in quantization_keys:
        quant_model_path = os.path.join(model_dir, "{}-{}.gguf".format(model_name, q))

        command = [
            binary_path,
            f16_model_path,
            quant_model_path,
            q,
        ]
        cmd_list.append(command)

    print(cmd_list)

    results = run_binary(cmd_list)
