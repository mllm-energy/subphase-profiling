1. Clone the VLLM repository
    # Clone the official vLLM source to allow for code modifications
    git clone https://github.com/vllm-project/vllm.git
    cd vllm

2. Virtual Environment Setup 
    We use a dedicated venv to prevent dependency conflicts between vLLM and the system.
    Here are the commands you need to run to set this up:
        - python3 -m venv <whatever name you want for the virtual environment>
        - source <name you chose above>/bin/activate
        - pip install --upgrade pip

4. Library Installation
    These are the specific libraries required to run InternVL3 within the vLLM engine.
        - pip install -e .
        - pip install torch torchvision torchaudio
        - pip install timm einops pillow      

5. Model Code Modifications
    To enable precise profiling of sub-phases, we modified the vLLM source code to include NVTX markers. 
    This allows the profiler to "see" where the model forward pass begins and ends.

    Paste the file I added into the folder named gpu_model_runner.py over the gpu_model_runner.py file 
    in the vllm repository you cloned. The path is specified below:

    File -> vllm/vllm//gpu_model_runner.py

    These are the modifications:

        # ADDED LINE BY ME 
        torch.cuda.nvtx.range_push("vLLM_Inference_Step")

        model_output = self._model_forward(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            **model_kwargs,
        )

        # ADDED LINE BY ME
        torch.cuda.nvtx.range_pop()

6. Profiling Script
    I created a specialized script which the profiler uses to run the closed-loop request cycle, and generate outputs. 

    This script will also be in this folder called profiler_subphases.py
    Make sure to add it in a separate folder from the vllm directory. 

7. Command for profiling    
    Finally, I ran the command that ran the NCU profiler.
    I used sudo to bypass Linux security restrictions that would cause crashes.

    Here is the script:

    sudo $(which ncu) --target-processes all \
        --replay-mode kernel \
        --nvtx \
        --nvtx-range "CLOSED_LOOP_INFERENCE" \
        --metrics sm__throughput.avg.pct_of_peak_sustained_active,dram__throughput.avg.pct_of_peak_sustained_elapsed \
        -o InternVL_Hardware_Profile \
        python3 profile_subphases.py

8. Exporting to CSV

    The command above produced a binary .ncu-rep file. 
    I then used the NCU export tool to turn that into the existing_data.csv we used for analysis.
    Make sure to download the NVIDIA nsight compute software so that you can get the CSV as well.
    
    Since our main goal is to find each subphase separately we need to add the NTVX hooks in different spots than I did



    
