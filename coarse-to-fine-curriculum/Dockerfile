FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ctf ./ctf
COPY scripts ./scripts
COPY tests ./tests
COPY train_coarse_to_fine.py .
COPY analyze_run.py .
COPY entrypoint.sh .
RUN chmod +x /workspace/entrypoint.sh /workspace/scripts/run_figure11_resnet18.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
