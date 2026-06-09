FROM pytorch/pytorch:2.5.0-cuda12.4-cudnn9-runtime

WORKDIR /workspace

COPY train_hcl.py .
COPY entrypoint.sh .
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
