FROM pytorch/pytorch:2.12.0-cuda13.2-cudnn9-runtime

WORKDIR /workspace

COPY train_hcl.py .
COPY entrypoint.sh .
RUN chmod +x /workspace/entrypoint.sh

ENTRYPOINT ["/workspace/entrypoint.sh"]
