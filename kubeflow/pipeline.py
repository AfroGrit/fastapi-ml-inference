from kfp import dsl

@dsl.pipeline(
    name="ML Inference Pipeline",
    description="A simple Kubeflow pipeline for inference."
)
def inference_pipeline():
    with dsl.ContainerOp(
        name="Run Inference",
        image="ghcr.io/AfroGrit/fastapi-ml:latest",
        command=["python", "run_inference.py"]
    ) as op:
        op.set_gpu_limit(1)

if __name__ == "__main__":
    from kfp.compiler import Compiler
    Compiler().compile(inference_pipeline, "pipeline.yaml")

