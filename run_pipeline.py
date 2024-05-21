import subprocess
import yaml

def run_stage(stage):
    cmd = ["python", stage["script"]]
    for arg, val in stage.get("arguments", {}).items():
        cmd.extend(["--" + arg, str(val)])
    subprocess.run(cmd, check=True)

if __name__ == "__main__":
    with open("pipeline.yaml", "r") as f:
        pipeline = yaml.safe_load(f)

    for stage in pipeline["stages"]:
        print(f"Running stage: {stage['name']}")
        run_stage(stage)