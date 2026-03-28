# Brev Setup

Go to [https://brev.nvidia.com/](https://brev.nvidia.com/) -> Sign up / Log in.

Create new instance -> Select GPU (A100) -> Select model (all similar) -> click VM Mode w/ Jupyter -> Docker Compose -> GitHub URL: `https://github.com/Borg-Robotics/borg_gr00t/blob/main/docker/cloud_compose.yaml` -> Validate -> Save and Continue -> Deploy.

Replace `awesome-gpu-name` with your instance name and run:
```bash
brev shell awesome-gpu-name
```

Clone the repo and enter it:
```bash
mkdir -p workspace
cd workspace
git clone https://github.com/Borg-Robotics/borg_gr00t.git
cd borg_gr00t
```

Rebuild the docker image in detached mode (`-d`):
```bash
docker compose -f docker/brev_compose.yaml up --build -d
```

Enter it with:
```bash
docker exec -it brev-borg-gr00t bash
```

Download the dataset with
```bash
gdown --folder https://drive.google.com/drive/folders/1qDykMJSplterueCXxjLe13bdGWcxw0HC
mkdir -p ./data
mv ./box_pickup_dataset ./data/box_pickup_dataset
chown -R 1002 ./data
```
note that `chown` assumes the outside user id is 1002, change accordingly if it is not (check with `id -u` **outside** of Docker). You will have permissions errors in vscode otherwise.

Run the training with
```bash
python scripts/finetune.py \
    --base-model-path nvidia/GR00T-N1.6-3B \
    --dataset-path /workspace/data/box_pickup_dataset \
    --num-gpus 1 \
    --max-steps 500 \
    --output-dir /tmp/borg-finetune
```

Run the server/client inference in two different terminals:
```bash
python scripts/inference_service.py \
    --server \
    --model-path /tmp/borg-finetune \
    --embodiment-tag borg \
    --port 5556
```

```bash
python scripts/inference_service.py \
    --client \
    --embodiment-tag borg \
    --port 5556
```
This will tell you if everything is working correctly and the model is able to run inference.

## Evaluate Fine-Tuned Model

This replays an unseen test episode through the fine-tuned model and compares predicted actions against ground truth.

Start the inference server (in one terminal):
```bash
python scripts/inference_server_zmq.py \
    --server \
    --model-path /tmp/borg-finetune \
    --embodiment-tag borg \
    --port 5556
```

Run the evaluation (in another terminal):
```bash
python scripts/evaluate_replay.py \
    --dataset-path /workspace/data/box_pickup_dataset \
    --episode-id 000005 \
    --port 5556
```

This outputs per-joint MAE metrics and saves a comparison plot.

## Test Remote Connection (Local -> Brev)

This verifies ZMQ connectivity from a local machine to the Brev instance.

On Brev (inside Docker), start the echo server:
```bash
python scripts/server_echo.py
```

On your local machine, forward local port 5557 to Brev port 5556 (replace `awesome-gpu-name2` with your instance name):
```bash
brev port-forward awesome-gpu-name2 -p 5557:5556
```

On your local machine, run the echo client:
```bash
python scripts/client_echo.py
```

## VS Code Remote Development

Open VS Code in the brev instance with (replace `awesome-gpu-name` with your instance name):
```bash
brev open awesome-gpu-name code
```

Install the Dev Containers extension in VS Code.

- CTRL+SHIFT+P ->
- Search for `Dev Containers: Attach to Running Container` and select it ->
- Select the brev docker image (`brev-borg-gr00t` which starts once you run the docker compose command above) ->
- Open the `/workspace` folder.
