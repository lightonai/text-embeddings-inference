# Deploy a model to SageMaker

## Install dependencies

```bash
pip install -r requirements-utils.txt
```

## Deploy a model

Deploy a model to SageMaker:

```bash
python deploy/tei-deploy.py --config_path deploy/configs/multilingual-e5-large.json
```

> You can create more models configs in the `deploy/configs` folder.

To clean up the SageMaker resources:

```bash
python deploy/cleanup.py --endpoint_name <endpoint_name>
```
