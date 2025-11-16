# Cloud Deployment Instructions for Azure ML

This document provides step-by-step instructions for deploying the Wine Quality model to Azure Machine Learning (AML) when you're ready to do the cloud portion of the labs.

## Prerequisites

Before starting, ensure you have:
- Azure subscription (with appropriate permissions)
- Azure ML workspace created
- Azure CLI installed and configured
- Python environment with required packages:
  ```bash
  pip install azureml-core azureml-mlflow pandas scikit-learn mlflow
  ```

## Step 1: Connect to Azure ML Workspace

### Option A: Using Configuration File
1. Go to Azure ML Studio (https://ml.azure.com)
2. Navigate to your workspace
3. Click on the workspace name in the top-right
4. Click "Download config.json"
5. Place `config.json` in your project directory

```python
from azureml.core import Workspace
ws = Workspace.from_config()
```

### Option B: Manual Connection
```python
from azureml.core import Workspace

subscription_id = 'YOUR_SUBSCRIPTION_ID'
resource_group = 'YOUR_RESOURCE_GROUP'
workspace_name = 'YOUR_WORKSPACE_NAME'
workspace_location = 'YOUR_LOCATION'  # e.g., 'eastus', 'francecentral'

ws = Workspace.create(
    name=workspace_name,
    location=workspace_location,
    resource_group=resource_group,
    subscription_id=subscription_id,
    exist_ok=True
)
```

## Step 2: Set MLFlow Tracking URI to Azure

```python
import mlflow

# Set tracking to Azure ML
mlflow.set_tracking_uri(ws.get_mlflow_tracking_uri())

# Create/set experiment
experiment_name = 'wine-quality-deployment'
mlflow.set_experiment(experiment_name)
```

## Step 3: Re-run Training with Azure Tracking

Run the training cells from the notebook. The experiments will now be tracked in Azure ML instead of locally.

## Step 4: Register the Model

After training, register your best model:

```python
from azureml.core import Experiment, Run
from azureml.core.model import Model

# Find your best run
experiment_name = 'wine-quality-deployment'
experiment = Experiment(ws, experiment_name)

# List all runs to find the best one
for run in experiment.get_runs():
    print(f"Run ID: {run.id}")
    print(f"Metrics: {run.get_metrics()}")
    print("-" * 50)

# Register the model from the best run
run_id = 'YOUR_BEST_RUN_ID'  # Replace with actual run ID
run = [r for r in experiment.get_runs() if r.id == run_id][0]

model = run.register_model(
    model_name='wine_quality_model',
    model_path='best_estimator/model.pkl'  # Or appropriate path
)

print(f"Registered model: {model.name}, version: {model.version}")
```

## Step 5: Create Inference Environment

Create a `conda.yaml` file with dependencies:

```yaml
name: wine-quality-env
channels:
  - conda-forge
dependencies:
  - python=3.8
  - pip
  - pip:
    - mlflow
    - scikit-learn==1.4.1
    - pandas==2.2.3
    - numpy==1.26.4
    - azureml-defaults==1.43
    - applicationinsights
```

Register the environment:

```python
from azureml.core import Environment

env = Environment.from_conda_specification(
    name='wine-quality-env',
    file_path='./conda.yaml'
)
env.register(ws)
```

## Step 6: Create Scoring Script

Create `score.py`:

```python
import json
import numpy as np
import pandas as pd
import joblib
from azureml.core.model import Model

def init():
    global model
    # Get the path to the registered model
    model_path = Model.get_model_path('wine_quality_model')
    model = joblib.load(model_path)
    print("Model loaded successfully")

def run(data):
    try:
        # Parse input data
        data = json.loads(data)

        # Convert to DataFrame if needed
        if 'input' in data:
            df = pd.DataFrame(json.loads(data['input']))
        else:
            df = pd.DataFrame(data)

        # Make predictions
        predictions = model.predict(df)
        probabilities = model.predict_proba(df)

        # Return results
        result = {
            'predictions': predictions.tolist(),
            'probabilities': probabilities.tolist()
        }

        return json.dumps(result)

    except Exception as e:
        error = str(e)
        return json.dumps({"error": error})
```

## Step 7: Local Deployment (Testing)

Before deploying to cloud, test locally:

```python
from azureml.core.model import InferenceConfig
from azureml.core.webservice import LocalWebservice

# Get registered model
model = Model(ws, 'wine_quality_model', version=1)  # Adjust version

# Create inference config
inference_config = InferenceConfig(
    environment=env,
    source_directory=".",
    entry_script="./score.py"
)

# Create local deployment config
deployment_config = LocalWebservice.deploy_configuration(port=6789)

# Deploy locally (requires Docker)
service = Model.deploy(
    workspace=ws,
    name='wine-quality-local',
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True
)

service.wait_for_deployment(show_output=True)
print(f"Local service URL: {service.scoring_uri}")
```

## Step 8: Deploy to Azure Container Instance (ACI)

For production-like deployment:

```python
from azureml.core.webservice import AciWebservice

# Create ACI deployment config
deployment_config = AciWebservice.deploy_configuration(
    cpu_cores=0.5,
    memory_gb=1,
    auth_enabled=False,  # Set to True for authentication
    enable_app_insights=True,  # For monitoring
    collect_model_data=True    # For data collection
)

# Deploy to ACI
service = Model.deploy(
    workspace=ws,
    name='wine-quality-aci',
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    overwrite=True
)

service.wait_for_deployment(show_output=True)
print(f"ACI service URL: {service.scoring_uri}")

# If auth is enabled, get the key
if service.auth_enabled:
    keys = service.get_keys()
    print(f"Primary key: {keys[0]}")
```

## Step 9: Test the Deployed Service

```python
import requests
import json

# Prepare test data
test_data = {
    "input": [{
        "fixed acidity": 7.4,
        "volatile acidity": 0.7,
        "citric acid": 0.0,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
    }]
}

# Convert to JSON
input_data = json.dumps(test_data)

# Set headers
headers = {'Content-Type': 'application/json'}

# If authentication is enabled
if service.auth_enabled:
    headers['Authorization'] = f'Bearer {keys[0]}'

# Make request
response = requests.post(
    service.scoring_uri,
    data=input_data,
    headers=headers
)

print(f"Status Code: {response.status_code}")
print(f"Response: {response.json()}")
```

## Step 10: Deploy to Azure Kubernetes Service (AKS)

For high-scale production deployment:

```python
from azureml.core.compute import AksCompute, ComputeTarget
from azureml.core.webservice import AksWebservice

# Create AKS cluster (one-time setup)
prov_config = AksCompute.provisioning_configuration(
    agent_count=3,
    vm_size='Standard_D3_v2'
)

aks_name = 'wine-quality-aks'
aks_target = ComputeTarget.create(
    workspace=ws,
    name=aks_name,
    provisioning_configuration=prov_config
)

aks_target.wait_for_completion(show_output=True)

# Create AKS deployment config
deployment_config = AksWebservice.deploy_configuration(
    cpu_cores=1,
    memory_gb=1,
    autoscale_enabled=True,
    autoscale_min_replicas=1,
    autoscale_max_replicas=10,
    auth_enabled=True,
    enable_app_insights=True
)

# Deploy to AKS
service = Model.deploy(
    workspace=ws,
    name='wine-quality-aks',
    models=[model],
    inference_config=inference_config,
    deployment_config=deployment_config,
    deployment_target=aks_target,
    overwrite=True
)

service.wait_for_deployment(show_output=True)
print(f"AKS service URL: {service.scoring_uri}")
```

## Step 11: Monitor and Manage

### View in Azure ML Studio
1. Go to https://ml.azure.com
2. Navigate to "Endpoints"
3. View your deployed endpoints
4. Monitor performance, logs, and metrics

### Get Service Logs
```python
# Get service logs
print(service.get_logs())

# Update service if needed
service.update(
    enable_app_insights=True,
    collect_model_data=True
)
```

### Delete Service
```python
# When no longer needed
service.delete()

# Delete AKS cluster (to save costs)
aks_target.delete()
```

## Troubleshooting

### Common Issues:

1. **Authentication Error**
   - Ensure you're logged into Azure CLI: `az login`
   - Check subscription: `az account show`

2. **Deployment Timeout**
   - Increase timeout: `service.wait_for_deployment(show_output=True, timeout_sec=1800)`

3. **Model Loading Error**
   - Check model path in `score.py`
   - Verify dependencies in `conda.yaml`

4. **Docker Issues (Local Deployment)**
   - Ensure Docker Desktop is running
   - Check Docker has enough resources allocated

## Cost Optimization

- **ACI**: Pay per second of use, good for development/testing
- **AKS**: Fixed cost for cluster, good for production with consistent load
- **Delete resources**: Always delete unused deployments and clusters

## Security Best Practices

1. Enable authentication on production endpoints
2. Use managed identities for Azure resources
3. Store secrets in Azure Key Vault
4. Enable HTTPS for endpoints
5. Monitor with Application Insights
6. Implement rate limiting

## Next Steps

1. Set up CI/CD pipeline with Azure DevOps
2. Implement model versioning strategy
3. Set up automated retraining
4. Configure alerting and monitoring
5. Implement A/B testing for model versions

## Additional Resources

- [Azure ML Documentation](https://docs.microsoft.com/azure/machine-learning/)
- [MLflow on Azure ML](https://docs.microsoft.com/azure/machine-learning/how-to-use-mlflow)
- [Deploy Models](https://docs.microsoft.com/azure/machine-learning/how-to-deploy-and-where)
- [Monitor Models](https://docs.microsoft.com/azure/machine-learning/how-to-enable-app-insights)
