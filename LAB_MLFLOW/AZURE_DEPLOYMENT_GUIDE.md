# Azure ML Deployment Guide - Quick Start

## Overview

You have successfully completed the local MLflow training. Now let's deploy to Azure ML!

## Files Created

- ✅ `config.json` - Your Azure workspace configuration
- ✅ `conda.yaml` - Environment dependencies for deployment
- ✅ `score.py` - Inference script for model serving
- ✅ `azure_deploy.py` - Automated deployment script
- ✅ `test_service.py` - Script to test deployed service
- ✅ `verify_azure_connection.py` - Verify Azure setup

## Prerequisites

### 1. Install Azure ML SDK

First, **close any running MLflow servers**, then install the required packages:

```bash
pip install azureml-core azureml-mlflow
```

If you get permission errors, try:
```bash
pip install --user azureml-core azureml-mlflow
```

### 2. Verify Azure CLI Login

Make sure you're logged into Azure:

```bash
az login
az account show
```

Verify the subscription ID matches your `config.json` file.

## Deployment Steps

### Step 1: Verify Azure Connection

Test your Azure ML workspace connection:

```bash
python verify_azure_connection.py
```

This will show:
- Workspace details
- Existing experiments
- Registered models
- Deployed services

### Step 2: Option A - Full Automated Deployment

Deploy everything with one command:

```bash
python azure_deploy.py --action all
```

This will:
1. Train the model with Azure MLflow tracking
2. Register the model in Azure ML
3. Deploy to Azure Container Instance (ACI)
4. Test the deployed service

**This process takes 10-15 minutes** (mostly deployment time)

### Step 3: Option B - Step-by-Step Deployment

If you prefer more control, run each step separately:

#### 3.1 Train and Log Model to Azure

```bash
python azure_deploy.py --action train
```

This trains a Random Forest model and logs it to Azure ML.

#### 3.2 Register the Model

```bash
python azure_deploy.py --action register
```

This registers the trained model in your Azure ML workspace.

#### 3.3 Deploy to Azure Container Instance

```bash
python azure_deploy.py --action deploy
```

This deploys the model as a REST API endpoint on ACI.

**Note**: Deployment takes 5-10 minutes.

### Step 4: Test Your Deployed Service

After deployment completes, you'll receive a scoring URI. Test it:

```bash
python test_service.py --url <YOUR_SCORING_URI>
```

Example:
```bash
python test_service.py --url http://abc123.region.azurecontainer.io/score
```

You can also test with custom samples:

```bash
# Test with 5 random samples
python test_service.py --url <YOUR_SCORING_URI> --samples 5 --random

# Test with CSV file
python test_service.py --url <YOUR_SCORING_URI> --csv data/wine_quality.csv
```

## What Gets Deployed?

### The Endpoint

Your model is deployed as a REST API that accepts JSON input:

**Input Format:**
```json
{
  "data": [[
    7.4,    // fixed acidity
    0.7,    // volatile acidity
    0.0,    // citric acid
    1.9,    // residual sugar
    0.076,  // chlorides
    11.0,   // free sulfur dioxide
    34.0,   // total sulfur dioxide
    0.9978, // density
    3.51,   // pH
    0.56,   // sulphates
    9.4     // alcohol
  ]]
}
```

**Output Format:**
```json
{
  "predictions": [1],
  "probabilities": [[0.234, 0.766]]
}
```

Where:
- `predictions`: 0 = Bad quality, 1 = Good quality
- `probabilities`: [prob_bad, prob_good]

### Manual Testing with Curl

```bash
curl -X POST \
  -H "Content-Type: application/json" \
  -d '{"data": [[7.4, 0.7, 0.0, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4]]}' \
  <YOUR_SCORING_URI>
```

## View in Azure ML Studio

1. Go to https://ml.azure.com
2. Select your workspace (LAB2)
3. Navigate to:
   - **Experiments** → See your training runs
   - **Models** → See registered models
   - **Endpoints** → See deployed services
   - **Metrics** → Monitor performance

## Common Issues & Solutions

### Issue 1: "ModuleNotFoundError: No module named 'azureml'"

**Solution:** Install Azure ML SDK:
```bash
pip install azureml-core azureml-mlflow
```

### Issue 2: "Authentication failed"

**Solution:** Login to Azure:
```bash
az login
```

### Issue 3: "Workspace not found"

**Solution:** Verify your `config.json` has correct values:
- subscription_id
- resource_group
- workspace_name

### Issue 4: "Deployment timeout"

**Solution:** Azure deployments can take time. Wait up to 15 minutes.

### Issue 5: "Service returns 503 error"

**Solution:** The service might be starting up. Wait 1-2 minutes and try again.

## Monitoring Your Service

### Get Service Logs

```python
from azureml.core import Workspace
from azureml.core.webservice import Webservice

ws = Workspace.from_config()
service = Webservice(ws, 'wine-quality-service')

# Get logs
print(service.get_logs())
```

### Check Service Status

```bash
python azure_deploy.py --action list
```

## Cost Management

### ACI Pricing
- **ACI** (Azure Container Instance): ~$0.0000125/second
- **Storage**: Minimal cost for model artifacts
- **Compute**: Only when the container is running

### Important: Clean Up Resources

When you're done testing, delete the service to avoid charges:

```python
from azureml.core import Workspace
from azureml.core.webservice import Webservice

ws = Workspace.from_config()
service = Webservice(ws, 'wine-quality-service')
service.delete()
```

Or via Azure Portal:
1. Go to Azure ML Studio
2. Navigate to Endpoints
3. Select your endpoint
4. Click "Delete"

## Next Steps (Optional)

### 1. Deploy to Azure Kubernetes Service (AKS)

For production-scale deployments with auto-scaling:
- See `CLOUD_DEPLOYMENT_INSTRUCTIONS.md` Step 10

### 2. Enable Authentication

Secure your endpoint with keys:
- Modify `azure_deploy.py` line 369: `auth_enabled=True`

### 3. Set Up CI/CD

Automate deployments with Azure DevOps or GitHub Actions.

### 4. Monitor with Application Insights

Your deployment already has App Insights enabled! View metrics in Azure Portal.

## Troubleshooting Commands

```bash
# Check Azure CLI version
az --version

# Check current subscription
az account show

# List all workspaces
az ml workspace list

# Check Python packages
pip list | grep azure

# Test workspace connection
python verify_azure_connection.py
```

## Support

If you encounter issues:

1. Check the error message carefully
2. Review the troubleshooting section above
3. Check Azure ML documentation: https://docs.microsoft.com/azure/machine-learning/
4. Review your workspace in Azure ML Studio

## Summary

✅ You have created all necessary files for Azure deployment
✅ Your `config.json` is configured with workspace details
✅ Run `python azure_deploy.py --action all` to deploy

**Total time for full deployment: ~15 minutes**

Good luck with your deployment! 🚀
