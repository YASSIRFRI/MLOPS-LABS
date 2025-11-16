"""
Azure ML Deployment Script for Wine Quality Model

This script automates the deployment of the Wine Quality model to Azure ML.
It handles:
1. Workspace connection
2. MLflow tracking configuration
3. Model training and logging to Azure
4. Model registration
5. Model deployment to ACI

Usage:
    python azure_deploy.py --action train      # Train and log model to Azure
    python azure_deploy.py --action register   # Register the best model
    python azure_deploy.py --action deploy     # Deploy to ACI
    python azure_deploy.py --action all        # Do all steps
"""

import argparse
import json
import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

# Azure ML imports
from azureml.core import Workspace, Environment, Experiment, Model
from azureml.core.model import InferenceConfig
from azureml.core.webservice import AciWebservice, Webservice

# ML imports
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
import joblib


class AzureMLDeployer:
    def __init__(self, config_path='config.json'):
        """Initialize Azure ML workspace connection"""
        print("="*60)
        print("Azure ML Deployment for Wine Quality Model")
        print("="*60)

        # Load config
        with open(config_path, 'r') as f:
            config = json.load(f)

        print(f"\nConnecting to Azure ML workspace...")
        print(f"Subscription: {config['subscription_id']}")
        print(f"Resource Group: {config['resource_group']}")
        print(f"Workspace: {config['workspace_name']}")

        # Connect to workspace
        try:
            self.ws = Workspace(
                subscription_id=config['subscription_id'],
                resource_group=config['resource_group'],
                workspace_name=config['workspace_name']
            )
            print(f"[OK] Connected to workspace: {self.ws.name}")
            print(f"  Location: {self.ws.location}")
        except Exception as e:
            print(f"[ERROR] Error connecting to workspace: {e}")
            sys.exit(1)

        # Set MLflow tracking to Azure
        mlflow.set_tracking_uri(self.ws.get_mlflow_tracking_uri())
        print(f"[OK] MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

        # Create/set experiment
        self.experiment_name = 'wine-quality-azure-deployment'
        mlflow.set_experiment(self.experiment_name)
        print(f"[OK] Experiment set to: {self.experiment_name}")

    def load_and_prepare_data(self):
        """Load and prepare the wine quality dataset"""
        print("\n" + "="*60)
        print("Loading and Preparing Data")
        print("="*60)

        # Check if data file exists, if not download it
        data_path = 'data/wine_quality.csv'
        if os.path.exists(data_path):
            print(f"Loading data from: {data_path}")
            df = pd.read_csv(data_path)
        else:
            print("Downloading Wine Quality dataset...")
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
            df = pd.read_csv(url, sep=';')

            # Create data directory if it doesn't exist
            os.makedirs('data', exist_ok=True)
            df.to_csv(data_path, index=False)
            print(f"[OK] Data saved to: {data_path}")

        print(f"Dataset shape: {df.shape}")

        # Create binary classification target
        df['quality_binary'] = (df['quality'] >= 6).astype(int)

        # Prepare features and target
        X = df.drop(['quality', 'quality_binary'], axis=1)
        y = df['quality_binary']

        print(f"Features: {list(X.columns)}")
        print(f"Target distribution: {y.value_counts().to_dict()}")

        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=2020
        )

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        print(f"[OK] Training set: {X_train_scaled.shape}")
        print(f"[OK] Test set: {X_test_scaled.shape}")

        # Save scaler for later use
        os.makedirs('models', exist_ok=True)
        joblib.dump(scaler, 'models/scaler.pkl')

        return X_train_scaled, X_test_scaled, y_train, y_test

    def train_and_log_model(self):
        """Train model and log to Azure ML"""
        print("\n" + "="*60)
        print("Training and Logging Model to Azure ML")
        print("="*60)

        # Load data
        X_train, X_test, y_train, y_test = self.load_and_prepare_data()

        # Enable MLflow autologging (this will handle all logging)
        mlflow.sklearn.autolog(
            log_input_examples=False,
            log_model_signatures=False,
            registered_model_name="wine_quality_model"
        )

        # Start MLflow run
        with mlflow.start_run(run_name=f"RF_Training_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            print(f"\nMLflow Run ID: {run.info.run_id}")

            # Train Random Forest
            print("\nTraining Random Forest Classifier...")
            params = {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'random_state': 2020
            }

            model = RandomForestClassifier(**params)
            model.fit(X_train, y_train)

            # Evaluate
            train_acc = model.score(X_train, y_train)
            test_acc = model.score(X_test, y_test)

            y_pred = model.predict(X_test)
            test_auc = roc_auc_score(y_test, y_pred)

            print(f"\nResults:")
            print(f"  Train Accuracy: {train_acc:.4f}")
            print(f"  Test Accuracy: {test_acc:.4f}")
            print(f"  Test AUC: {test_auc:.4f}")

            # Save model locally
            model_path = 'models/wine_quality_model.pkl'
            joblib.dump(model, model_path)
            print(f"\n[OK] Model saved locally to: {model_path}")
            print(f"[OK] Model logged to Azure ML (via autologging)")

            self.latest_run_id = run.info.run_id

        print(f"\n[OK] Training completed successfully!")
        print(f"  Run ID: {self.latest_run_id}")
        print(f"\nView in Azure ML Studio:")
        print(f"  {self.ws.get_portal_url()}")

        return self.latest_run_id

    def register_model(self, run_id=None):
        """Register the model in Azure ML"""
        print("\n" + "="*60)
        print("Registering Model in Azure ML")
        print("="*60)

        if run_id is None:
            # Find the latest run
            experiment = Experiment(self.ws, self.experiment_name)
            runs = list(experiment.get_runs())

            if not runs:
                print("[ERROR] No runs found in experiment")
                return None

            # Get the latest run
            latest_run = runs[0]
            run_id = latest_run.id
            print(f"Using latest run: {run_id}")

        try:
            # Check if model is already registered
            model_name = 'wine_quality_model'

            try:
                existing_model = Model(self.ws, model_name)
                print(f"Model '{model_name}' already registered (Version: {existing_model.version})")
                return existing_model
            except:
                print(f"Model '{model_name}' not yet registered, registering now...")

            # Register model from local file
            model_path = 'models/wine_quality_model.pkl'

            if not os.path.exists(model_path):
                print(f"[ERROR] Model file not found at: {model_path}")
                print("  Please run training first: python azure_deploy.py --action train")
                return None

            model = Model.register(
                workspace=self.ws,
                model_name=model_name,
                model_path=model_path,
                description="Wine Quality Prediction Model (Random Forest)",
                tags={
                    'framework': 'scikit-learn',
                    'type': 'RandomForest',
                    'dataset': 'Wine Quality',
                    'run_id': run_id
                }
            )

            print(f"[OK] Model registered successfully!")
            print(f"  Name: {model.name}")
            print(f"  Version: {model.version}")
            print(f"  ID: {model.id}")

            return model

        except Exception as e:
            print(f"[ERROR] Error registering model: {e}")
            return None

    def deploy_to_aci(self, model_name='wine_quality_model'):
        """Deploy model to Azure Container Instance"""
        print("\n" + "="*60)
        print("Deploying Model to Azure Container Instance")
        print("="*60)

        try:
            # Get the registered model
            model = Model(self.ws, model_name)
            print(f"Using model: {model.name} (Version: {model.version})")

            # Create environment from conda file
            print("\nCreating environment from conda.yaml...")
            env = Environment.from_conda_specification(
                name='wine-quality-env',
                file_path='./conda.yaml'
            )

            # Create inference config
            print("Creating inference configuration...")
            inference_config = InferenceConfig(
                environment=env,
                source_directory="deploy",
                entry_script="score.py"
            )

            # Create deployment config
            print("Creating deployment configuration...")
            deployment_config = AciWebservice.deploy_configuration(
                cpu_cores=1,
                memory_gb=1,
                auth_enabled=False,
                enable_app_insights=True,
                collect_model_data=True,
                description="Wine Quality Prediction Service"
            )

            # Deploy
            service_name = 'wine-quality-service'
            print(f"\nDeploying to ACI (service name: {service_name})...")
            print("This may take several minutes...")

            # Check if service already exists
            try:
                existing_service = Webservice(self.ws, service_name)
                print(f"Service '{service_name}' already exists. Updating...")
                existing_service.update(
                    models=[model],
                    inference_config=inference_config
                )
                service = existing_service
            except:
                print(f"Creating new service '{service_name}'...")
                service = Model.deploy(
                    workspace=self.ws,
                    name=service_name,
                    models=[model],
                    inference_config=inference_config,
                    deployment_config=deployment_config,
                    overwrite=True
                )

            service.wait_for_deployment(show_output=True)

            print(f"\n[OK] Deployment successful!")
            print(f"  Service name: {service.name}")
            print(f"  Scoring URI: {service.scoring_uri}")
            print(f"  State: {service.state}")

            # Test the service
            self.test_deployment(service)

            return service

        except Exception as e:
            print(f"[ERROR] Error during deployment: {e}")
            import traceback
            traceback.print_exc()
            return None

    def test_deployment(self, service):
        """Test the deployed service"""
        print("\n" + "="*60)
        print("Testing Deployed Service")
        print("="*60)

        import requests

        # Prepare test data
        test_data = {
            "data": [[
                7.4,    # fixed acidity
                0.7,    # volatile acidity
                0.0,    # citric acid
                1.9,    # residual sugar
                0.076,  # chlorides
                11.0,   # free sulfur dioxide
                34.0,   # total sulfur dioxide
                0.9978, # density
                3.51,   # pH
                0.56,   # sulphates
                9.4     # alcohol
            ]]
        }

        input_data = json.dumps(test_data)

        print(f"\nSending test request to: {service.scoring_uri}")
        print(f"Input data: {test_data}")

        try:
            headers = {'Content-Type': 'application/json'}
            response = requests.post(
                service.scoring_uri,
                data=input_data,
                headers=headers
            )

            print(f"\nResponse status: {response.status_code}")
            if response.status_code == 200:
                result = response.json()
                print(f"Prediction result: {result}")
                print("\n[OK] Service is working correctly!")
            else:
                print(f"[ERROR] Error: {response.text}")

        except Exception as e:
            print(f"[ERROR] Error testing service: {e}")

    def list_deployments(self):
        """List all deployed services"""
        print("\n" + "="*60)
        print("Deployed Services")
        print("="*60)

        services = Webservice.list(self.ws)

        if not services:
            print("No services deployed")
            return

        for service in services:
            print(f"\nService: {service.name}")
            print(f"  Type: {service.type}")
            print(f"  State: {service.state}")
            print(f"  Scoring URI: {service.scoring_uri}")


def main():
    parser = argparse.ArgumentParser(description='Deploy Wine Quality Model to Azure ML')
    parser.add_argument('--action', type=str, required=True,
                       choices=['train', 'register', 'deploy', 'list', 'all'],
                       help='Action to perform')
    parser.add_argument('--config', type=str, default='config.json',
                       help='Path to config.json file')

    args = parser.parse_args()

    # Initialize deployer
    deployer = AzureMLDeployer(config_path=args.config)

    # Perform action
    if args.action == 'train':
        deployer.train_and_log_model()

    elif args.action == 'register':
        deployer.register_model()

    elif args.action == 'deploy':
        deployer.deploy_to_aci()

    elif args.action == 'list':
        deployer.list_deployments()

    elif args.action == 'all':
        print("\nPerforming full deployment pipeline...")

        # Step 1: Train
        run_id = deployer.train_and_log_model()

        # Step 2: Register
        model = deployer.register_model(run_id)

        if model is None:
            print("\n[ERROR] Model registration failed. Aborting deployment.")
            return

        # Step 3: Deploy
        service = deployer.deploy_to_aci(model.name)

        if service:
            print("\n" + "="*60)
            print("[OK] DEPLOYMENT COMPLETED SUCCESSFULLY!")
            print("="*60)
            print(f"\nYour model is now deployed and accessible at:")
            print(f"  {service.scoring_uri}")
            print(f"\nTo test the service, use:")
            print(f"  python test_service.py --url {service.scoring_uri}")

    print("\n" + "="*60)
    print("Done!")
    print("="*60)


if __name__ == "__main__":
    main()
