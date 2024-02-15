import boto3
from botocore.config import Config
import fire


def cleanup(endpoint_name: str, region: str = "us-west-2"):
    """
    Cleanup a model from SageMaker.

    Args:
        endpoint_name: The name of the endpoint to cleanup.
        region: The AWS region to cleanup from.
    """
    config = Config(region_name=region)
    sm_client = boto3.client(service_name="sagemaker", config=config)

    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    endpoint_config_name = resp["EndpointConfigName"]
    model_name = sm_client.describe_endpoint_config(
        EndpointConfigName=endpoint_config_name
    )["ProductionVariants"][0]["ModelName"]

    sm_client.delete_endpoint(EndpointName=endpoint_name)
    sm_client.delete_endpoint_config(EndpointConfigName=endpoint_config_name)
    sm_client.delete_model(ModelName=model_name)


if __name__ == "__main__":
    fire.Fire(cleanup)
