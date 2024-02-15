import re
import boto3
import fire
import random
import string
import json
from typing import Literal


def generate_random_string(length):
    characters = string.ascii_letters + string.digits
    return "".join(random.choice(characters) for _ in range(length))


def get_sagemaker_vars(region, base_image_name):
    account_id = boto3.client("sts").get_caller_identity()["Account"]
    return (
        f"{account_id}.dkr.ecr.{region}.amazonaws.com/{base_image_name}:latest",
        f"arn:aws:iam::{account_id}:role/SMRole",
    )


def print_color(text, color):
    colors = {
        "red": "\033[91m",
        "green": "\033[92m",
        "yellow": "\033[93m",
        "blue": "\033[94m",
        "purple": "\033[95m",
        "cyan": "\033[96m",
        "white": "\033[97m",
    }
    print(colors[color] + text + "\033[0m")


def read_json_file(config_path: str):
    with open(config_path, "r") as file:
        data = json.load(file)
    return data


def get_value(config_data, cli_value, key):
    """
    Get the value from the CLI if it exists, otherwise use the config value.
    """
    if cli_value is not None:
        print_color(f"Using {key} from CLI: {cli_value}", "blue")
        return cli_value
    elif config_data is not None:
        print_color(f"Using {key} from config: {config_data.get(key, None)}", "yellow")
        return config_data.get(key, None)
    else:
        return None


def deploy(
    config_path: str,
    model: str = None,
    endpoint_name: str = None,
    image: str = None,
    instance_type: str = None,
    tokenization_workers: int = None,
    dtype: Literal["float16", "float32"] = None,
    pooling: Literal["cls", "mean"] = None,
    max_concurrent_requests: int = None,
    max_batch_tokens: int = None,
    max_batch_requests: int = None,
    max_client_batch_size: int = None,
    hf_api_token: str = None,
    hostname: str = None,
    port: str = None,
    uds_path: str = None,
    huggingface_hub_cache: str = None,
    json_output: bool = None,
    otlp_endpoint: str = None,
    cors_allow_origin: str = None,
    region: str = "us-west-2",
):
    """
    Deploy a TEI model to SageMaker.

    Args:
        config_path: Path to a json file containing model configuration.
        model: The model name to deploy (i.e. `mistralai/Mistral-7B-Instruct-v0.2`).
        endpoint_name: The name to give to the created SageMaker endpoint.
        image: The Docker image to use for inference.
        instance_type: The instance type to deploy to.
        tokenization_workers: Optionally control the number of tokenizer workers used for payload tokenization, validation and truncation. Default to the number of CPU cores on the machine
        dtype: The dtype to be forced upon the model.
        pooling: Optionally control the pooling method for embedding models.
          If `pooling` is not set, the pooling configuration will be parsed from the model `1_Pooling/config.json` configuration.
          If `pooling` is set, it will override the model pooling configuration.
        max_concurrent_requests: The maximum amount of concurrent requests for this particular deployment.
          Having a low limit will refuse clients requests instead of having them wait for too long and is usually good
          to handle backpressure correctly
        max_batch_tokens:**IMPORTANT** This is one critical control to allow maximum usage of the available hardware.
          This represents the total amount of potential tokens within a batch.
          For `max_batch_tokens=1000`, you could fit `10` queries of `total_tokens=100` or a single query of `1000` tokens.
          Overall this number should be the largest possible until the model is compute bound. Since the actual memory overhead depends on the model implementation, text-embeddings-inference cannot infer this number automatically.
        max_batch_requests: Optionally control the maximum number of individual requests in a batch.
        max_client_batch_size: Control the maximum number of inputs that a client can send in a single request.
        hf_api_token: Your HuggingFace hub token.
        hostname: The IP address to listen on.
        port: The port to listen on.
        uds_path: The name of the unix socket some text-embeddings-inference backends will use as they communicate internally with gRPC.
        huggingface_hub_cache: The location of the huggingface hub cache. Used to override the location if you want to provide a mounted disk for instance.
        json_output: Outputs the logs in JSON format (useful for telemetry).
        otlp_endpoint:
        cors_allow_origin:
        region: The AWS region to deploy to.
    """
    config_data = read_json_file(config_path)
    model = get_value(config_data, model, "model")
    image = get_value(config_data, image, "image")
    instance_type = get_value(config_data, instance_type, "sagemaker_instance_type")

    env_params = {
        "TOKENIZATION_WORKERS": get_value(config_data, tokenization_workers, "tokenization_workers"),
        "DTYPE": get_value(config_data, dtype, "dtype"),
        "POOLING": get_value(config_data, pooling, "pooling"),
        "MAX_CONCURRENT_REQUESTS": get_value(config_data, max_concurrent_requests, "max_concurrent_requests"),
        "MAX_BATCH_TOKENS": get_value(config_data, max_batch_tokens, "max_batch_tokens"),
        "MAX_BATCH_REQUESTS": get_value(config_data, max_batch_requests, "max_batch_requests"),
        "MAX_CLIENT_BATCH_SIZE": get_value(config_data, max_client_batch_size, "max_client_batch_size"),
        "HF_API_TOKEN": get_value(config_data, hf_api_token, "hf_api_token"),
        "HOSTNAME": get_value(config_data, hostname, "hostname"),
        "PORT": get_value(config_data, port, "port"),
        "UDS_PATH": get_value(config_data, uds_path, "uds_path"),
        "HUGGINGFACE_HUB_CACHE": get_value(config_data, huggingface_hub_cache, "huggingface_hub_cache"),
        "JSON_OUTPUT": get_value(config_data, json_output, "json_output"),
        "OTLP_ENDPOINT": get_value(config_data, otlp_endpoint, "otlp_endpoint"),
        "CORS_ALLOW_ORIGIN": get_value(config_data, cors_allow_origin, "cors_allow_origin")
    }

    random_id = generate_random_string(5)
    name = model.split("/")[-1] if endpoint_name is None else endpoint_name
    endpoint_name = "tei-" + re.sub("[^0-9a-zA-Z]", "-", name) + "-" + random_id
    model_name = f"{endpoint_name}-mdl"
    endpoint_config_name = f"{endpoint_name}-epc"

    # get sagemaker image and role
    tei_image_uri, role = get_sagemaker_vars(region, image)

    assert model is not None

    container_env = {
        "MODEL_ID": model
    }
    for key, value in env_params.items():
        if value is not None:
            container_env[key] = str(value)

    print("\nThis configuration will be applied: ")
    print_color(
        json.dumps(
            {
                "container_env": container_env,
                "instance_type": instance_type,
                "sagemaker_endpoint": endpoint_name,
                "sagemaker_model": model_name,
                "sagemaker_endpoint_config": endpoint_config_name,
                "region": region,
                "image_uri": tei_image_uri,
            },
            indent=4,
        ),
        "green",
    )

    # create model
    sm_client = boto3.client(service_name="sagemaker", region_name=region)
    create_model_response = sm_client.create_model(
        ModelName=model_name,
        ExecutionRoleArn=role,
        PrimaryContainer={"Image": tei_image_uri, "Environment": container_env},
    )
    print("Model Arn: " + create_model_response["ModelArn"])

    # create endpoint configuration
    create_endpoint_config_response = sm_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                "InstanceType": instance_type,
                "InitialVariantWeight": 1,
                "InitialInstanceCount": 1,
                "ModelName": model_name,
                "VariantName": "AllTraffic",
            }
        ],
    )
    print(
        "Endpoint Config Arn: " + create_endpoint_config_response["EndpointConfigArn"]
    )

    # create endpoint
    create_endpoint_response = sm_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name,
    )

    print(f"Waiting for {endpoint_name} endpoint to be in service...")
    waiter = sm_client.get_waiter("endpoint_in_service")
    waiter.wait(EndpointName=endpoint_name)

    resp = sm_client.describe_endpoint(EndpointName=endpoint_name)
    print("Endpoint Arn: " + create_endpoint_response["EndpointArn"])
    print("Endpoint Status: " + resp["EndpointStatus"])
    print("=" * 20)
    print("Endpoint name: " + endpoint_name)
    print("=" * 20)

    return endpoint_name


if __name__ == "__main__":
    fire.Fire(deploy)
