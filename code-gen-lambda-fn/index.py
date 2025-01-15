import logging
import tools_final

# Configure logger for AWS Lambda
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def handler(event, context):
    # Log the received event
    logger.info("Received event: %s", event)

    # Initialize response code to None
    response_code = None

    # Extract the action group, api path, and parameters
    action = event["actionGroup"]
    api_path = event["apiPath"]
    parameters = event["parameters"]
    inputText = event["inputText"]
    httpMethod = event["httpMethod"]

    logger.info("inputText: %s", inputText)

    # Get the query value from the parameters
    query = parameters[0]["value"]
    logger.info("Query: %s", query)

    if api_path == "/gen_code":
        # Call the code_gen_tool from the tools module
        body = tools_final.gen_code(query)
        response_body = {"application/json": {"body": str(body)}}
        response_code = 200
    else:
        # Unrecognized api path
        body = {"{}::{} is not a valid api, try another one.".format(action, api_path)}
        response_body = {"application/json": {"body": str(body)}}
        response_code = 400

    # Log the response body
    logger.info("Response body: %s", response_body)

    # Create a dictionary containing the response details
    action_response = {
        "actionGroup": action,
        "apiPath": api_path,
        "httpMethod": httpMethod,
        "httpStatusCode": response_code,
        "responseBody": response_body,
    }

    # Return the final response
    api_response = {"messageVersion": "1.0", "response": action_response}
    return api_response
