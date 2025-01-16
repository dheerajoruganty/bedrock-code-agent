import os
import re
import time
import json
import weave
import random
import logging
from pathlib import Path
from litellm import completion
from litellm import RateLimitError
import boto3
import uuid

# create an boto3 bedrock agent client
client = boto3.client("bedrock-agent-runtime")

# Setup logging
logging.basicConfig(format='[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s', level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Litellm needs region to be set
os.environ["AWS_REGION_NAME"] = "us-east-1"

# sometimes the model refused to generate content due to internal guardrails
# so this is a custom exception to catch that error
class NoContentGeneratedException(Exception):
    pass

# Canned code to use in case the model generates misformatted code
# so this code will also cause the test to fail so in that sense the overall
# accuracy of the model for this benchmark remain unaffected and this code simply
# helps the harness to move to the next problem in the benchmark
FAILED_RESPONSE = """
import sys

def main():
    input = sys.stdin.read
    data = input().split()
    
    # do nothing, this is a canned response so that the eval for
    # this task can silently fail

    print(data)

if __name__ == "__main__":
    main()
"""

# Regular expression pattern to extract Python code blocks from text
# Matches content between ```python and ``` markers, capturing everything in between
REGEX_FOR_PY_CODE_EXTRACTION: str = r"```python\n(.*?)```"

def _get_python_code(
    text: str, 
    regex_for_code_extraction: str = REGEX_FOR_PY_CODE_EXTRACTION,
    failure_response: str = FAILED_RESPONSE
) -> str:
    """
    Extracts Python code from text that contains markdown-style code blocks.
    
    Args:
        text (str): The input text containing Python code blocks
        regex_for_code_extraction (str): Regular expression pattern to match code blocks
            Defaults to REGEX_FOR_PY_CODE_EXTRACTION
        failure_response (str): Response to return if no code is found
            Defaults to FAILED_RESPONSE
    
    Returns:
        str: The extracted Python code if found, otherwise returns the failure_response
    
    Note:
        - Expects code to be formatted in markdown style with ```python and ``` markers
        - Uses re.DOTALL flag to match newlines within the code block
    """
    # Search for all matches of the regex pattern in the text
    # re.DOTALL allows the dot (.) to match newline characters
    matches = re.findall(regex_for_code_extraction, text, re.DOTALL)
    
    # If no matches found, log an error and return the failure response
    if matches is None:
        logger.error(
            f"no python code found in {text}, returning the canned failure response\n{failure_response}"
        )
        return failure_response
    
    # Return the first code block found (matches[0] contains the content between markers)
    return matches[0]

def run(input: dict[str, dict], **kwargs) -> dict[str, str]:
    """
    Runs inference for each task in the benchmark.
    
    Args:
        input (dict): dictionary containing info for the tasks being run
        kwargs (dict): model name, prompt template path and other params needed for inference
    Returns:
        dict: Input dictionary with the "response" (completion) field added
    
    """
    assert "model_name" in kwargs, "model_name is required"
    logger.info(f"model Name={kwargs['model_name']}, prompt_template_path={kwargs.get('prompt_template_path')}")

    # result for each task
    results: dict = {}
    # generated code for each task
    code: dict = {}

    # Iterate through all the tasks and get responses from each task
    for task_id, task in input.items():
        task_id = 1
        # Debug: logger.info task details        
        logger.info(f"Processing Task ID: {task_id}, Task: {json.dumps(task, indent=2)}")
        prompt_template = ""
        template_path = kwargs["prompt_template_path"]
        if os.path.exists(template_path):
            with open(template_path, 'r') as f:
                prompt_template = f.read()
        else:
            raise FileNotFoundError(f"Prompt template not found at {template_path}, current file path is {os.path.abspath(__file__)}")
        
        # Debug: logger.info weave attributes before entering the context
        logger.info(f"Setting weave attributes for Task ID {task_id}")
        weave_attrs = {"weave_task_id": task_id}
        logger.info(f"Weave Attributes:{weave_attrs}")

        with weave.attributes(weave_attrs):
            try:
                # Debug: Confirm weave context is active
                logger.info("Weave Context Active: Attributes set")

                ########################################################################
                # inference parameters used, these are important as they can directly
                # impact the quality of the code generated
                ########################################################################
                inference_params = dict(max_tokens=2000, temperature=0.1, n=1)

                logger.info("Sending request to Bedrock with parameters: {inference_params}")
                # logger.info("Messages:", [{"role": "user", "content": input}])
                #logger.info(f"Passing the input to the bedrock model: {task['input']}")
                formatted_prompt = prompt_template.format(question=task['description'])
                logger.info(f"formatted prompt: {formatted_prompt}")

                # run inference 
                agent_id:str = "PWVMWWDIBB" # note this from the agent console on Bedrock
                agent_alias_id:str = '2Q1X7YLRFR' # fixed for draft version of the agent
                session_id:str = str(uuid.uuid1()) # random identifier
                enable_trace:bool = True

                response = client.invoke_agent(inputText=formatted_prompt,
                                                agentId=agent_id,
                                                agentAliasId=agent_alias_id,
                                                sessionId=session_id,
                                                enableTrace=enable_trace
                                            )

                # Debug: logger.info raw response
                logger.debug(f"Raw Response: {response}")

                # # Extract the content and store it in results
                # results[task_id] = response["choices"][0]["message"]["content"]
                # logger.info(f"response content to the code issue: {results[task_id]}")

    
                event_stream = response['completion']
                try:
                    for event in event_stream:        
                        if 'chunk' in event:
                            results = event['chunk']['bytes']
                            logger.info(f"Final answer ->\n{results.decode('utf8')}") 
                            results = results.decode('utf8')
                            end_event_received = True
                        elif 'trace' in event:
                            logger.info(json.dumps(event['trace'], indent=2))
                        else:
                            raise Exception("unexpected event.", event)
                except Exception as e:
                    raise Exception("unexpected event.", e)   
                 
                # extract the python code from the full output
                python_code = _get_python_code(results)
                code = python_code
                logger.info(f"task_id={task_id}, python code=\n{python_code}")

                # Confirm successful processing for the task
                logger.info(f"Task ID {task_id} processed successfully")

            except Exception as e:
                # logger.info exception details
                logger.info(f"Error processing Task ID {task_id}: {e}")
                logger.error(f"going to use a canned response, this will fail the evaluation for this task but allow the rest of the tests to proceed")
                code = FAILED_RESPONSE
    
    # # assign the response field for all tasks
    # for task_id, task in input.items():

    #     input['response'] = code
    input['response'] = code

    # final results
    logger.debug(f"Final Results: {input}")

    return input

if __name__ == "__main__":
    
    # sample input
    input = json.loads(Path("input.txt").read_text())

    print(input)
    # model name and prompt template
    kwargs = dict(model_name="bedrock/amazon.nova-lite-v1:0",
                  prompt_template_path="prompt_templates/nova.txt")
    logger.info(f"kwargs={kwargs}")

    # run the agent code
    run(input, **kwargs)
    logger.info("all done")
    

