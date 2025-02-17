import os
import json
from pathlib import Path
import fnmatch
from langchain_community.chains import SequentialChain
from langchain_core.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_community.tools import Tool, tool

# Constants
EMPTY_EVENT_PATH = ""
SUPPORTED_EVENTS = {"opened", "synchronize"}
ERROR_MESSAGE_EVENT_FILE = "No GitHub event file found at {}"
AI_PROMPT_TEMPLATE = """
Analyze the following diff file from a pull request and generate useful comments for a code review. Include actionable recommendations.
Pull Request Title: {pr_title}
Pull Request Description: {pr_description}
Diff File:
{diff}
Generate:
- Comments for each code chunk, if improvements can be made.
- Avoid duplicating recommendations or commenting on deleted files.
"""


# Environment Variables Helper
def get_env_var(key: str, default: str = "") -> str:
    """
    Retrieve the value of an environment variable.

    This function fetches the value of an environment variable using its key. If the
    specified key does not exist in the environment, a default value will be returned.

    :param key: The name of the environment variable to retrieve.
    :param default: The value to return if the specified environment variable is not found.
    :return: The value of the environment variable or the default value if the variable is
             not present.
    """
    return os.getenv(key, default)


# Initialize OpenAI Chat Model
llm = ChatOpenAI(
    openai_api_key=get_env_var("OPENAI_API_KEY"),
    model=get_env_var("OPENAI_API_MODEL", "gpt-4"),
)

@tool
def read_github_event() -> dict:
    """
    Reads the GitHub event data from the file specified in the GITHUB_EVENT_PATH
    environment variable. If the file is not found, it uses an empty event path
    as a fallback and raises an exception if the file still does not exist.

    This utility reads and parses the JSON data from the event file, providing a
    dictionary representing the event payload.

    :raises FileNotFoundError: If the event file does not exist.

    :return: A dictionary containing the parsed GitHub event data.
    :rtype: dict
    """
    event_path = Path(get_env_var("GITHUB_EVENT_PATH", EMPTY_EVENT_PATH))
    if not event_path.exists():
        raise FileNotFoundError(ERROR_MESSAGE_EVENT_FILE.format(event_path))
    with event_path.open("r") as event_file:
        return json.load(event_file)


@tool
def get_pr_details(event_data: dict) -> dict:
    """
    Extracts and returns specific details about a pull request from the provided event data.

    This function processes the input dictionary, typically received from an event payload such as
    a webhook, and extracts information related to the repository and pull request. It ensures
    that the relevant details are structured and returned in a dictionary for easy usage.

    :param event_data: A dictionary containing event details, particularly the pull request and
        associated repository information.
    :type event_data: dict

    :return: A dictionary containing the pull request details, including 'owner', 'repo',
        'pull_number', 'title', and 'description'.
    :rtype: dict
    """
    repository = event_data.get("repository", {})
    return {
        "owner": repository.get("owner", {}).get("login", ""),
        "repo": repository.get("name", ""),
        "pull_number": event_data.get("number"),
        "title": event_data.get("pull_request", {}).get("title", ""),
        "description": event_data.get("pull_request", {}).get("body", ""),
    }


@tool
def get_diff(pr_details: dict) -> str:
    """
    Provides a dummy implementation for generating diff content for a given pull request (PR). This implementation
    constructs a string showcasing brief details of the PR, including its pull number, owner, and repository.

    :param pr_details: A dictionary containing pull request details including:
                       - `pull_number`: The number identifying the pull request.
                       - `owner`: The owner of the repository.
                       - `repo`: The name of the repository.
    :type pr_details: dict

    :return: A string containing dummy diff content with the PR's identifying information.
    :rtype: str
    """
    return f"Dummy diff content for PR {pr_details['pull_number']} from {pr_details['owner']}/{pr_details['repo']}"


@tool
def filter_files(file_diffs: list) -> list:
    """
    Filters a list of file changes based on exclude patterns obtained from environment
    variables. This function is useful for determining which files should be ignored
    from further processing.

    The exclude patterns are defined in the environment variable `INPUT_EXCLUDE`,
    which is expected to be a comma-separated list of glob patterns. Each pattern is
    stripped of leading and trailing whitespaces before comparison. If a file's
    `new_path` matches any of the exclude patterns, it will be excluded from the
    returned list.

    :param file_diffs: A list of dictionaries, each representing information about
                       a file change. Each dictionary should contain a `new_path`
                       key indicating the new path of the file after the change.
    :return: A list of dictionaries from the input that do not match any of the
             exclude patterns.
    """
    exclude_patterns = [pattern.strip() for pattern in get_env_var("INPUT_EXCLUDE", "").split(",")]
    return [
        file for file in file_diffs
        if not any(fnmatch.fnmatch(file.get("new_path", ""), pattern) for pattern in exclude_patterns)
    ]


@tool
def analyze_code(filtered_files: list, pr_details: dict) -> list:
    """
    Analyzes the code changes by utilizing AI-driven prompts. The function processes
    a list of filtered files and generates comments for each file based on its diff
    content. It dynamically creates a prompt for the AI using the pull request
    title, description, and file diff. The generated AI responses are collected as
    comments corresponding to each file.

    :param filtered_files: A list of dictionaries where each dictionary represents a file.
                            Each file dictionary should contain the 'diff' key, which holds
                            the changes made in that file. If 'diff' is absent or empty,
                            the file is skipped.
    :type filtered_files: list
    :param pr_details: A dictionary containing details of the pull request. It must
                       include 'title' and 'description' keys representing the pull
                       request's title and description, respectively.
    :type pr_details: dict
    :return: A list of strings where each string contains comments related to the
             analyzed diff of a file. If a file's diff doesn't produce comments, it is
             not included in the returned list.
    :rtype: list
    """
    comments = []
    for file in filtered_files:
        diff = file.get("diff")
        if not diff:
            continue
        # Create dynamic prompt for each diff file
        prompt = PromptTemplate.from_template(AI_PROMPT_TEMPLATE).format(
            pr_title=pr_details["title"],
            pr_description=pr_details["description"],
            diff=diff,
        )
        ai_response = llm.generate([{"content": prompt}])
        if ai_response and ai_response.generations[0].text:
            comments.append(f"Comments for {file.get('new_path', 'unknown')}:\n{ai_response.generations[0].text}")
    return comments


@tool
def create_review_comment(pr_details: dict, comments: list):
    """
    Generates and prints review comments for a pull request.

    This function takes details of a pull request and a list of comments,
    then prints each comment with the pull request ID. It is generally
    used to simulate the posting of comments on a pull request during
    code review processes.

    :param pr_details: Details of the pull request, containing at minimum the 'pull_number' key.
    :type pr_details: dict
    :param comments: A list of comments to be posted on the pull request.
    :type comments: list
    :return: None
    :rtype: None
    """
    for comment in comments:
        print(f"Posting comment to PR {pr_details['pull_number']}:\n{comment}")


# Workflow Chain
def main_chain():
    """
    Constructs the main chain of tools for processing GitHub events and creating
    review comments. Each tool in the chain performs a specific operation in
    sequence, such as reading a GitHub event, extracting pull request details,
    calculating differences, filtering files, analyzing code, and generating
    review comments.

    :raises Exception: If there is any failure in the initialization of the
        tool chain or during chain execution.

    :rtype: SequentialChain
    :return: The sequential chain of tools containing tasks for handling
        and processing GitHub events.
    """
    action = get_env_var("REPO_RECOMMENDER_ACTION", "prreview").lower()
    return sequentialChainForAction(action)

def sequentialChainForAction(action):
    if(action == "prreview"):
        tools = [
            Tool(name="Read GitHub Event", func=read_github_event),
            Tool(name="Get PR Details", func=get_pr_details),
            Tool(name="Get Diff", func=get_diff),
            Tool(name="Filter Files", func=filter_files),
            Tool(name="Analyze Code", func=analyze_code),
            Tool(name="Create Review Comment", func=create_review_comment),
        ]
    elif(action == ""):
        tools = [
            Tool(name="Read GitHub Event", func=read_github_event),
            Tool(name="Get PR Details", func=get_pr_details),
            Tool(name="Get Diff", func=get_diff),
            Tool(name="Filter Files", func=filter_files),
            Tool(name="Analyze Code", func=analyze_code),
            Tool(name="Create Review Comment", func=create_review_comment),
        ]
    return SequentialChain(chains=tools, verbose=True)

if __name__ == "__main__":
    try:
        pr_chain = main_chain()
        pr_chain.run()
    except Exception as exc:
        print(f"Error occurred: {exc}")
