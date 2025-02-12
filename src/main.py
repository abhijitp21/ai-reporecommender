import json
import os
from pathlib import Path
import fnmatch

from langchain.chains.sequential import SequentialChain
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.tools import Tool, tool

# Initialize OpenAI API key and model
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_API_MODEL = os.getenv("OPENAI_API_MODEL", "gpt-4")

# Initialize GitHub token
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Inputs and Constants
GITHUB_EVENT_PATH_ENV = "GITHUB_EVENT_PATH"
DEFAULT_GITHUB_EVENT_PATH = ""
INPUT_EXCLUDE_ENV = "INPUT_EXCLUDE"
SUPPORTED_EVENTS = {"opened", "synchronize"}

# Initialize ChatOpenAI for AI-powered analysis
llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, model=OPENAI_API_MODEL)

# LangChain prompt template for AI analysis
prompt_template = """
Analyze the following diff file from a pull request and generate useful comments for a code review. Include actionable recommendations.

Pull Request Title: {pr_title}
Pull Request Description: {pr_description}

Diff File:
{diff}

Generate:
- Comments for each code chunk, if improvements can be made.
- Avoid duplicating recommendations or commenting on deleted files.
"""  # Template for generating AI comments for PR files.


@tool
def read_github_event() -> dict:
    """Reads and parses the GitHub event data from the specified file path."""
    event_path = Path(os.getenv(GITHUB_EVENT_PATH_ENV, DEFAULT_GITHUB_EVENT_PATH))
    if not event_path.exists():
        raise FileNotFoundError(f"No GitHub event file found at {event_path}")
    with event_path.open("r") as event_file:
        return json.load(event_file)


@tool
def get_pr_details(event_data: dict) -> dict:
    """Fetches pull request details from the GitHub Event payload."""
    repository = event_data.get("repository", {})
    pull_request_number = event_data.get("number")
    owner = repository.get("owner", {}).get("login")
    repo_name = repository.get("name")

    return {
        "owner": owner or "",
        "repo": repo_name or "",
        "pull_number": pull_request_number,
        "title": event_data.get("pull_request", {}).get("title", ""),
        "description": event_data.get("pull_request", {}).get("body", ""),
    }


@tool
def get_diff(pr_details: dict) -> str:
    """Fetches the diff of the pull request."""
    owner = pr_details["owner"]
    repo = pr_details["repo"]
    pull_number = pr_details["pull_number"]
    # Using GitHub API client to fetch PR diff (would replace this with a functional tool if needed)
    # Diff retrieval (real external fetch example depends on GitHub API usage)
    return f"Dummy diff content for PR {pull_number} from {owner}/{repo}"


@tool
def filter_files(parsed_diff: list) -> list:
    """Filters files based on exclude patterns from the environment."""
    exclude_patterns = os.getenv(INPUT_EXCLUDE_ENV, "").split(",")
    filtered_files = [
        file for file in parsed_diff
        if not any(fnmatch(file.get("new_path", ""), pattern.strip()) for pattern in exclude_patterns)
    ]
    return filtered_files


@tool
def analyze_code(filtered_files: list, pr_details: dict) -> list:
    """Analyzes parsed diff files and generates comments using AI."""
    comments = []
    for file in filtered_files:
        diff = file.get("diff")  # Dummy diff content by file
        if not diff:
            continue

        # Create dynamic prompt for each diff file
        prompt = PromptTemplate.from_template(prompt_template).format(
            pr_title=pr_details["title"],
            pr_description=pr_details["description"],
            diff=diff,
        )

        # Generate AI response for code review
        ai_response = llm.generate([{"content": prompt}])
        if ai_response and ai_response.generations[0].text:
            comments.append(f"Comments for {file.get('new_path', 'unknown')}:\n{ai_response.generations[0].text}")
    return comments


@tool
def create_review_comment(pr_details: dict, comments: list):
    """Creates a review comment on the pull request using the GitHub API."""
    # Example placeholder GitHub comment interaction (replace with GitHub API logic if required)
    for comment in comments:
        print(f"Posting comment to PR {pr_details['pull_number']}:\n{comment}")


# Define the LangChain sequential chain
def main_chain():
    tools = [
        Tool(name="Read GitHub Event", func=read_github_event),
        Tool(name="Get PR Details", func=get_pr_details),
        Tool(name="Get Diff", func=get_diff),
        Tool(name="Filter Files", func=filter_files),
        Tool(name="Analyze Code", func=analyze_code),
        Tool(name="Create Review Comment", func=create_review_comment),
    ]

    chain = SequentialChain(
        chains=[
            Tool(name="Read GitHub Event", func=read_github_event),
            Tool(name="Get PR Details", func=get_pr_details),
            Tool(name="Get Diff", func=get_diff),
            Tool(name="Filter Files", func=filter_files),
            Tool(name="Analyze Code", func=analyze_code),
            Tool(name="Create Review Comment", func=create_review_comment),
        ],
        verbose=True,
    )
    return chain


# Run the LangChain-based workflow
if __name__ == "__main__":
    try:
        pr_chain = main_chain()
        pr_chain.run()
    except Exception as exc:
        print(f"Error occurred: {exc}")
