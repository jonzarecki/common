import os
import re
from typing import List

from dotenv import load_dotenv
from jira import Comment, Issue, JIRA, User

from common.atlassian_utils.general import parse_date_str
from common.atlassian_utils.jira_utils.user import load_user

load_dotenv(f"{os.path.dirname(__file__)}/.env")  # config = {"USER": "foo", "EMAIL": "foo@example.org"}


def _extract_tagged_users(jira: JIRA, s: str) -> List[User]:
    ids = [m_str[12:-1] for m_str in re.findall(r"\[~accountid:\S*]", s)]
    return [load_user(jira, aid) for aid in ids]


def ticket_has_unanswered_question(jira: JIRA, iss: Issue) -> bool:
    """Simple check to see if an issue has unanswered question."""
    try:
        last_comment: Comment = iss.fields.comment.comments[-1]  # type: ignore
    except AttributeError:
        return False  # no comment
    author_email = last_comment.author.emailAddress
    creation_date_no_tz = parse_date_str(last_comment.created, with_tz=False)  # in local time
    assert creation_date_no_tz is not None and author_email is not None
    tagged_users = _extract_tagged_users(jira, last_comment.body)
    return len(tagged_users) > 0


def main() -> None:
    jira = JIRA(server=os.environ["JIRA_URL"], basic_auth=(os.environ["JIRA_USERNAME"], os.environ["JIRA_PASSWORD"]))
    iss = jira.issue("JT-3")
    assert ticket_has_unanswered_question(jira, iss)
    load_user(jira, "557058:8f50afc7-9921-4c17-8d6a-6cce70d675fd")


if __name__ == "__main__":
    main()
