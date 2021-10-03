from datetime import datetime, timedelta
from typing import List, Optional

from jira import Comment, Issue, JIRA

from common.atlassian_utils.general import parse_atls_date_str
from common.atlassian_utils.jira_utils.board_ci import rules_and_filters


def flag_issue(iss_key: str, jira_obj: JIRA) -> None:
    iss = jira_obj.issue(iss_key)
    if not rules_and_filters.flagged(iss):  # type: ignore
        try:
            iss.fields.customfield_10021 = [{"value": "Impediment"}]
            iss.update(jira=jira_obj)
        except:  # noqa
            print("Failed to flag")


def write_jira_comment(iss_key: str, comment_body: str, jira_obj: JIRA) -> None:
    jira_obj.add_comment(iss_key, comment_body)


def check_if_comment_already_exists(iss: Issue, violation_body: str) -> Optional[str]:
    comments: List[Comment] = iss.fields.comment.comments  # type: ignore
    now = datetime.now()

    for c in comments:
        if c.body == violation_body:
            creation_date_no_tz = parse_atls_date_str(c.created, with_tz=False)  # in local time
            if now - creation_date_no_tz < timedelta(days=2):
                return str(c.id)
    return None


def write_violation_as_comment(iss: Issue, violation_body: str, jira_obj: JIRA) -> bool:
    """Writes the violation as a UNIQUE comment and flags the issue.

    Check if violation exists, if it does and was written more than 2 days ago, delete it and write a new one.
    """
    # check if violation exists
    violation_comment_id = check_if_comment_already_exists(iss, violation_body)
    if violation_comment_id is not None:
        return False

    write_jira_comment(iss.key, violation_body, jira_obj)
    flag_issue(iss.key, jira_obj)
    return True
