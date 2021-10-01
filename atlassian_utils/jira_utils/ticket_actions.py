from jira import Issue, JIRA


def flag_issue(iss_key: str, jira_obj: JIRA) -> None:
    raise NotImplementedError()


def write_comment(iss_key: str, comment_body: str, jira_obj: JIRA) -> None:
    raise NotImplementedError()


def check_if_comment_already_exists(iss: Issue, violation_body: str) -> bool:
    raise NotImplementedError()


def write_violation_as_comment(iss: Issue, violation_body: str, jira_obj: JIRA) -> None:
    """Writes the violation as a UNIQUE comment and flags the issue.
    Check if violation exists, if it does and was written more than 2 days ago, delete it and write a new one.

    Args:
        iss:
        violation_body:
        jira_obj:

    """
    # check if violation exists
    # write violation
    raise NotImplementedError()
