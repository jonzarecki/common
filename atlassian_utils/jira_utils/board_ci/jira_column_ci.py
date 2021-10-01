from dataclasses import dataclass
from typing import Dict, List

from jira import Issue, JIRA


@dataclass(frozen=True, init=True)
class JiraRule:
    """Class for defining rules for tickets to adhere in a JIRA project."""

    rule_name: str  #: member rule name, should be unique and readable
    project_jql: str  #: member jql query for the project
    filter_jql: str  #: member jql query wanted tickets to apply the rule on

    def does_ticket_violate_rule(self, iss: Issue) -> bool:
        """Returns whether issue violates the rule."""
        pass


def check_rules_compliance(rules: List[JiraRule], jira: JIRA) -> Dict[str, List[JiraRule]]:
    """Checks the compliance of the rules for the given rules.

    Args:
        rules: List of rules to check
        jira: jira object to interact with

    Returns:
        Dictionary between every violated issue-key and the list of rules it violated
    """
    violations_dict: Dict[str, List[JiraRule]] = {}  # type: ignore

    for rule in rules:
        matching_issues = jira.search_issues(f"{rule.project_jql} AND {rule.filter_jql}")
        for iss in matching_issues:
            if not rule.does_ticket_violate_rule(iss):
                continue
            violations_dict[iss.key] = violations_dict.get(iss.key, []) + [rule]  # type: ignore

    return violations_dict
