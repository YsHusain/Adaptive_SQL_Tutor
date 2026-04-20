"""
Hint Provider Agent — rule-based, pedagogically grounded.

Produces a short hint or follow-up when the student answers incorrectly,
adapting the hint strategy based on the concept and the student's current
mastery level. This is a rule-based agent (no learning) but it's still a
distinct agent in the orchestration with a clear responsibility.

Hint strategies:
    - CONCEPT_REVIEW      : restate what the concept is about (used when mastery < 0.3)
    - PROCEDURAL_SCAFFOLD : step-by-step prompt (0.3 <= mastery < 0.7)
    - TARGETED_NUDGE      : subtle hint about the specific gap (mastery >= 0.7)

These correspond to different "zones" of the Zone of Proximal Development,
giving more scaffolding for weaker concepts and less for stronger ones.
"""

from ..question_bank import CONCEPTS

CONCEPT_REVIEW, PROCEDURAL_SCAFFOLD, TARGETED_NUDGE = 0, 1, 2
HINT_TYPE_NAMES = {
    CONCEPT_REVIEW: "concept_review",
    PROCEDURAL_SCAFFOLD: "procedural_scaffold",
    TARGETED_NUDGE: "targeted_nudge",
}

# One hint text per (concept, hint_type). Real systems would generate these
# with an LLM; for the assignment static hints are fine and keep it deterministic.
_HINTS = {
    0: {  # SELECT
        CONCEPT_REVIEW: "SELECT retrieves columns from a table. Syntax: SELECT col1, col2 FROM table.",
        PROCEDURAL_SCAFFOLD: "Start with SELECT, list the columns you need, then FROM table_name.",
        TARGETED_NUDGE: "Check whether you used aliases (AS) correctly and that column names are spelled right.",
    },
    1: {  # WHERE
        CONCEPT_REVIEW: "WHERE filters rows based on a condition. Comes after FROM.",
        PROCEDURAL_SCAFFOLD: "Pick the column, pick the comparison (=, >, <, LIKE, IN), pick the value.",
        TARGETED_NUDGE: "Watch out for NULL handling — use IS NULL / IS NOT NULL, not = NULL.",
    },
    2: {  # ORDER BY / LIMIT
        CONCEPT_REVIEW: "ORDER BY sorts results; LIMIT caps how many rows return.",
        PROCEDURAL_SCAFFOLD: "ORDER BY <col> [ASC|DESC], then LIMIT <n>. Multi-column ordering: comma-separate.",
        TARGETED_NUDGE: "Remember ORDER BY runs before LIMIT — so you get the top-N sorted rows, not a random N.",
    },
    3: {  # AGGREGATES
        CONCEPT_REVIEW: "Aggregates (COUNT, SUM, AVG, MIN, MAX) reduce many rows to one number.",
        PROCEDURAL_SCAFFOLD: "Pick your aggregate function, put it in SELECT, and optionally filter with WHERE first.",
        TARGETED_NUDGE: "COUNT(*) counts all rows; COUNT(col) skips NULLs. Check if that matters here.",
    },
    4: {  # GROUP BY
        CONCEPT_REVIEW: "GROUP BY splits rows into groups so aggregates apply per group.",
        PROCEDURAL_SCAFFOLD: "Put aggregate cols in SELECT, group-by cols in GROUP BY. Every non-aggregate col in SELECT must be in GROUP BY.",
        TARGETED_NUDGE: "Check that every non-aggregate column in your SELECT is also in GROUP BY.",
    },
    5: {  # HAVING
        CONCEPT_REVIEW: "HAVING filters groups after GROUP BY — like WHERE but for aggregates.",
        PROCEDURAL_SCAFFOLD: "Use WHERE to filter rows first, GROUP BY to make groups, then HAVING for aggregate filters.",
        TARGETED_NUDGE: "HAVING can reference aggregates directly (e.g., HAVING COUNT(*) > 5); WHERE can't.",
    },
    6: {  # JOINS
        CONCEPT_REVIEW: "JOIN combines rows from two tables based on a related column.",
        PROCEDURAL_SCAFFOLD: "FROM t1 JOIN t2 ON t1.key = t2.key. Pick INNER / LEFT / RIGHT / FULL based on needs.",
        TARGETED_NUDGE: "Think about which side can have missing matches — that decides INNER vs LEFT JOIN.",
    },
    7: {  # SUBQUERIES
        CONCEPT_REVIEW: "A subquery is a SELECT inside another SELECT — treated as a temporary table or value.",
        PROCEDURAL_SCAFFOLD: "Write the inner query first and test it. Then plug it into the outer query.",
        TARGETED_NUDGE: "Consider whether EXISTS or IN would be clearer / faster than a correlated subquery here.",
    },
    8: {  # CTEs
        CONCEPT_REVIEW: "WITH name AS (SELECT ...) defines a named subquery (CTE) you can reference below.",
        PROCEDURAL_SCAFFOLD: "WITH cte_name AS (inner query) SELECT ... FROM cte_name. You can chain multiple CTEs.",
        TARGETED_NUDGE: "For hierarchical data, consider WITH RECURSIVE. Check your base case and recursive case.",
    },
    9: {  # WINDOW FUNCTIONS
        CONCEPT_REVIEW: "Window functions compute per-row values over a set of related rows (a 'window').",
        PROCEDURAL_SCAFFOLD: "Pick your window function, add OVER(PARTITION BY ... ORDER BY ... ROWS BETWEEN ...).",
        TARGETED_NUDGE: "Double-check your PARTITION BY — wrong partition changes the whole computation.",
    },
}


class HintProvider:
    """Rule-based hint generator. Stateless (but called through the orchestrator
       so it counts as a distinct agent in the multi-agent system)."""
    name = "hint_provider"

    def get_hint(self, concept: int, est_mastery: float, return_type=False):
        """Return a hint string suited to the student's current mastery."""
        if est_mastery < 0.30:
            htype = CONCEPT_REVIEW
        elif est_mastery < 0.70:
            htype = PROCEDURAL_SCAFFOLD
        else:
            htype = TARGETED_NUDGE
        text = _HINTS[concept][htype]
        if return_type:
            return text, HINT_TYPE_NAMES[htype]
        return text
