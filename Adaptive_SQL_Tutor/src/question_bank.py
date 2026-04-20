"""
SQL Question Bank: 10 concepts x 3 difficulties x 3 questions = 90 total questions.

Concepts form a dependency graph (prereqs must be learned before dependents
can be efficiently learned). This lets the RL agent discover curriculum ordering.
"""

# Concept definitions with dependencies
CONCEPTS = [
    {"id": 0,  "name": "SELECT",            "prereqs": []},
    {"id": 1,  "name": "WHERE",             "prereqs": [0]},
    {"id": 2,  "name": "ORDER_BY_LIMIT",    "prereqs": [0]},
    {"id": 3,  "name": "AGGREGATES",        "prereqs": [0]},
    {"id": 4,  "name": "GROUP_BY",          "prereqs": [1, 3]},
    {"id": 5,  "name": "HAVING",            "prereqs": [4]},
    {"id": 6,  "name": "JOINS",             "prereqs": [1]},
    {"id": 7,  "name": "SUBQUERIES",        "prereqs": [1, 6]},
    {"id": 8,  "name": "CTES",              "prereqs": [7]},
    {"id": 9,  "name": "WINDOW_FUNCTIONS",  "prereqs": [4, 8]},
]

N_CONCEPTS = len(CONCEPTS)
DIFFICULTIES = ["easy", "medium", "hard"]
DIFFICULTY_PENALTY = {"easy": 0.0, "medium": 0.08, "hard": 0.18}


# Question bank: 3 per (concept, difficulty)
QUESTIONS = {
    # SELECT
    (0, "easy"):   ["List all columns of table employees.",
                    "Show the name column from customers.",
                    "SELECT all rows from orders."],
    (0, "medium"): ["Select name and salary from employees.",
                    "Return distinct departments from employees.",
                    "Select first 10 employee names (any order)."],
    (0, "hard"):   ["Select name aliased as full_name and salary*12 as annual.",
                    "Use DISTINCT on a computed column salary/1000.",
                    "Combine SELECT with column concatenation."],
    # WHERE
    (1, "easy"):   ["Employees with salary > 50000.",
                    "Orders where status = 'shipped'.",
                    "Customers from city 'Boston'."],
    (1, "medium"): ["Employees hired after 2020 in engineering dept.",
                    "Orders with total BETWEEN 100 AND 500.",
                    "Customers whose name starts with 'A'."],
    (1, "hard"):   ["Employees not in (sales, hr) with salary > 70k.",
                    "Orders placed in last 30 days AND status IN ('shipped','delivered').",
                    "Complex WHERE with NULL handling and NOT IN."],
    # ORDER BY / LIMIT
    (2, "easy"):   ["Top 5 highest-paid employees.",
                    "All orders sorted by date descending.",
                    "Customers sorted alphabetically."],
    (2, "medium"): ["Top 10 orders by total, tiebreak by date.",
                    "Page 2 of results (rows 11-20) sorted by id.",
                    "Last 5 orders by creation time."],
    (2, "hard"):   ["Top-N per group emulation using ORDER BY + LIMIT.",
                    "Sort NULLS LAST with custom tiebreaks.",
                    "Pagination with deterministic multi-column sort."],
    # AGGREGATES
    (3, "easy"):   ["Count of all employees.",
                    "Sum of order totals.",
                    "Average salary across all employees."],
    (3, "medium"): ["Count distinct departments.",
                    "Min and max salary in one query.",
                    "Average order total excluding cancelled."],
    (3, "hard"):   ["Conditional aggregation with CASE inside SUM.",
                    "Percentage of orders that are shipped.",
                    "Weighted average using SUM(x*w)/SUM(w)."],
    # GROUP BY
    (4, "easy"):   ["Count of employees per department.",
                    "Total sales per customer.",
                    "Average salary per department."],
    (4, "medium"): ["Orders per customer per month.",
                    "Employees per department per hire year.",
                    "Revenue per region per product category."],
    (4, "hard"):   ["GROUP BY with ROLLUP for subtotals.",
                    "Multi-level GROUP BY with aggregate expressions.",
                    "GROUP BY with complex HAVING-less filtering."],
    # HAVING
    (5, "easy"):   ["Departments with more than 5 employees.",
                    "Customers with more than 3 orders.",
                    "Products with average rating > 4."],
    (5, "medium"): ["Departments with avg salary > 60k AND count > 3.",
                    "Months where total orders exceeded 1000.",
                    "Categories with revenue above overall average."],
    (5, "hard"):   ["HAVING with correlated subquery.",
                    "HAVING combining multiple aggregate conditions.",
                    "Filter groups where max(x) - min(x) exceeds threshold."],
    # JOINS
    (6, "easy"):   ["Inner join employees with departments.",
                    "Left join orders with customers.",
                    "List all customers and their orders if any."],
    (6, "medium"): ["Three-way join: orders, customers, products.",
                    "Self-join employees to find manager names.",
                    "Left join with coalesce for missing values."],
    (6, "hard"):   ["Full outer join with conditional aggregation.",
                    "Anti-join using LEFT JOIN ... WHERE IS NULL.",
                    "Join on composite keys with range conditions."],
    # SUBQUERIES
    (7, "easy"):   ["Employees earning more than the company average.",
                    "Customers who have placed at least one order.",
                    "Products never ordered."],
    (7, "medium"): ["Top earner per department via correlated subquery.",
                    "Customers whose latest order is within 30 days.",
                    "Employees in departments with avg salary > 70k."],
    (7, "hard"):   ["Nested subquery: departments where best employee's salary > overall average.",
                    "EXISTS vs IN performance-aware rewrite.",
                    "Correlated subquery with aggregate in outer SELECT."],
    # CTES
    (8, "easy"):   ["Define a CTE for active_customers, then select from it.",
                    "CTE for monthly_orders, select top month.",
                    "Simple CTE to pre-aggregate sales."],
    (8, "medium"): ["Two chained CTEs: base filter then aggregation.",
                    "CTE referenced multiple times in final query.",
                    "CTE with window function then filter."],
    (8, "hard"):   ["Recursive CTE to traverse employee hierarchy.",
                    "Recursive CTE for date series generation.",
                    "Multi-stage CTE pipeline for cohort analysis."],
    # WINDOW FUNCTIONS
    (9, "easy"):   ["ROW_NUMBER over order date.",
                    "RANK employees by salary.",
                    "Running total of order amounts."],
    (9, "medium"): ["Top-3 per group using ROW_NUMBER() OVER(PARTITION BY ...).",
                    "LAG to compute day-over-day change.",
                    "Moving 7-day average using ROWS BETWEEN."],
    (9, "hard"):   ["Gaps-and-islands detection with LAG and SUM window.",
                    "Percentile rank within partition with tiebreakers.",
                    "Sessionization using window functions and conditional logic."],
}


def num_actions() -> int:
    """Total action space = concepts x difficulties."""
    return N_CONCEPTS * len(DIFFICULTIES)


def action_to_concept_diff(action: int):
    """Decode flat action id -> (concept_id, difficulty_str)."""
    concept = action // len(DIFFICULTIES)
    diff_idx = action % len(DIFFICULTIES)
    return concept, DIFFICULTIES[diff_idx]


def concept_diff_to_action(concept: int, difficulty: str) -> int:
    return concept * len(DIFFICULTIES) + DIFFICULTIES.index(difficulty)


def get_question(concept: int, difficulty: str, idx: int = 0) -> str:
    bank = QUESTIONS[(concept, difficulty)]
    return bank[idx % len(bank)]
