.PHONY: install test verify verify-1 verify-2 verify-3 verify-4 verify-5a verify-5b grade grade-slow grade-all examples handout handout-smoke self-test self-test-slow clean

install:
	uv sync --extra rl

test:
	uv run pytest tests/

verify:
	uv run pytest verify/ -m "not slow"

verify-1:
	uv run pytest verify/level_1.py -v

verify-2:
	uv run pytest verify/level_2.py -v

verify-3:
	uv run pytest verify/level_3.py -v

verify-4:
	uv run pytest verify/level_4.py -m slow -v

# Interviewer-only ↓ (the hidden grader; do NOT ship to candidates)

grade:
	uv run pytest _solution/grader/ -m "not slow"

grade-slow:
	uv run pytest _solution/grader/ -m slow

grade-all:
	uv run pytest _solution/grader/

verify-5a:
	uv run pytest verify/level_5a.py -v

verify-5b:
	uv run pytest verify/level_5b.py -v

examples:
	uv run python examples/random_policy_isaaclab.py
	uv run python examples/random_policy_mujoco.py

# Interviewer-only ↓

handout:
	uv run python scripts/make_handout.py

handout-smoke:
	uv run python scripts/make_handout.py --smoke

self-test:
	uv run pytest _solution/tests/ -m "not slow"

self-test-slow:
	uv run pytest _solution/tests/

clean:
	rm -rf .pytest_cache dist
