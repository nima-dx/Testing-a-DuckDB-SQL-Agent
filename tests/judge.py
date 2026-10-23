from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.messages import ModelMessagesTypeAdapter, ModelRequest, UserPromptPart


class JudgeCriterion(BaseModel):
    criterion_description: str
    passed: bool
    judgement: str


class JudgeFeedback(BaseModel):
    criteria: list[JudgeCriterion]
    feedback: str


judge_instructions = """
You are an expert judge evaluating the performance of an AI agent.

You will receive:
- the instructions the agent was given
- the user's question
- the agent's final output
- the full log of the agent's run, including tool calls and tool results
- a list of criteria

For each criterion, decide whether the agent's run satisfies it.
Use the log to check the agent's claims: numbers in the output must
match what the tools actually returned.

Return one entry per criterion, in the same order, with:
- criterion_description: the criterion, copied exactly
- passed: true or false
- judgement: a short explanation of your decision

Then give overall feedback on the agent's performance.
""".strip()

judge = Agent(
    name="judge",
    model="openai:gpt-4o-mini",
    instructions=judge_instructions,
    output_type=JudgeFeedback,
)


def _find_user_prompt(messages) -> str:
    for message in messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, UserPromptPart):
                    return str(part.content)
    return ""


def _find_instructions(messages) -> str:
    for message in messages:
        if isinstance(message, ModelRequest) and message.instructions:
            return message.instructions
    return ""


def evaluate_agent_performance(criteria, result, output_transformer=None) -> JudgeFeedback:
    """Ask the judge agent to evaluate an agent run against a list of criteria."""
    messages = result.all_messages()

    output = result.output
    if output_transformer is not None:
        output = output_transformer(output)

    log = ModelMessagesTypeAdapter.dump_json(messages).decode("utf-8")
    criteria_text = "\n".join(f"- {c}" for c in criteria)

    user_prompt = f"""
<INSTRUCTIONS>
{_find_instructions(messages)}
</INSTRUCTIONS>

<USER_INPUT>
{_find_user_prompt(messages)}
</USER_INPUT>

<AGENT_OUTPUT>
{output}
</AGENT_OUTPUT>

<LOG>
{log}
</LOG>

<CRITERIA>
{criteria_text}
</CRITERIA>
""".strip()

    judge_result = judge.run_sync(user_prompt)
    return judge_result.output


def assert_criteria(result, criteria, output_transformer=None) -> JudgeFeedback:
    """Evaluate an agent run and fail the test if any criterion is not met."""
    feedback = evaluate_agent_performance(criteria, result, output_transformer)

    for c in feedback.criteria:
        status = "PASS" if c.passed else "FAIL"
        print(f"[{status}] {c.criterion_description}\n       {c.judgement}")
    print(f"Feedback: {feedback.feedback}")

    failed = [c for c in feedback.criteria if not c.passed]
    assert not failed, "Criteria not met:\n" + "\n".join(
        f"- {c.criterion_description}: {c.judgement}" for c in failed
    )
    assert len(feedback.criteria) == len(criteria), (
        f"Judge returned {len(feedback.criteria)} results for {len(criteria)} criteria"
    )

    return feedback
