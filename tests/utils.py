def collect_tools(messages):
    """Return all tool calls from an agent's message history, in order."""
    tool_calls = []

    for message in messages:
        for part in message.parts:
            if part.part_kind == "tool-call":
                tool_calls.append(part)

    return tool_calls
