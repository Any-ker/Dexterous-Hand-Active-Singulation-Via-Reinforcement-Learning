from ast import literal_eval


def get_AgentIndex(config):
    """
    Return the agent index list used by run_train.py. 
    """
    raw_value = config["env"].get("handAgentIndex", "[[0, 1, 2, 3, 4, 5]]")

    if isinstance(raw_value, str):
        try:
            base_index = literal_eval(raw_value)
        except (ValueError, SyntaxError) as err:
            raise ValueError(f"Invalid handAgentIndex string: {raw_value}") from err
    else:
        base_index = raw_value

    agent_index = [base_index, base_index]
    return agent_index