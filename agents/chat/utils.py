def postprocess_message(agent_message):
    try:
        agent_message = agent_message.split('</think>')[-1]
        return agent_message
    except:
        return agent_message
