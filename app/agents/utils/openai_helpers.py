def function_tool(func):
    func.openai_schema = {
        "name": func.__name__,
        "description": func.__doc__,
        "parameters": {}
    }
    return func
