from backend.config.tools import AVAILABLE_TOOLS, ToolName


TOOL_CONFIGS = {
    ToolName.Search_File: {
        "type": "function",
        "function": {
            "name": ToolName.Search_File.value,
            "description": AVAILABLE_TOOLS[ToolName.Search_File].description,
            "parameters": {
                "type": "object",
                "properties": {
                    "search_query": {
                        "type": "string",
                        "description": "Textual search query to search over the file's content for",
                    },
                    "filenames": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "A list of one or more uploaded filename strings to search over",
                    },
                },
                "required": ["search_query", "filenames"],
            },
        },
    },
    # ToolName.Read_File: {
    #     "type": "function",
    #     "function": {
    #         "name": ToolName.Read_File.value,
    #         "description": AVAILABLE_TOOLS[ToolName.Read_File].description,
    #         "parameters": {
    #             "type": "object",
    #             "properties": {
    #                 "filename": {
    #                     "type": "string",
    #                     "description": "The name of the attached file to read.",
    #                 },
    #             },
    #             "required": ["filename"],
    #         },
    #     },
    # },
}
