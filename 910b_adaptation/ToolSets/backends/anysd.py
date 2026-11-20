from ToolSets.toolset_utils.tool_base import StandaloneBackendTool


class AnySDTool(StandaloneBackendTool):
    def __init__(self, device="npu:6", model_path=None, tool_info=None):
        super().__init__(name='AnySD_EditTool', device=device, model_path=model_path, tool_info=tool_info)
