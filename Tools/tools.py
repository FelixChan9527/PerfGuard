import importlib
from prompt.AIGC_tools_desciption import AIGC_DESCRIPTION
from Tools import AIGC_tools
import random

class Tools():
    def __init__(self):
        self.AIGC_Tools = self.get_class_info(AIGC_tools, AIGC_DESCRIPTION)
        
        shuffled_items = list(self.AIGC_Tools.items())
        random.shuffle(shuffled_items)
        self.AIGC_Tools = dict(shuffled_items)
        
        self.AIGC_Tools_description = self.get_description(self.AIGC_Tools, "AIGC Tools")
    
    def get_class_info(self, object, description):
        tools = {}
        tools_names = description.keys()
        for tool_name in tools_names:
            tool_cls = getattr(object, tool_name)
            tool_instance = tool_cls(description, tool_name)
            tools[tool_name] = tool_instance
        return tools
    
    def get_description(self, Tools_dict, toos_set_name):
        description = "The following are the **{}** descriptions:\n".format(toos_set_name)
        for tool_name, tool in Tools_dict.items():
            description = description + "**"+tool.name+"**: " + tool.description + "\n"
        
        return description



if __name__ == "__main__":
    tools = Tools()
