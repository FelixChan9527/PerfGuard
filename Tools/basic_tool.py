from metagpt.actions import Action
import textwrap
import ast
import os, json
import random
from datetime import datetime


class Basic_Tool(Action):
    def __init__(self, info_dict, name):
        super().__init__()
        """
        工具的基础初始化和说明
        name: 工具名称
        descrtion：工具描述
        usage：工具用法
        example：调用示例
        query: 用于询问llm从而获取工具的输入方式
        """
        self.name = name
        tool_info = info_dict[self.name]
        self.tool_info = tool_info  # 加载全部工具信息
        self.description = textwrap.dedent(self.tool_info["description"]).replace("\n", " ").strip()
        self.usage = self.tool_info["usage"]
        
        preference_file = "prompt/preference_logs/preference_lastest.json"
        # preference_file = "prompt/preference_logs/preference_ori.json"
        with open(preference_file, "r", encoding="utf-8") as f:
            preference_data = json.load(f)
        
        try:
            self.category = self.tool_info["category"]
            # self.preference = self.tool_info["preference"]
            self.preference = preference_data[self.category][self.name]
        except:
            self.category = None
            self.preference = None
        self.save_path = "./temp"
        os.makedirs(self.save_path, exist_ok=True)
        
        self.input_format, self.output_format, self.example, self.query = self.parse_json()
    
    def reflash_preference(self):
        preference_file = "prompt/preference_logs/preference_lastest.json"
        with open(preference_file, "r", encoding="utf-8") as f:
            preference_data = json.load(f)
        
        self.category = self.tool_info["category"]
        self.preference = preference_data[self.category][self.name]
    
    def parse_content(self, reply:str, tag:str):
        close_tag = self.get_closing_tag(tag)
        content = (reply.split(tag)[-1]).split(close_tag)[0].strip()
        
        return content
    
    def get_save_output_dirs(self, num):
        dirs_list = []
        current_date = datetime.now()
        formatted_date = current_date.strftime("%m%d")
        current_time = current_date.strftime("%H%M%S")
        for i in range(num):
            img_dir = os.path.join(self.save_path, formatted_date+current_time+"_"+self.name+"_" + str(i).zfill(2) +".png")
            dirs_list.append(img_dir)
        return dirs_list
    
    def get_closing_tag(self, tag: str) -> str:
        if tag.startswith("<") and tag.endswith(">"):
            return "</" + tag[1:]
        raise ValueError("Invalid tag format")
    
    def parse_json(self):
        input_dic = self.usage["input"]
        input_tag = list(input_dic.keys())
        input_format = {}
        output_dic = self.usage["output"]
        output_tag = list(output_dic.keys())
        output_format = {}
        
        query = "Please refer to the example and enter the conditions of **{}**: ".format(self.name)
        example = ""
        for i, tag in enumerate(input_tag):
            close_tag = self.get_closing_tag(tag)
            query = query + tag+input_dic[tag]["description"]+close_tag
            example += tag+str(input_dic[tag]["example"])+close_tag + "\n"
            input_format[tag] = input_dic[tag]["type"]
            if i != len(input_tag)-1:   # 如果是最后一个
                query += ", "
        query = query + " " + "as input conditions for {}.\n".format(self.name)
        
        query = query + "The processing results of **{}** will be represented by ".format(self.name)
        example += "The following are the processing results of **{}**:\n".format(self.name)
        for i, tag in enumerate(output_tag):
            close_tag = self.get_closing_tag(tag)
            query = query + tag+output_dic[tag]["description"]+close_tag
            example += tag+str(output_dic[tag]["example"])+close_tag + "\n"
            output_format[tag] = output_dic[tag]["type"]
            if i != len(output_tag)-1:   # 如果是最后一个
                query += ", "
        query = query + "\n\n" + "Please provide your **{}** input information strictly following the given format:\n".format(self.name)
        
        example_prompt = query + example
        
        example_prompt = "<example>\n" + example_prompt + "</example>\n"
        
        return input_format, output_format, example_prompt, query
        
    async def run_function(self, conditions: list) -> list:
        """
        代码的运行逻辑在这里写
        """
        return []
    
    def __parse_input__(self, prompt):
        """
        准备调用工具，在调用工具前经过输入工具的query后，对llm输出的条件进行解析
        """
        condition_str_clean = ""
        conditions = []
        for tag, tag_type in self.input_format.items():  # 遍历每个条件的tag，取出对应的条件
            close_tag = self.get_closing_tag(tag)
            condition_str = (prompt.split(tag)[-1]).split(close_tag)[0].strip()

            # 针对不同类型处理方式不同
            if tag_type is str:
                # 去除引号和空格
                condition = condition_str.replace("'", "").replace('"', "")
            elif tag_type is list:
                # 字符串形式的列表，需要保留原始格式用 ast 解析
                condition = ast.literal_eval(condition_str)
            else:
                # 其他类型也用 ast 解析（如 dict, tuple, int 等）
                condition = ast.literal_eval(condition_str)

            conditions.append(condition)
            condition_str_clean = condition_str_clean + tag+condition_str+close_tag+"\n"

        return conditions, condition_str_clean
    
    def __parse_output__(self, results: list)->dict:
        """
        调用完工具后，需要输入到LLM中的响应。解析输出的结果，用于给LLM进行理解。
        results的格式是dict，里面写的是{tag: result}
        """
        
        reply = "The following are the processing results of **{}**:\n".format(self.name)
        chech_flag = True
        for i, tag in enumerate(self.output_format):
            result = results[i]
            # 判断输出类型是否正确
            if isinstance(result, self.output_format[tag]):
                close_tag = self.get_closing_tag(tag)
                reply = reply + tag+str(result)+close_tag+"\n"
            else:
                # 类型错误
                chech_flag = False
        # for i, tag in enumerate(self.output_format):
        #     result = results
        #     close_tag = self.get_closing_tag(tag)
        #     reply = reply + tag+str(result)+close_tag+"\n"

        if chech_flag == False:
            reply = None
        
        return reply
    
    async def run(self, context):
        """
        在此处编写有关运行，并规定设置输入和输出
        """
        
        conditions, condition_str_clean = self.__parse_input__(context)
        results = await self.run_function(conditions)
        reply = self.__parse_output__(results)
        if reply != None:
            return reply, condition_str_clean
        else:
            raise ValueError("output result type error!")

