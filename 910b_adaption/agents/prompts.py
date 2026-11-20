ANALYST_EVAL_CONTEXT = """Please answer the following questionnaire: does the input text semantics or step specify any of the following factors?

1. Entity Category and Quantity: Indicates which key entities and their corresponding quantities are specified in the prompt. 
Need to be distinguished from the **background**; background-related terms do not belong to entity categories, such as "water" in "in the water" or "lake" in "by the lake"—these are not considered entities. 
If specified, record them in the output JSON under "category" and "number". 
"category" should list entity types only (without any attribute words), and "number" should contain integers representing the count of each entity. If the number of entities specified in the prompt is unclear—such as using terms like "some", "many", or "several"—then set the "number" as -1.
If an object is supposed to be removed (e.g., in an editing task), then reduce its quantity to the correct number (e.g., 0); if an object is to be added, then increase its quantity to the correct number.
If not specified, write "null" for both "category" and "number".
example:
<goal>
{
    "category_number": {"pig": 2,"dog": 4}
}
</goal>

2. Reference Consistency: Indicates whether the appearance or shape of entities in the prompt are required to match a reference image. 
If specified, write the object category, reference image path, and reference type in the output JSON under "identity". 
The reference type should be one of ["face", "pose", "identity"], where:
"face" means the person"s face should match the reference image,
"pose" means the body posture should match,
"identity" means the object"s overall appearance should match.
**Note 1: The prompt must explicitly mention reference to a specific image in order to determine that it involves Reference Consistency.**
**Note 2: For the editing task, the original image provided by the user is not considered a reference image here.**
If not specified, it is 'null'.
example:
<goal>
{
    "identity": [{"dog": "image.jpg", "type": "identity"}]
}
</goal>

3. Positional relationship: Indicates whether the prompt specifies concrete positional relationships such as "A is above B" or "C is in front of D". 
Entities involved in positional relationships should be considered key entities in the prompt. 
Abstract relationships like "A holding B" or "C hugging D" do *not* count. If specified, write them under "position" in the output JSON.
If not specified, it is 'null'.
example:
<goal>
{
    "position": ["A pig is next to the cow.","A car is to the right of a bicycle."]
}
</goal>

4. Attribute binding: Indicates whether key entities in the prompt are described with specific attribute associations. 
Please note that attribute binding requires the entity to have an explicitly stated attribute word, such as material or color (quantity does not count).
Entities must be independent individuals, not body parts. If an entity does not have clearly stated attributes in the task, then no attribute description is required.
If so, present the key entities and their bound attributes in the form of clauses in the output JSON under "attribute-binding".
If not specified, it is 'null'.
example:
<goal>
{
    "attribute-binding": ["a red dog.", "A wooden mouse.", "A tattered backpack"]
}
</goal>

5. Image Style: Indicates whether the prompt specifies a particular overall visual style for the image, such as "realistic style", "cartoon style", "black-and-white photography", etc. 
If specified, include it under "style" in the output JSON.
If not specified, it is 'null'.
example:
<goal>
{
    "style": "Dreamy style"
}
</goal>

6. Background: Indicates whether the prompt specifies the background environment where the entities are located, such as "deep sea", "forest", "city", etc. 
If specified, include it under "background" in the output JSON.
If not specified, it is 'null'.
example:
<goal>
{
    "background": "Dreamy style"
}
</goal>

All the questions in the above questionnaire should be written in a single JSON file, like this:
example:
<goal>
{
    "category_number": {"pig": 2,"dog": 4},
    "identity": [{"dog": "image.jpg", "type": "identity"}],
    "position": ["A pig is next to the cow.","A car is to the right of a bicycle."],
    "attribute-binding": ["two pink pigs"],
    "style": "Dreamy style",
    "background": "Dreamy style"
}
</goal>

If any item is not specified, set it directly to `null`.
example:
<goal>
{
    "category_number": {"pig":2, "dog": 4},
    "identity": null,
    "position": null,
    "attribute-binding": null,
    "style": "Dreamy style",
    "background": "forest"
}
</goal>

Here are some sample references for your consideration:
<input>
prompt: A dog sits by a round clock under a white lamp, with two green apples and four birds nearby, and the overall environment is in a quiet room. The dog"s appearance refers to the image "./temp/test.png".
</input>

<output>
<goal>
{
    "category_number": {"dog": 1, "clock": 1, "lamp": 1, "apple": 2, "bird": 4},
    "identity": [{"dog": "./temp/test.png", "type": "identity"}],
    "position": ["A pig is next to the cow.","A car is to the right of a bicycle."],
    "attribute-binding": ["a round clock.", "a white lamp.", "two green apples"],
    "style": null,
    "background": "a quiet room"
}
</goal>
</output>

Output JSON data in XML format (). Please refer to the provided example for details."""

ANALYST_TASK_SOURCE = """
        1. Please analyze the user's needs based on the provided content and summarize their requirements.
        If a specific image is referenced, the path to the reference image must be specified.
        **No assumptions are allowed about the user-provided information; the output must closely align with the user’s given information.**
        **The output must be derived through precise and correct reasoning, rather than copying the user's input.**
        Transform the user input into concrete visual elements for the final image, avoiding overly simple or abstract terms.
        The output task must be **precise and concise, within 20 tokens**.
        Output the task in the format: <task>Your summary task</task>.
        
        2. Please provide the semantics of the final output image (i.e., what the final rendered image looks like) in textual form.
        The output semantic should be described in terms of key objects in the image, their attributes (numeracy, categories, color, texture, etc.), spatial relationships, background, and image style, etc..
        The output semantic must be **precise and concise, within 20 tokens**.
        Output the semantic in the format: <semantic>Textual semantic information of the target image.</semantic>.
        
        ================= The following is a reference example ================= 
        Example 1:
        <task>Generate an image of a cyberpunk city at night, with neon lights, flying cars, and a futuristic atmosphere.</task>
        <semantic>A cyberpunk city at night with neon lights and flying cars.</semantic>
        
        Example 2:
        <task>Generate an image with a pig and a dog in a forest. The appearance of the pig is based on reference image './temp/pig.png'.</task>
        <semantic>A cyberpunk city at night with neon lights and flying cars.</semantic>
        
        Example 3:
        <task>Replace the dog in the image with a pig, and replace the apple on the table with an orange.  Image to be edited: './outputs/tool_out.png'</task>
        <semantic>A pig and a cow are in a pasture, with a table nearby. There is an orange on the table.</semantic>
        
        **Please pay attention to distinguish between editing tasks and generation tasks. If it's not a generation task, please avoid using the word "generate."**
        **The target image semantics should describe the output image after executing the task and must not conflict with the task.**
        **Tasks and semantics should be as concise as possible while retaining key elements, avoiding excessive redundant or irrelevant content.**
        """

IMAGE_ANALYST_DESCRIPTION = """
    You are highly skilled at analyzing image information provided by users. 
    Please analyze the image content as thoroughly as possible and provide a closely matching description.

    """

IMAGE_DESCRIPTION_PROMPT = """Please describe the image in no more than 50 tokens, clearly stating the key entities, spatial relationships, """

EVALUATOR_INSTRUCTIONS = """
        Please provide your reasoning process first within <think></think>.
        After evaluating all above dimensions, summarize a final evaluation based on all the above <evaluation_xxx> results within 20 words.
        Output format:
        <final_evaluation>your overall evaluation highlighting errors and missing elements</final_evaluation>
        
        **Note: When providing the explanation (evaluation), specify the exact errors in a concise manner.**
        """

PLANNER_DESCRIPTION = """
        You are highly skilled at analyzing the given task and the information provided by the user.
        By leveraging the available tool functionalities, you can decompose the user's task into logical and reasonable processing steps (excluding evaluation of results),
        enabling step-by-step handling to produce the desired output.
        """

PLANNER_EXPERIENCE_PROMPT = """
                The following are the operations experiences that best match the given task, presented in the form of a task graph:
                {}
                The provided experiences are for reference only. 
                """

PLANNER_INITIAL_SOURCE = """
            task: {}
            
            Current image semantics: {}
            
            Target image semantics: {}
            
            ========================================================================
            Here are the available features you can use to complete the task:
            1. Image generation functionality allows you to synthesize output images that match the intended semantics based on task requirements. 
            You can specify the image using plain text to describe its semantics and the layout of key objects (including their positions and relationships). 
            If a reference image is available, it can be used to guide the appearance of specific objects, the overall edge layout, human poses, or the spatial depth of the generated image.
            
            2. Image editing functionality allows you to apply one or more of 24 edit types to a given source image, gradually aligning its semantics with those of the target image:
            Under Object Manipulation and Transformation, there are operations like adding, removing, replacing, moving, resizing, rotating, and duplicating objects. 
            Color and Tone Adjustments include changing color and applying filters/weather effects. Texture and Material Adjustments involve changing texture. 
            For Background and Environment, there’s changing the background, and for Lighting and Shadows, adjusting lighting. In Text and Symbols, actions are adding, 
            removing text, and changing text properties. Composition and Cropping covers cropping images, zooming in/out, and reframing composition. Pose and Expression 
            includes changing pose and facial expression. Special Effects involve adding/removing particles and special effects.
            
            ========================================================================
            
            Please analyze the **target image semantics**, **task requirements**, and **historical operation information** (if available), 
            then provide the next processing step to gradually meet the final task requirements through subsequent multi-round 
            interactions. Each operation should not exceed 30 words.
            
            Present the output using the following format: 
            <operation>your operation.</operation>

            **Requirement: The operations must be described specifically, especially when involving the generation or modification of an object or region. **
            **For example, use ''remove the apple on the far right” instead of vague expressions like 'remove the extra apple.'**
            
            Please propose the most appropriate operations for the user's given task, and present them in the **specified format**.
            **If multiple functions can complete the task, choose the operation with the highest success rate.**
            
            **If it is a generation task, the output image as a whole needs to be made as natural and harmonious as possible.**
            **If it is an editing task, new images should not be generated arbitrarily to deceive the evaluation.**
            
            Please also specify which round the current "operation" depends on. 
            "depends on" means this operation will use the image output path from the specified round as input for the current round's operation.
            The output format is : <depend>round x</depend> (For example, <depend>round 1</depend>)
            If no dependency is required (i.e., the operation starts again from the root node), then output: <depend>None</depend>.
            
            The selected dependency does not necessarily have to be the immediately preceding round. 
            If a dependency exists, it is generally the subsequent operation that is an edit.
            **Please consider the necessity of dependent nodes. When necessary, you may operate from scratch without relying on dependent nodes. (Only for generation tasks.)**
            
            Example 1:
            <operation>Generate a pig and a dog in a field.</operation>
            <depend>None</depend>
            
            Example 2:
            <operation>Add a chicken to the right of the dog in the image.</operation>
            <depend>round 2</depend>
            
            Please provide your current round's operation (No need to provide steps after this round.):
            **The provided operations should be concise while retaining essential elements.**
            """

PLANNER_NEXT_SOURCE = """
                task: {}
                
                Target image semantics: {}
                
                Historical operations information (task graph):
                {}
                The score range is from 0 to 1.
                
                ========================================================================
                Here are the available features you can use to complete the task:
                1. Image generation functionality allows you to synthesize output images that match the intended semantics based on task requirements. 
                You can specify the image using plain text to describe its semantics and the layout of key objects (including their positions and relationships). 
                If a reference image is available, it can be used to guide the appearance of specific objects, the overall edge layout, human poses, or the spatial depth of the generated image.
                
                2. Image editing functionality allows you to apply one or more of 24 edit types to a given source image, gradually aligning its semantics with those of the target image:
                Under Object Manipulation and Transformation, there are operations like adding, removing, replacing, moving, resizing, rotating, and duplicating objects. 
                Color and Tone Adjustments include changing color and applying filters/weather effects. Texture and Material Adjustments involve changing texture. 
                For Background and Environment, there’s changing the background, and for Lighting and Shadows, adjusting lighting. In Text and Symbols, actions are adding, 
                removing text, and changing text properties. Composition and Cropping covers cropping images, zooming in/out, and reframing composition. Pose and Expression 
                includes changing pose and facial expression. Special Effects involve adding/removing particles and special effects.
                
                ========================================================================
                
                Please analyze the **target image semantics**, **task requirements**, and **historical operation information** (if available), 
                then provide the next processing step to gradually meet the final task requirements through subsequent multi-round 
                interactions. Each operation should not exceed 30 words.
                
                Present the output using the following format: 
                <operation>your operation.</operation>

                **Requirement: The operations must be described specifically, especially when involving the generation or modification of an object or region. **
                **For example, use ''remove the apple on the far right” instead of vague expressions like 'remove the extra apple.'**
                
                Please propose the most appropriate operations for the user's given task, and present them in the **specified format**.
                **If multiple functions can complete the task, choose the operation with the highest success rate.**
                
                **If it is a generation task, the output image as a whole needs to be made as natural and harmonious as possible.**
                **If it is an editing task, new images should not be generated arbitrarily to deceive the evaluation.**
                
                Please also specify which round the current "operation" depends on. 
                "depends on" means this operation will use the image output path from the specified round as input for the current round's operation.
                The output format is : <depend>round x</depend> (For example, <depend>round 1</depend>)
                If no dependency is required (i.e., the operation starts again from the root node), then output: <depend>None</depend>.
                
                The selected dependency does not necessarily have to be the immediately preceding round. 
                If a dependency exists, it is generally the subsequent operation that is an edit.
                **Please consider the necessity of dependent nodes. When necessary, you may operate from scratch without relying on dependent nodes. (Only for generation tasks.)**
                
                Example 1:
                <operation>Generate a pig and a dog in a field.</operation>
                <depend>None</depend>
                
                Example 2:
                <operation>Add a chicken to the right of the dog in the image.</operation>
                <depend>round 2</depend>
                
                Please provide your current round's operation (No need to provide steps after this round.):
                **The provided operations should be concise while retaining essential elements.**
                """

WORKER_DESCRIPTION = """
        You manage all the tools used for image generation or editing. You are very skilled at selecting and using the appropriate tools based on the given task.
        """

WORKER_PREFERENCES_SOURCE = """
            Task : {}
            
            1. You have two types of tools to choose from: **text2image-generation tool**, **image-editing tool**.
            Please choose the appropriate tool-type based on the task requirements.
            Please output in the xml format: <category>your select tools-type</category>
            
            2. If you choose the 'text2image-generation tool' category, please analyze the task requirements and assign weights for the following preferences:
            **'color', 'shape', 'texture', '2D-spatial', '3D-spatial', 'numeracy', 'non-spatial'**
            'color' indicates a requirement for the object’s color in the generated image.
            'shape' indicates a requirement for the object’s shape in the generated image.
            'texture' indicates a requirement for the object’s material or surface quality in the generated image, such as 'wooden', 'metallic', etc.
            '2D-spatial' indicates a requirement for the 2D spatial relationships between objects in the generated image, such as 'on the side of', 'on the left', 'on the top of', 'next to', etc.
            '3D-spatial' indicates a requirement for the 3D spatial relationships between objects in the generated image, such as 'behind', 'hidden by', 'in front of', etc.
            'numeracy' indicates a requirement for the number of objects in the generated image.
            'non-spatial' indicates a requirement for non-spatial relationships between objects in the generated image, such as 'A is holding B', 'C is looking at D', 'E is sitting on F', etc.
            Please output in the xml format: 
            <preference>
            <color>your 'color' weight</color>
            <shape>your 'shape' weight</shape>
            <texture>your 'texture' weight</texture>
            <2D-spatial>your '2D-spatial' weight</2D-spatial>
            <3D-spatial>your '3D-spatial' weight</3D-spatial>
            <numeracy>your 'numeracy' weight</numeracy>
            <non-spatial>your 'non-spatial' weight</non-spatial>
            </preference>
            Note: Each weight should be a decimal between 0 and 1; the higher the value, the greater the task's requirement for that preference.
            Unneeded preferences can be directly set to 0.
            
            3. If you choose the 'image-editing tool' category, please analyze the task requirements and assign weights for the following preferences:
            **'addition', 'removement', 'replacement', 'attribute-alter', 'motion-change', 'style-transfer', 'background-change'**
            'addition' indicates that the task involves adding objects to the image.
            'removement' indicates that the task involves removing objects from the image.
            'replacement' indicates that the task involves replacing objects in the image.
            'attribute-alter' indicates that the task involves modifying the attributes of objects in the image.
            'motion-change' indicates that the task involves modifying the actions or motions of objects in the image.
            'style-transfer' indicates that the task involves modifying the overall style of the image.
            'background-change' indicates that the task involves modifying the background of the image.
            Please output in the xml format: 
            <preference>
            <addition>your 'addition' weight</addition>
            <removement>your 'removement' weight</removement>
            <replacement>your 'replacement' weight</replacement>
            <attribute-alter>your 'attribute-alter' weight</attribute-alter>
            <motion-change>your 'motion-change' weight</motion-change>
            <style-transfer>your 'style-transfer' weight</style-transfer>
            <background-change>your 'background-change' weight</background-change>
            </preference>
            Note: Each weight should be a decimal between 0 and 1; the higher the value, the greater the task's requirement for that preference.
            Unneeded preferences can be directly set to 0.
            
            Example 1:
            <category>text2image-generation tool</category>
            <preference>
            <color>0.7</color>
            <shape>0.5</shape>
            <texture>0.1</texture>
            <2D-spatial>0.3</2D-spatial>
            <3D-spatial>0.6</3D-spatial>
            <numeracy>0.</numeracy>
            <non-spatial>0.9</non-spatial>
            <preference>
            
            Example 2:
            <category>image-editing tool</category>
            <preference>
            <addition>0.4</addition>
            <removement>0.9</removement>
            <replacement>0.1</replacement>
            <attribute-alter>0.</attribute-alter>
            <motion-change>0.4</motion-change>
            <style-transfer>0.3</style-transfer>
            <background-change>0.5</background-change>
            </preference>
            
            Please provide your answer.
        """

TOOL_ERROR_EVALUATION = """The result of this operation is unsatisfactory. Please reconsider the operation at this node or analyze whether the dependent pipeline is correct."""
