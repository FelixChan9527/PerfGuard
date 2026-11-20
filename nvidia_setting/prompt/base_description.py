BASE_DESCRIPTION=\
{
    "FLUX": {
        "description":  # 简要介绍该工具是什么，用途是什么，用英文。
                        """
                        FLUX is a text-to-image generation tool that produces images with highly realistic object textures and achieves a high degree of alignment with textual semantics.
                        """,
        "usage": {      
                    "input": {
                        "<text>": {"description": "input text prompt", "type":str, "example": "A cat holding a sign that says hello world."},
                        "<i-img>": {"description": "input reference image path", "type":str, "example": "./temp/test.png"},
                        "<o-img>": {"description": "generate image path", "type":str, "example": "./temp/flux_out.png"},
                    },
                    "output":{  # 输出标识，可支持多个输出，并给出样例
                        "<img>": {"description": "generate image path", "type":str, "example": "./temp/flux_out.png"},
                    }
        },
    }
}