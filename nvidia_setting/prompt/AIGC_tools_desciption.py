AIGC_DESCRIPTION=\
{
    "FLUX": {
        "category": "text2image-generation tool",
        "description":
                        """
                        Image generation tool. A text-to-image generation tool that produces images with highly realistic object textures and achieves a high degree of alignment with textual semantics. 
                        """,
        "usage": {
                    "input": {
                        "<text>": {"description": "Output image semantics", "type":str, "example": "A cat holding a sign that says hello world."},
                    },
                    "output":{  # 输出标识，可支持多个输出，并给出样例
                        "<save_path>": {"description": "generate image path", "type":str, "example": "./temp/flux_out.png"},
                    }
        },
        "preference": {
            "color": 0.7407, "shape": 0.5718, "texture": 0.6922, "2D-spatial": 0.2863, "3D-spatial": 0.3866, "numeracy": 0.6185, "non-spatial": 0.9213
        } 
    },

    "EasyControl": {
        "description": 
                        """
                        Image generation tool. This tool is used for customized generation based on a reference image and text.                        
                        """,
        "usage": {
                    "input": {
                        "<text>": {"description": "Output image semantics","type": str,"example": "A SKS in the library"},
                        "<image>": {"description": "input subject image path","type": str,"example": "./EasyControl/subject_0.png"},
                    },
                    "output": {
                        "<save_path>": {"description": "generated image path","type": str,"example": "./test_image/output/generated_subject.png"}
                    }
        }
    },
    "DreamO": {
        "description": 
                        """
                        This tool is used for customized generation based on a reference image and text.                        
                        """,
        "usage": {
                    "input": {
                        "<text>": {"description": "Output image semantics", "type": str, "example": "a woman playing guitar in the street"},
                        "<image>": {"description": "reference IP image path", "type": str, "example": "DreamO/example_inputs/woman1.png"},
                    },
                    "output": {
                        "<save_path>": {"description": "path to the generated image", "type": str, "example": "test_image/output/dreamo_out.png"}
                    }
        }
    },
    "Layout_to_Image": {
        "category": "text2image-generation tool",
        "description": 
                        """
                        Image generation tool. Generate an image based on a text prompt and given bounding boxes with object descriptions.
                        """,
        "usage": {      # 在此处输入用法，写明对应的输入输出，输入输出标记符用<>来表示。后续使用时，条件会以<c1>condition1</c1>的形式来使得LLM进行输入和输出，并方便解析代码
                    "input": {  # 输入条件标识，可支持多个输入，并给出样例，不能重复
                        "<prompt>": {"description": "Output image semantics", "type":str, "example": "Seven pigs on the left, eight cows on the right."},
                    },
                    "output":{  # 输出标识，可支持多个输出，并给出样例
                        "<save_path>": {"description": "generate image path", "type":str, "example": "./temp/lauout2img.png"},
                    }
        }
    },
    "Step1X_Edit_ImageEditTool": {
        "category": "image-editing tool",
        "description": 
                        """
                        Image editing tool. ImageEditTool can edit the input image based on the instructions provided in the text.
                        """,
        "usage": {
                    "input": {
                        "<original_img_path>": {"description": "Path to the original image","type": str,"example": "inputs/original_image.png"},
                        "<instruction>": {"description": "Editing instruction","type": str,"example": "Please add a woman to the right of this person."}
                    },
                    "output": {
                        "<save_path>": {"description": "File path of the saved final image","type": str,"example": "'./output/generated_image.png'"}
                    }
        },
        "preference": {
            "addition": 0.78, "removement": 0.522, "replacement": 0.69, "attribute-alter": 0.626, "motion-change": 0.686, "style-transfer": 0.888, "background-change": 0.638
        }
    },
    "AnySD_EditTool": {
        "category": "image-editing tool",
        "description": 
                        """
                        Image editing tool. AnySD_EditTool enables diverse editing of images based on prompt texts
                        """,
                        # """
                        # The tool can modify the image background based on editing instructions.
                        # """,
        "usage": {
                    "input": {
                        "<instruction>": {"description": "instruction for image editing, e.g. 'Make it underwater'","type": str,"example": "Make it underwater"},
                        "<image_path>": {"description": "Local path of the input image","type": str,"example": "./assets/bear.jpg"}
                    },
                
                "output": {
                    "<save_path>": {"description": "Path of the output generated image","type": str,"example": "./outputs/AnySDEditTool_88_0.png"}
                }
        },
        "preference": {
            "addition": 0.624, "removement": 0.468, "replacement": 0.542, "attribute-alter": 0.532, "motion-change": 0.662, "style-transfer": 0.654, "background-change": 0.474
        }
    },
    "UltraEdit_Tool": {
        "category": "image-editing tool",
        "description": 
                        """
                        Image editing tool. UltraEdit_Tool enables advanced and controllable image editing based on provided prompt texts.
                        """,
                        # """
                        # The tool can modify the image style based on editing instructions.
                        # """,
        "usage": {
                    "input": {
                        "<image_path>": {"description": "Path of the image to be edited.","type": str,"example": "./test/img.jpg"},
                        "<instruction>": {"description": "Instruction to edit the image.","type": str,"example": "Change the image style to anime."}
                    },
                    "output": {
                        "<save_path>": {"description": "Path to the edited image.","type": str,"example": "./outputs/UltraEdit_Tool_23_0.png"}
                    }
        },
        "preference": {
            "addition": 0.726, "removement": 0.342, "replacement": 0.626, "attribute-alter": 0.602, "motion-change": 0.714, "style-transfer": 0.738, "background-change": 0.662
        }
    },

    "IPAdapterPlus": {
        "description": 
                        """
                        Image generation tool. This is a method that combines reference images and textual semantics to achieve customized image synthesis. 
                        The entities in the input image correspond to the key entities mentioned in the text. 
                        """,
        "usage": {
                    "input": {
                        "<image_path>": {"description": "Reference identity image path.","type": str,"example": "test_image/human5.png"},
                        "<prompt>": {"description": "Natural language description for controlling the style or content of the image editing.","type": str,"example": "best quality, wearing sunglasses on the beach"}
                    },
                    "output": {
                        "<save_path>": {"description": "Path of the generated image after saving","type": str,"example": "outputs/IPAdapterPlusImageEditTool_41_0.png"}
                    }
        }
    },

    "SD3_Tool": {
        "category": "text2image-generation tool",
        "description": 
                        """
                        Image generation tool. A text-to-image generation tool that produces images with highly realistic object textures and achieves a high degree of alignment with textual semantics.
                        """,
        "usage": {
                    "input": {
                        "<prompt>": {"description": "input text prompt","type": str,"example": "A cat holding a sign that says hello world"}
                    },
                        "output": {"<save_path>": {"description": "Path of the generated image after saving","type": str,"example": "outputs/SD3_ModelScope_Tool_42_0.png"}
                    }
        },
        "preference": {
            "color": 0.8132, "shape": 0.5885, "texture": 0.7334, "2D-spatial": 0.3200, "3D-spatial": 0.4084, "numeracy": 0.6174, "non-spatial": 0.9093
        }
    },
    "Pixart": {
        "category": "text2image-generation tool",
        "description": 
                        """
                        Image generation tool. A text-to-image generation tool that produces images with highly realistic object textures and achieves a high degree of alignment with textual semantics.
                        """,
        "usage": {
                    "input": {
                        "<prompt>": {"description": "input text prompt","type": str,"example": "A cat holding a sign that says hello world"}
                    },
                        "output": {"<save_path>": {"description": "Path of the generated image after saving","type": str,"example": "outputs/Pixart_42_0.png"}
                    }
        },
        "preference": {
            "color": 0.669, "shape": 0.4927, "texture": 0.6477, "2D-spatial": 0.2064, "3D-spatial": 0.3901, "numeracy": 0.5032, "non-spatial": 0.8620
        }
    }
}



    



