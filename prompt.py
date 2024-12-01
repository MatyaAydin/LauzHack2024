
from langchain.llms import OpenAI




def get_prompt_request(promt_request):
    prompt_request = "How many boats passed from 5pm to 8pm?"

    prompt = f"""
    Given the following request:
    {prompt_request}


    Answer only in the following format:
    count-start-end

    where start and end are the starting and ending hours the user want the count to end. Use a 24 hours scale
    """
    return prompt

#answer = llm.invoke(prompt)


answer_to_question = 5
def get_prompt_answer(prompt_request,answer_to_question):
    prompt_answer = f"""
    Someones asked the following: {prompt_request}

    We got the following answer: {answer_to_question}

    Format a nice answer to the question
    """
    return prompt_answer

def get_prompt_answer_image(query):

    COCO_CLASSES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella',
        'N/A', 'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
        'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
        'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',
        'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
        'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    prompt = f"""
            Given the following question:

            {query}

            Answer only with the most relevant item in the following list:
            {COCO_CLASSES}
            """
    
    return prompt


#final_answer = llm.invoke(prompt_answer)



#print(final_answer)