import cv2
import numpy as np
from base64 import b64encode
import websocket #NOTE: websocket-client (https://github.com/websocket-client/websocket-client)
import uuid
import json
import urllib.request
import urllib.parse
import configparser

setting = configparser.ConfigParser()
setting.read('config.ini')

server_address = setting['COMFYUI']['address']
client_id = str(uuid.uuid4())

def readImage(path):
    img = cv2.imread(path)
    retval, buffer = cv2.imencode('.png', img)
    b64img = b64encode(buffer).decode("utf-8")
    return b64img

def queue_prompt(prompt):
    p = {"prompt": prompt, "client_id": client_id}
    data = json.dumps(p).encode('utf-8')
    req = urllib.request.Request("http://{}/prompt".format(server_address), data=data)
    return json.loads(urllib.request.urlopen(req).read())


def get_image(filename, subfolder, folder_type):
    data = {"filename": filename, "subfolder": subfolder, "type": folder_type}
    url_values = urllib.parse.urlencode(data)
    with urllib.request.urlopen("http://{}/view?{}".format(server_address, url_values)) as response:
        return response.read()

def get_history(prompt_id):
    with urllib.request.urlopen("http://{}/history/{}".format(server_address, prompt_id)) as response:
        return json.loads(response.read())

def get_images(ws, prompt):
    prompt_id = queue_prompt(prompt)['prompt_id']
    output_images = {}
    while True:
        out = ws.recv()
        if isinstance(out, str):
            message = json.loads(out)
            if message['type'] == 'executing':
                data = message['data']
                if data['node'] is None and data['prompt_id'] == prompt_id:
                    break #Execution is done
        else:
            continue #previews are binary data

    history = get_history(prompt_id)[prompt_id]
    for o in history['outputs']:
        for node_id in history['outputs']:
            node_output = history['outputs'][node_id]
            if 'images' in node_output:
                images_output = []
                for image in node_output['images']:
                    image_data = get_image(image['filename'], image['subfolder'], image['type'])
                    images_output.append(image_data)
    output_images[node_id] = images_output

    return output_images

def transform_image(workflow_path, img_path, input_positive=None):
    # img = readImage(r'C:\Users\chsjk\Documents\data\real_face\hard\1.png')
    img = readImage(img_path)
    with open(workflow_path, 'r', encoding='utf-8') as f:
        prompt = json.load(f)
    # prompt['38']['inputs']['image'] = img

    key = [key for key in prompt.keys() if prompt[key]['class_type'] == 'ETN_LoadImageBase64'][-1]
    prompt[key]['inputs']['image'] = img

    save_key = [key for key in prompt.keys() if prompt[key]['class_type'] == 'SaveImage'][-1]
    file_name = img_path.split('\\')[-1]
    prompt[save_key]['inputs']['filename_prefix'] = file_name.split('.')[0]

    if input_positive:
        ss = [key for key in prompt.keys() if prompt[key]['class_type'] == 'SDXLPromptStyler'][-1]
        prompt[ss]['inputs']['text_positive'] = input_positive

    ws = websocket.WebSocket()
    ws.connect("ws://{}/ws?clientId={}".format(server_address, client_id))
    images = get_images(ws, prompt)

    for node_id in images:
        for image_data in images[node_id]:
            from PIL import Image
            import io
            image = Image.open(io.BytesIO(image_data))

    return image




