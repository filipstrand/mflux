# Server for image generation with mflux
# This class (C) 2024 by @orbiter Michael Peter Christen
# This code is licensed under the Apache License, Version 2.0

import os
import io
import sys
import argparse
import time
import base64
import hashlib
import threading
from PIL import Image
from flask import Flask, request, Response, jsonify
from flask_restx import Api, Resource, fields
from flask_cors import CORS

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


from mflux.config.model_config import ModelConfig
from mflux.config.config import Config
from mflux.flux.flux import Flux1
from mflux.post_processing.image_util import ImageUtil

import requests

# Monkey patching the Session to ignore SSL verification
old_request = requests.Session.request

def new_request(self, *args, **kwargs):
    kwargs['verify'] = False
    return old_request(self, *args, **kwargs)

requests.Session.request = new_request

app = Flask(__name__)
api = Api(app, version='1.0', title='MFLUX API Server',
          description='An image generation server. Workflow: /generate -> /status -> /image', doc='/swagger')
CORS(app, resources={r"/*": {"origins": "*"}})

tasklist = []         # list which holds the image computation tasks
flux = None           # the flux object, initialized in main()
pixels = 1024 * 1024  # the number of pixels in all of the computed images (start value)
ctime = 80            # the total computation time for all images in seconds (start value)

# we implement image generation as asynchronous task
# this will be executed in a separate thread
def compute_image_task():
    global flux, tasklist, pixels, ctime
    # we loop forever and in every iteration we check if there is a task to process
    while True:
        if flux == None or len(tasklist) == 0:
            time.sleep(1)
            continue
        
        # loop through the tasklist and get the first task which has no image assigned
        foundimage = False
        for task in tasklist:
            if 'image' in task: continue          
            # found a task without image
            compute_time = time.time()
            task['compute_time'] = compute_time
            # generate the image
            image = flux.generate_image(
                seed=task['seed'],
                prompt=task['prompt'],
                config=Config(
                    num_inference_steps=task['steps'],
                    height=task['height'],
                    width=task['width'],
                    guidance=task['guidance'],
                )
            )
            end_time = time.time()
            ctime += end_time - compute_time
            pixels += task['height'] * task['width']
            
            # convert the image (we do not count this on the computation time on purpose)
            # we do this here and not during retrieval to save memory in the tasklist
            pilimage = image.image
            format = task['format'].upper()
            if format not in ['PNG', 'JPEG']: format = 'JPEG'
            if format == 'PNG':
                png_image = io.BytesIO()
                pilimage.save(png_image, format=format)
                png_image.seek(0)
                task['image'] = png_image
            else:
                quality = task['quality']
                jpeg_image = io.BytesIO()
                pilimage.save(jpeg_image, format=format)
                jpeg_image.seek(0)
                task['image'] = jpeg_image
            
            task['end_time'] = end_time # end time of the task
            foundimage = True
            break
        
        # if we did not found any task without image, we sleep for 1 second
        if not foundimage: time.sleep(1)

def str_to_bool(value):
    return value.lower() in ['true', '1', 't', 'y', 'yes']


# generate image endpoint

task_model = api.model('TaskInput', {
    'prompt': fields.String(description='The textual description of the image to generate.', default='A beautiful landscape', required=True),
    'seed': fields.String(description='Entropy Seed', default=str(int(time.time())), required=False),
    'height': fields.Integer(description='Image height', default=1024, required=False),
    'width': fields.Integer(description='Image width', default=1024, required=False),
    'steps': fields.Integer(description='Inference Steps', default=4, required=False),
    'guidance': fields.Float(description='Guidance Scale', default=3.5, required=False),
    'format': fields.String(description='The image format (JPEG or PNG), default is JPEG', default="JPEG", required=False),
    'quality': fields.Integer(description='JPEG compression quality (1-100) if format is JPEG, default is 85', default=85, required=False),
    'priority': fields.Boolean(description='Set to true to put this task to the head of the queue', default=False, required=False)
})

generate_response_model = api.model('GenerateResponse', {
    'task_id': fields.String(description='ID of the image generation task'),
    'task_length': fields.Integer(description='Length of the image generation task queue excluding this new one'),
    'expected_time_seconds': fields.Float(description='Expected time in seconds for the image generation task to complete')
})

# function which counts number of pixels in images from the tasklist up to a certain index
def count_pixels(index):
    global tasklist
    pixels = 0
    for i in range(index):
        if i >= len(tasklist): break
        task = tasklist[i]
        if not 'image' in task:
            pixels += task['width'] * task['height']
    return pixels

@api.route('/generate')
class GenerateImage(Resource):
    @api.expect(task_model, validate=True)
    @api.response(200, 'Success', generate_response_model)
    @api.response(404, 'Cannot append task')
    def post(self):
        """
        The /generate endpoint is used to generate an image as an asynchronous task.
        This will put the task in the queue and return the task ID.
        The task is either at the end of the queue or at the beginning if priority is set to true.
        To save memory, the image is not stored in it's raw form but in the form demanded by the client.
        Therefore the format has to be declared in the request at generation time in this endpoint.
        """
        global tasklist, pixels, ctime
        # Parse the JSON body into a dictionary
        args = request.json
        prompt = args.get('prompt', 'A beautiful landscape')
        seed = args.get('seed', str(int(time.time())))
        height = int(args.get('height', 1024))
        width = int(args.get('width', 1024))
        steps = int(args.get('steps', 4))
        guidance = float(args.get('guidance', 3.5))
        format = args.get('format', 'JPEG').upper()
        quality = args.get('quality', 85)
        priority = args.get('priority', False)

        start_time = time.time()
        # taskid is a 8-digit hex hash to identify the image
        md5 = hashlib.md5()
        md5.update(str(start_time).encode())
        task_id = md5.hexdigest()[:8]

        task_metadata = {
            'task_id': task_id,
            'prompt': prompt,
            'seed': seed,
            'height': height,
            'width': width,
            'steps': steps,
            'guidance': guidance,
            'format': format,
            'quality': quality,
            'priority': priority,
            'start_time': start_time
        }
        
        # compute waiting time based on the number of pixels in the queue
        wait_for_pixels = width * height # include the current task
        if priority and len(tasklist) > 1:
            wait_for_pixels += count_pixels(1)
            tasklist.insert(1, task_metadata)
        else:
            wait_for_pixels += count_pixels(len(tasklist))
            tasklist.append(task_metadata)

        expected_time_seconds = ctime * wait_for_pixels / pixels
        return {
            'task_id': task_id,
            'task_length': len(tasklist) - 1,
            'expected_time_seconds': expected_time_seconds
        }, 200


# status endpoint

status_model = api.model('Status', {
    'status': fields.String(description='Status of the image generation task'),
    'pos': fields.Integer(description='Position in queue')
})

@api.route('/status')
class GetStatus(Resource):
    @api.doc(params={'task_id': 'The ID of the image generation task'})
    @api.response(200, 'Success', status_model)
    @api.response(404, 'Task not found')
    def get(self):
        """
        The /status endpoint is used to check the image generation progress of a task.
        The returned status can be i.e. when the task is not ready, position 3 in the queue, estimated time remaining 43 seconds:
        { "status": "waiting", "pos": 3, "wait_remaining": 43}
        .. or when the task is done:
        { "status": "done"}
        When the status is "done", the image can be retrieved with the /image endpoint.
        If the task / the task_id is unknown, the endpoint returns a 404 status code.
        """
        task_id = request.args.get('task_id', default='')
        c = -1
        for i, task in enumerate(tasklist):
            if not 'image' in task: c += 1
            if task['task_id'] == task_id:
                if 'image' in task:
                    return jsonify({'status': 'done'})
                else:
                    # compute the remaining time
                    wait_remaining = count_pixels(i + 1) * ctime / pixels
                    start_time = task.get('start_time', 0)
                    compute_time = task.get('compute_time', start_time)
                    wait_remaining = int(wait_remaining - (time.time() - compute_time))
                    if wait_remaining < 1: wait_remaining = 1
                    return jsonify({'status': 'waiting', 'pos': c, 'wait_remaining': wait_remaining})
        return Response(status=404)


# image retrieval endpoint; image format was already defined in the generate endpoint

@api.route('/image')
class GetImage(Resource):
    @api.doc(params={
        'task_id': 'The ID of the image generation task',
        'base64': 'Set to true to return the image as base64 encoded string, default false',
        'delete': 'Set to true to delete the task after getting the image, default is true'
    })
    @api.response(200, 'Success')
    @api.response(404, 'Task not found')
    def get(self):
        """
        The /image endpoint is used to get the produced image after a task has completed.
        The image is already encoded in PNG or JPEG according to the formet given in the /generate endpoint.
        The image can be returned as base64 encoded string or as binary data.
        By default calling this endpoint will delete the task from the queue;
        this means the image can only be retrieved once. To keep the task in the queue set delete to false.
        If the image is not ready at the time of the request, the endpoint returns a 404 status code.
        """
        task_id = request.args.get('task_id', default='')
        for task in tasklist:
            if task['task_id'] == task_id:
                if 'image' in task:
                    image = task['image']
                    format = task['format']
                    base64p = str_to_bool(request.args.get('base64', default='false'))
                    deletep = str_to_bool(request.args.get('delete', default='true'))
                    if deletep: tasklist.remove(task)
                    if base64p:
                        return Response(base64.b64encode(image.getvalue()), mimetype='text/plain; charset=utf-8')
                    else:
                        return Response(image.getvalue(), mimetype='image/png' if format == 'PNG' else 'image/jpeg')
        return Response(status=404)


# cancel task endpoint

@api.route('/cancel')
class CancelTask(Resource):
    @api.doc(params={'task_id': 'The ID of the image generation task'})
    @api.response(200, 'Success')
    @api.response(404, 'Task not found')
    def get(self):
        """
        The /cancel endpoint is used to cancel a task.
        """
        task_id = request.args.get('task_id', default='')
        for task in tasklist:
            if task['task_id'] == task_id:
                tasklist.remove(task)
                return Response(status=200)
        return Response(status=404)


# tasks lising endpoint

task_output_model = api.inherit('TaskOutput', task_model, {
    'task_id': fields.String(description='ID of the image generation task', default=None, required=False),
    'start_time': fields.String(description='Time when the image generation task was submitted', default=None, required=False),
    'compute_time': fields.String(description='Time when the image computation started', default=None, required=False),
    'end_time': fields.String(description='Time when the image generation task ended', default=None, required=False)
})
tasks_model = api.model('Tasks', {
    'tasks': fields.List(fields.Nested(task_output_model), description='List of tasks')
})

@api.route('/tasks')
class GetTasks(Resource):
    @api.response(200, 'Success', tasks_model)
    def get(self):
        """
        The /tasks endpoint is used to list all tasks.
        This can be used to implement a task manager.
        """
        tasklist0 = []
        for task in tasklist:
            task0 = task.copy()
            if 'image' in task0: del task0['image']
            tasklist0.append(task0)        
        return jsonify(tasklist0)


# clear tasks endpoint

@api.route('/clear')
class ClearTasks(Resource):
    @api.response(200, 'Success')
    def get(self):
        """
        The /clear endpoint is used to clear all tasks.
        """
        tasklist.clear()
        return Response(status=200)
   
def main():
    parser = argparse.ArgumentParser(description='Start a server to generate images with mflux.')
    parser.add_argument('--model', "-m", type=str, default="schnell", choices=["dev", "schnell"], help='The model to use ("schnell" or "dev").')
    parser.add_argument('--quantize',  "-q", type=int, choices=[4, 8], default=None, help='Quantize the model (4 or 8, Default is None)')
    parser.add_argument('--path', type=str, default=None, help='Local path for loading a model from disk')
    parser.add_argument('--host', type=str, default='127.0.0.1', help='The host to listen on')
    parser.add_argument('--port', type=int, default=4030, help='The port to listen on')
    args = parser.parse_args()

    if args.path and args.model is None:
        parser.error("--model must be specified when using --path")

    global flux
    flux = Flux1(
        model_config=ModelConfig.from_alias(args.model),
        quantize=args.quantize,
        local_path=args.path
    )

    threading.Thread(target=compute_image_task).start()
    print(f"Server started, view swagger API documentation at http://{args.host}:{args.port}/swagger")
    app.run(host=args.host, port=args.port)

if __name__ == '__main__':
    main()
