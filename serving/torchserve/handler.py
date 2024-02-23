from diffusers import AutoPipelineForText2Image

import torch
import os
import os.path as osp
import time
import numpy as np
from ts.torch_handler.base_handler import BaseHandler

class ModelHandler(BaseHandler):
    def __init__(self):
        self._context = None
        self.initialized = False
        self.model = None
        self.device = None
    
    def initialize(self, context):
        """
        Invoke by torchserve for loading a model
        :param context: context contains model server system properties
        :return:
        """

        #  load the model
        self.manifest = context.manifest

        properties = context.system_properties
        self.device = torch.device("cuda:" + str(properties.get("gpu_id")) if torch.cuda.is_available() else "cpu")
        self.model = AutoPipelineForText2Image.from_pretrained("stabilityai/sd-turbo")
        self.model.to("cuda")
        self.initialized = True

    
    def handle(self, data, context):
        start_time = time.time()

        self._context = context
        metrics = self._context.metrics

        if not self.initialized:
            self.initialize(context)
        if data is None:
            return None

        prompt = data[0]['body']['inputs'][0]['data'][0]
        print(prompt)
        image = self.model(prompt=prompt, num_inference_steps=1, guidance_scale=0.0).images[0]

        stop_time = time.time()
        metrics.add_time(
            "HandlerTime", round((stop_time - start_time) * 1000, 2), None, "ms"
        )
        image.save("/data/ephemeral/home/boostcamp_level3_demo/result.jpg")
        image = np.array(image).tolist()
        return [image]
