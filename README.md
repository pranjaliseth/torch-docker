# For Training

- `lung-segmentation-torch-ssl.ipynb` -- executes multi-tasking self-supervised objective with the supervised objective for image segmentation.
- `lung-segmentation-torch-sup.ipynb` -- executes supervised objective for image segmentation. 

This repository builds on [lucidrain's DINO PyTorch implementation](https://github.com/lucidrains/vit-pytorch).

For reproducibility: All the required libraries are mentioned in requirements.txt and Docker version 27.1.1.



# For Deployment

Once you have trained a model and what to deploy it for inference using docker, create a docker image using the following command:

`docker build -t torch-flask-app .`

Once the docker image is built, use the following code to run the flask instance of the pre-trained model loaded using the docker image
`docker run -p 6001:5000 torch-flask-app`
