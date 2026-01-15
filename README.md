# Digit Eye

Training a convolutional neural network to recognize and classify handwritten data. Pulled MNIST data (a dataset of 60,000 handwritten digits) to train and test on.
## Demo

![Demo of the Project](./demo.gif)

## Hugging Face

You can test out the model for yourself [here](https://huggingface.co/spaces/waffleshd/digit-eye) on hugging face!
## Run Locally

Clone the project

```bash
  git clone https://github.com/waffleshd/digit-eye
```

Go to the project directory

```bash
  cd digit-eye
```

Install dependencies

```bash
  npm install
```

Start the server

```bash
  gradio app.py
```

The terminal will direct you to a localhost link. When you open it, the site should be there.

## Training

Once you have cloned the repository, you can train your own model, by switching up some parameters in the config.py file:

**LEARNING_RATE** : Affects how "much" the neural net will correct itself during training. Setting a larger number may lead it to "overshoot", where it will struggle to reach the optimal set of weights. Setting this number to a smaller one though, may lead to much longer learning times, or lead it to get stuck in a local minimum. 

**NUM_EPOCHS** : This will determine how many times the model will run through the dataset and train itself. This project uses an optimizer specifically for MNIST data, so even after 5 epochs the model is almost perfect at detecting handwritten digits.

**BATCH_SIZE** : This determines the number of images that will be passed into the model for evaluation every "batch". This is used in optimizers such as Stochastic Gradient Descent (SGD) to prevent hitting local minima. This usually doesn't need to be changed.

Check out [this article](https://www.ibm.com/think/topics/learning-rate) by IBM for more information.
