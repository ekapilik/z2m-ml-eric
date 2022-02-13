## What is deep learning?
Machine learning is turning things (data) into numbers and finding patterns in those numbers.
How? Code and math.

Artificial Intelligence
  Machine Learning
    Deep Learning

Traditional programming vs Machine learning algorithm
Traditional: Inputs -> Rules -> Output
Machine learning algorithm: Inputs -> Ideal Output -> Rules



## Why use machine learning?
For a complex problem, can you think of all the rules?
Example, self-driving -> all the rules and contexts are huge

"Use ML for anything as long as you can convert it into numbers and program it to find patterns." - some YouTube comment

"If you can build a simple rule-based system that doesn't require machine learning, do that."
- Rule 1 of Google's Machine Learning Handbook.
https://developers.google.com/machine-learning/guides/rules-of-ml



What is deep learning good for?
- Problems with long lists of rules -- when the traditional approach fails, machine learning/deep learning may help.
- Continually changing environments -- deep learning can adapt ('learn') to new scenarios.
- Discovering insights within large collections of data -- can you imagine trying to hand-craft rules for what 101 different kinds of food look like?

What is deep learning not good for?
- When you need explainability -- the patterns learned by a deep learning model are typically uninterpretable by a human.
- When the traditional approach is a better option -- if you can accomplish what you need with a simple rule-base system.
- When errors are unacceptable -- since the outputs of deep learning model aren't always predicatable.
- When you dont' have much data -- deep learning models require a lot of data



Machine Learning vs Deep Learning
- Machine learning:
  - good for structured data
    - something in a database
  - common "shallow" algorithms
    - Random forest
    - Naive bayes
    - Nearest neighbour
    - Support vector machine
    - ...many more


- Deep Learning:
  - good for unstructured data
    - natural language text
    - image recognition
    - summarizing text
  - common algorihtms
    - *neural networks
    - *fully connected neural network
    - *convolutional neural network
    - *recurrent neural network
    - transformer
    - ... many more

    ** -> the types we will look a twith tensor flow



## What are neural networks?

Definition:
  https://en.wikipedia.org/wiki/Neural_network
  In a modern sense, it is network of artificial neurons or nodes, used for solving artificial intelligence problems.
  The connections between nodes are modelled as weights.
  A positive weight reflects an excitatory connection, while negative values mean inhibitory connections. 
  All inputs to a node are weighted and summed (linear combination).
  Finally, an activation function controls the amplitude of the output. It could be between 0 and 1, or -1 and 1.


Behaviourly:
  Inputs ->  
  Numerical encoding 'tensor' -> 
  Neural network learns representations (patterns/features/weights) ->
  Representation outputs ->
  Convert to human understandable format

Anatomy of Neural networks
  - Input layer (data goes in)
  - Hidden layer(s) (learns patterns in data)
  - Output layer (outputs learned representation or prediction posibilities)
"Patterns" is an arbitrary term, you'll often hear "embedding", "weights", "feature representation", "feature vectors" all referring to similar things.


Types of Learning
  - Supervised learning
    - data + labels
  - Semi-supervised learning
    - data + some labels
  - Unsupervised learning
    - find patterns
  - Transfer learning
    - take pattern from one deep learning model and use it on another problem



What is deep learning actually used for?
  - recommendation systems
  - sequence to sequence
    - translation
    - speech recognition
  - classification/regression
    - **computer vision**
    - natural language processing





What is TensorFlow?
- End-to-end platform for machine learning
- Write fast deep learning code in Python/other accessible languages (able to run on a GPU/TPU)
- Able to access many pre-built deep learning models (TensorFLow Hub)
- whole stack: preprocess data, model data, deploy model in your application
- Originally designed and used in-house by Google (now open-source)

Why TensorFlow?
https://www.tensorflow.org/
- easy model building
- robust ML production anywhere
- powerful experimentation for research

Can run on GPU/TPU
- GPU: Graphics Processing UNit -> crunch numbers, find patterns in number
- TPU: Tensor Processing Unit -> faster optimized chip by Google for NN



What is a Tensor?
- Numerical encoding of information
- could be anything
https://www.youtube.com/watch?reload=9&v=f5liqUk0ZTw
- they represent the same data in all frames of reference

- scalars are rank-0 tensors because there are no directional components 
- vectors (Ax, Ay, Az) are rank-1 tensors because each component has a single directional vector
- rank-1 tensor: nine components and nine sets of two basis vectors 
  ((Axx, Axy, Axz),
   (Ayx, Ayy, Ayz),
   (Azx, Azy, Azz))
   for example, in 3D, Axx is the force acting on a plane pointing in the x direction. Axy is the force
- rank-2 tensor: 27 components...

The benefit??? 
- All observers, in all reference frames, agree. Not on the basis vectors, not on the components. But on the combination of components and basis vectors.
  The reason is that the basis vectors transform one way between reference frames, and the components transform in just such a way so as to keep the combination of components and basis vectors the same for all observers.

What is the difference between a matrix and a tensor?
https://medium.com/@quantumsteinke/whats-the-difference-between-a-matrix-and-a-tensor-4505fbdc576c
  - matrix is a grid (n x m), add/subtract same size, multiple compatible sizes, multiply by a constant. 2D grid of numbers.
  - tensor is a generalization of a matrix. It could be 1-D matrix (vector), 3-D matrix (cube of numbers). dimension=rank

  Tensor: a mathematical entity that lives in a structure and interacts with other mathematical entities.
  If one transforms the other entities in the structure in a regular way, then the tensor must obey a related transformation rule.
  "Dynamical" property -> key distinguishing factor



  What we're going to cover (broadly)
    - TensorFlow basics and fundamentals
    - Preprocessing data (getting it into tensors)
    - Building and using pretrained deep learning models
    - Fitting a model to the data (learning patterns)
    - Making predictions with a model (using patterns)
    - Evaluating model predictions
    - Saving and loading models
    - Using a trained model to make predictions on custom data

  - Experiment lots!!

TensorFlow workflow
1. Get data ready (turn into tensors)
2. Build or pick a pretrained model (to suit your problem)
3. Fit the model to the data and make a prediction
4. Evaluate the model
5. Improve through experimentation
6. Save and reload your trained model


How to approach this course
  - write code (lots of it, follow along, let's make mistakes together)
    - Motto #1: "If in doubt, run the code"
  - Explore & experiment
    - Motto #2: "Experiment, experiment, experiment"
    - Motto #3: "Visualize, visualize, visualize" (recreate things in a way you can understand them.
  - Ask questions (including the "dumb" ones)
  - Do the exercuses (try them yourself before looking at solutions)
  - Share your work, teach someone else.
  - AVOID:
    - Overthinking the proces
    - The "I can't learn it" mentality (that's bullshit..)
