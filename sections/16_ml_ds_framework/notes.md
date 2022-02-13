Data collection
Data modelling
    - data set -> ML -> insights
Deployment
    - users, API


1. Create a framework
2. Match to data science and machine learning tools
3. Learn by doing

YES
    - write code
    - make mistakes
    - build projects

NO
    - overthink the process
    - try to make things perfect
    - build things from scratch

## Overview
The framework we'll be using:
1. Problem definition
    - "What problem are we trying to solved?"
2. Data
    - "What kind of data do we have?"
        - structured
        - unstructured
3. Evaluation
    - "What defines success for us?"
4. Features
    - "What do we already know about the data?"
5. Modelling
    - "Based on our poblem and data, what model should we use?"
        - problem 1 -> model 1
        - problem 2 -> model 2
        ...
6. Experimentation
    - "How could we improve/what can we try next?"


When shouldn't you use machine learning?
    - Will a simple hand-coded instructionn based system work?

Main types of machine learning
    - Supervised learning
        - classification
            binary classification
                "is this example one thing or another?"
            multi-class classification
        - regression
            how much will this house sell for?
    - Unsupervised learning
        - 
    - Transfer learning
        - find existing model that classifies cars and transfer knowledge to classify dogs
        - model may already know how to find shapes, edges, etc.
    - Reinforcement learning
        - let it do stuff, reward or penalize it


## Detailed look
1. Matching your problem
    - "I know my inputs and outputs" -> supervised learning
    - "I'm not sure of the outputs but I have inputs" -> Unsupervised
    - "I think my problem may be similar to something else" -> transfer learning

2. Data
    - Structured -> can be organized into a table
        - static
        - the more data the better
    - Streaming data -> 
    - Unstructured -> images, audio
    - Data science workflow
        1. static data sources
        2. jupyter notebook
        3. pandas to organise the dtaa
        4. matplotlib to viusalize
        5. machine learning model

3. Evaluation
    - "what defines success for us?"
    - "for this project to be worth pursuing further, we need a machine learning model with over 99% accuracy.
    - types of metrics
        - Classification
            - Accuracy
            - Precision
            - Recall
        - Regression
            - Mean absolute error (MAE)
            - Mean squeared error (MSE)
            - Root mean squared error (RMSE)
        - Recommendation
            - Precision at K

4. Features
    - "What do we already know about the data?"
    - Different features of data
        - Numerical features
        - Categorial features
    - Feature engineering:
        Looking at different features of data and creating new ones/altering existingin ones.
    - What features should you use?
        - Feature coverage: "How many samples have different features? Ideally, every sample has the same features."

5. Modelling - Splitting Data
    - "What kind of model should we use?"
    Three parts to modelling
    1. Choosing and training a model
    2. Tuning a model
    3. Model comparison

   - The most important concept in machine learning:
    [---------------YOUR DATA----------------------]
                    split
                    vvvv               vvv      vvv
    [-----------TRAINING---------][validation][test]
    TRAINING: train your model
    VALIDATION: tune your model
    TEST: test and compare on this
    (T/S/T) 70%-15%-15% for example
            80%-10%-10%

    Generalization: The ability for a machine learning model to perform well on data it hasn't seen before.

    When things go wrong:
        - over fitting

   1. Choosing and traing a model:
        - Problem 1 -> Model 1
        - Problem 2 -> Model 2

        For now,
        Structure Data
            - CatBoost
            - dmlc XGBoost
            - Random Forest
        Unstructured Data
            - Deep learning
            - Transfer learning
    2. Traing a model:
        - traing data
        - X (data), Y (label)
        Goal: Minimise time between experiments
            - use like 10% of your data set to do some quick validation
        
        Things to remember
            - Some models work better than others on different problems
            - Don't be afraid to try things
            - Start small and build up (add complexity) as you need
       
    3. Tuning a model
        - validation data
        
        Random forest
            - 3 trees vs 5 trees

        Neural Networks
            - 2 hidden layers
            - 3 hidden layers

        Things to remember
            - machine learning models have hyperparameters you can adjust
            - a models first results aren't its last
            - tuning can take place on training or validation data sets

    4. Comparison
        "How will our model perform in the real world?"
        - like the final test
        - adapt data 

        Testing a model
            - 98% on training data
            - 96% on test data

        Underfitting: perforamnce(test data) << performance(training data)
        Goldilocks: perforamnce(test data) < performance(training data)
        Overfitting: performance(test data) > perforamnce(training data)

        Don't allow data leakage!! (training and test data) --> overfit

        Data mismatch, test on same type of test data as training data

        FIX Underfitting
            + try a more advanced model
            + increase model hyperparameters
            - reduce amount of features
            + train longer

        FIX Overfitting
            + collect more data
            - try a less advanced model

        Things to remember
            - want to avoid overfitting and underfitting (head towards generality)
            - keep the test set separate at all costs
            - compare apples to apples
            - one best performance metric does not equal best model


6. Experimentation
    - "How could we improve/what can we try next?"
    - this has been an iterative process


Tools we will use:
    - Jupyter
    - matplotlib
    - NumPy
    - pandas
    - scikit-learn
    - CatBoost
    - dmlc XGBoost
