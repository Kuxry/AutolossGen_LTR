## Automatic Loss Function Generation for Learning-to-rank

1. This project addresses the challenge of manually designing loss functions by combining AutoML technology with LTR models. By using an automatic loss function generation framework, it is possible to automatically generate optimal loss functions and adapt to different datasets.

2. The effectiveness of the automatically generated loss functions was evaluated through comparative experiments between models using traditionally designed loss functions and models using automatically generated loss functions.

3. This research is applicable to real-world problems in the fields of Information Retrieval (IR) and Learning to Rank (LTR), contributing to the development of more efficient and effective models.

### 1. Background and Key Scientific Questions 
Information Retrieval (IR) is the research field that focuses on providing effective results to users from a large-scale dataset, such as web search and recommendation systems. Learning to rank (LTR) is a key technique in the field of IR. It is a class of techniques that explores how to use machine learning to solve ranking problems. According to recent studies, a traditional LTR architecture consists of two vital parts: the score function and the loss function to optimize the LTR model. 

Furthermore, the current LTR models predominantly rely on manually designed loss functions, which need significant expertise and human effort. To the best of our knowledge, no automatic loss function generation for LTR systems has ever been deployed or proposed.

The choice of the loss function is critical in LTR systems, as a good loss may significantly improve the model performance. This is because the training of an LTR model eventually depends on minimizing the loss function, and the gradient of the loss function supervises the optimization direction of the LTR model. As a result, any inconsistency between the optimization goal and the optimization direction may hurt the model performance.

However, __manually designing an effective loss function is a big challenge due to the complexity of the problem__. There are still many challenging problems. **Firstly**, a large fraction of previous work focuses on handcrafted loss functions, which require comprehensive analysis and understanding of the task, despite the meticulous design of researchers with their expertise and efforts. This is very time-consuming when designing the model. **Secondly**, given different datasets, the best loss could be different. This means it is very difficult to choose the best loss function for different scenarios. Hence, there is a significant need to develop an automatic framework that can help to generate the best loss which is easy to tailor given different settings.

### 2. Novelty
1. Automatic loss generation helps to remove or reduce the manual efforts in loss function design.
2. The loss functions generated from the proposed framework perform better than handcrafted base losses.
3. It can help to generate the best loss tailored to a specific model-dataset combination.


### 3. The Framework of Automatic Loss Function Generation
This research proposes an automatic loss function generation framework for Learning-to-rank (LTR), which is implemented by reinforcement learning (RL) and optimized in iterative and alternating schedules. There are two stages here:

#### Phase I: Loss Search
For the LTR model, this study will use the algorithm of stochastic gradient descent (SGD) to update parameters and apply the RL algorithm to update the RNN controller model. The RNN controller uses reward checking to determine how good or bad the loss function is. After the LTR network is trained and evaluated, a positive or negative reward is provided to update the controller.

#### Phase II: Effectiveness Test
This phase randomly initializes an LTR model and trains the model to convergence using the loss function. It then obtains the final performance for the loss function and keeps the best performing loss as the finally selected loss function.

![image](https://github.com/user-attachments/assets/a735ad19-0c5d-4c2a-999c-558f8aa37996)



### 4. Experimental Setup

#### 4.1 Dataset Description

Our experiments are conducted on two widely-used benchmark datasets of recommender systems, namely, MQ2008 and MSLR-WEB30K, which are both publicly available. The detailed statistics of the datasets are shown in Table 3, and we briefly introduce these two datasets in the following part.

- **MQ2008**
  - It is a learning-to-rank (LETOR) dataset containing 1692 queries. Each query-document pair was represented by a 46-dimensional feature vector and awarded a relevance label ranging from 0 (irrelevant) to 4 (perfectly relevant) over five levels.

- **MSLR-WEB30K**
  - Each query-document pair was represented by a 136-dimensional feature vector and awarded a relevance label ranging from 0 (irrelevant) to 4 (perfectly relevant) over five levels.

#### 4.2 Baseline Losses

We compare with the following baseline losses in the experiment.

- **RankMse**
  - RankMSE is a mean squared error-based loss function that minimizes the discrepancies between predicted and target rankings. It assigns higher scores to more relevant items and lower scores to less relevant items, improving the overall ranking performance.

- **RankNet**
  - RankNet focuses on pairwise comparisons and learns a ranking function by comparing pairs of items. It optimizes the model's parameters using a binary cross-entropy loss, capturing pairwise preferences and enhancing the ranking accuracy.

- **LambdaRank**
  - LambdaRank aims to optimize listwise ranking effectiveness by considering both pairwise preferences and relevance differences between items. It directly optimizes a ranking measure, such as NDCG, using a ranking-oriented cost function, ensuring that the model assigns scores aligned with the desired ranking.

#### 4.3 Evaluation Metrics

To evaluate the final performance of the generated loss functions, we use the Normalized Discounted Cumulative Gain (NDCG) for evaluating both the classification and regression tasks. NDCG is a widely used evaluation metric in learning-to-rank models, which takes into account the relevance and ranking positions of the items in the predicted list. By measuring the quality of the ranking order and considering the graded relevance of the items, NDCG provides a comprehensive evaluation of the effectiveness of the learning-to-rank models.

