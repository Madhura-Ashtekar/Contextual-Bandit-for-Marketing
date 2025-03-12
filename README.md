# Contextual Bandit Framework for Recommendation Optimization

## Overview
This project implements a contextual bandit framework for optimizing marketing campaigns and recommendations. Using the MovieLens dataset as a simulation environment, it demonstrates how reinforcement learning can be used to personalize content delivery and maximize user engagement.

The system learns which content (movie recommendations serving as a proxy for marketing campaigns) works best for different user segments by balancing exploration of new options with exploitation of known effective strategies.

## Background

### What are Contextual Bandits?
Contextual bandits are a class of reinforcement learning algorithms that make decisions based on both historical performance and contextual information about the current situation. Unlike traditional multi-armed bandits, contextual bandits consider features about the user (context) when selecting actions.

### Application to Marketing
In marketing, contextual bandits can be used to:
- Personalize content for different user segments
- Optimize email and mobile campaign selection
- Increase cross-selling by matching users with relevant products
- Continuously improve through feedback loops

## Key Features
- **LinUCB Algorithm**: Implements Linear Upper Confidence Bound for contextual decision making
- **User Context Modeling**: Extracts and processes user features from rating history
- **Campaign Clustering**: Groups similar content into distinct marketing campaigns
- **Reward Simulation**: Realistically models user responses to different campaigns
- **Performance Visualization**: Tracks and visualizes learning progress over time

## Dataset
This project uses the [MovieLens Small](https://grouplens.org/datasets/movielens/latest/) dataset which contains:
- 100,000+ ratings from 600+ users
- 9,000+ movies with genre information
- Rating timestamps for temporal features

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/contextual-bandit-framework.git
cd contextual-bandit-framework

# Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn vowpal_wabbit
```

## Code Structure
- `LinUCBArm`: Class implementing a single arm (campaign) in the bandit algorithm
- `LinUCBPolicy`: Class implementing the multi-armed bandit policy
- `MarketingCampaignSimulator`: Class simulating user responses to campaigns

## Results

The framework demonstrates significant improvements in recommendation quality:
- Average reward increased from 0.6781 to 0.7147 over 10,000 iterations
- Best campaign (Campaign 4: Drama, Comedy, Thriller) achieved 0.7574 average reward
- Worst campaign achieved 0.6103 average reward (24% difference)
- System effectively balanced exploration and exploitation

### Visualizations
The code generates several visualizations:
- Average reward over time
- Campaign selection distribution
- Moving average reward
- Performance comparison between campaigns

## Usage

```python
# Import the necessary libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# 1. Download and process the MovieLens dataset
# Code for data preprocessing

# 2. Create the contextual bandit framework
linucb = LinUCBPolicy(n_arms=n_campaigns, d=context_dim, alpha=1.0)

# 3. Create the simulation environment
simulator = MarketingCampaignSimulator(user_context, movies_df, ratings_df)

# 4. Run the training loop
for i in range(n_iterations):
    # Select user
    user_id = np.random.choice(user_ids)
    
    # Get context
    context = simulator.get_user_context(user_id)
    
    # Select campaign using policy
    arm_idx, _, _ = linucb.select_arm(context)
    
    # Get reward
    reward = simulator.get_reward(user_id, arm_idx)
    
    # Update policy
    linucb.update(arm_idx, context, reward)
```

## Future Improvements
- Implement additional bandit algorithms (Thompson Sampling, EXP3)
- Add online learning capabilities for real-time adaptation
- Integrate with real marketing campaign data
- Add user clustering for improved context modeling
- Implement batch learning for efficiency at scale

## References
- Li, L., Chu, W., Langford, J., & Schapire, R. E. (2010). A contextual-bandit approach to personalized news article recommendation. In Proceedings of the 19th international conference on World wide web.
- F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. ACM Transactions on Interactive Intelligent Systems (TiiS).
