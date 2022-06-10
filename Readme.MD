#Signality Corner detection challenge

# Summary: 
* There are 40 features I derive from each frame of the tracking data.
* Then store them all in a 2D numpy array, with features as columns and frames as rows.
* Using a Gradient boosted Decision tree (XGBoost library) to classify if a frame is close to a corner event.
* Plotted the Decision Tree and its learned feature importance 
* classes :
```
one_hot_states =
{
    'no_corner':0,
    'away_corner':1,   # experimental, explained in challenges
    'home_corner':2,   # experimental, explained in challenges
    'away_setup':3,
    'home_setup':4
    }
```
## Depends on :
* numpy 1.19
* matplotlib 3.2.2
* scikit-learn
* xgboost


## Scripts:

* ``` download_data.py``` downloads all the json files present in the corner detection_challenge
* ``` create_dataset.py``` reads the json and saves numpy array with all the features and the groundtruth
* ``` XGBClassifier.py``` reads the numpy array, splits into train, test trains a XGBoost classifier to classify the frame as corner kicks.
#### Additional:
* ``` create_corner_dataset.py``` parses though groundtruths and plots the pitch plot  
* ``` visualize.ipynb``` jupyter notebook that I used to visualize the frames, to get insights into features.
* ``` tree.png ``` image showing decision tree generated along with features
* ``` feature_importance.png``` image showing the importance of each feature for the generated tree
* ``` sample_corner.png ``` image showing the visualization of generated features

## Challenges faced:
understanding events, tracks, info_data, stats
problems faced/ solution:
1. Timezone alteration to find the closest frame to the corner kick / when trying to find the closest frame, 
   I wasn't able to match any of the corners to the frames, then after some debugging and confirming match timings
   in Sweden time, I realised that I was using my localtime zone for utc time HH:MM:SS->unixtimestamp instead of using the UTC timezone
2. There were about 70000 frames for each half but only about 5 of them were matched with corners. I implemented an augmentation 
   that would flip the data along x/y axis So, I added about 200 frames close to the actual corner kick to the ground truth as these were frames related with corner kick.
   My intuition here is, that after the model classifies all frames, we can group the frames that have been classified as corners 
   and then pick the median/mean location of the corner frame from a set of neighbouring frames and set the utc time of the frame
   as the event time. (I have not implemented this here.)
3. Selecting important features/ to start with, I was writing down all possible features with the ball and players that might be useful. 
   Its only after visualizing the frames of the corners, I was getting more ideas on feature engineering from existing data.
   example, mean bounding box , previous ,next ball speed

## ML approches:
1. single image based approaches: time stamp closest to the kick did not have a ball , the player was not closest to the corner, often we needed information from neighbouring frames
2. video_instance segmentation, multiview image based approach: felt it might be a overkill, (window size might be hard to set, the ball sometimes goes missing for around 1000 frames around corner kicks)
3. approach it as a tabular classification problem and solve with decision tree, using gradient boosting
    1. feature selection ideas for detecting corners:
       * ball is crucial, current , prev and next locations of the ball 
       * referee position may give a clue, as they often position opposite to the corner area and are mostly stationary till 
         kick is complete
       * player location individually may not be the best approach, but mean average location of players 
       in a team would be better 
       * gravity metric distance from penalty spot: A feature similar to the gravitation potential that 
       increases when players are close and negligible when far apart (inversely proportional to distance)  
       * location of players at the extremes, the intention is to draw a bounding box of player of each team,
       this could complement mean average position of the team
       * distribution of players on the field : A metric that gives an intuition of where the players are on the pitch 
         along the pitch length, this sounds redundant but might give more detailed information.
    2.referee's whistle would have been very useful in some form as a feature
         
       

   


