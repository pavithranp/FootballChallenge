##Features list:
### Current ball features
'ball_pos_x','ball_pos_y','ball_pos_z'
* The ball is placed at the corners for corner kick so this is one of the main features of the problem. 
  when it is available this is the most important feature of the frame
### Previous ball features
'prev_pos_x','prev_pos_y','prev_pos_z'
'prev_speed_x','prev_speed_y','prev_speed_z'
* the ball tracking often lost before a corner because the corner occurs mostly after a shot from the opposition team
so this metric gives us a clue of where the ball was last seen before losing track.
* speed is another derived metric using utc time difference between frames, and the position of the ball.
* This gives us an idea of the direction and magnitude of where the ball is heading  
### Next ball features
'next_pos_x','next_pos_y','next_pos_z'
'next_speed_x','next_speed_y','next_speed_z'
* the intuition is similar to the previous metric, this could be complementary to previous ball, 
  as the ball appears from somewhere near the corner.
* combining the prev_ball and next_ball features we could 
  differentiate the crosses from the corner kicks.   

### Team average features
'mean_homex','mean_homey','mean_homespeed'
'mean_awayx','mean_awayy','mean_awayspeed'
* mean average position of the team is calculated by adding positions of all the players present on the field and diving 
  by that number. sameway speed of player is calculated.
* intuition is to identify where the players of a team are positioned. This metric gives uses crucial information on which side a team is trying to score.
* the average speed is crucial as there could be a sudden drop of speed just before the kick.
* the mean also gives us information on which side the corner is taken from (left/right of goal), even though this information is not available in the ground truth 

### Team distribution and Bounding rectangle features
#### 'front_home', 'back_home', 'left_home', 'right_home', 'len(home_players)', 'la_ratio', 'ra_ratio', 'ma_ratio',
#### 'front_away', 'back_away', 'left_away', 'right_away', 'len(away_players)', 'lh_ratio', 'rh_ratio', 'mh_ratio',

* the first four features gives the locations of extremes in the team 
* corner kick is one of the rare instances in the game where the front most player of the attacking team might be ahead/very close to the last defender of defending team, while at slow speeds.
* the attacking goalkeeper often comes a bit forward from his defensive position or sometimes could come to the opposition box during closing stages of the game.
* the ratios give us information complementary to the team mean, to identify where most of the players are located
* len(team) parameter was selected, as I noted that the person taking the corner was missing sometimes
* Ratio calulation : (no. of teamplayers in that area) / (total teamplayers tracked on the pitch)  
### Referee features
'refx','refy','refspeed'
* as far as I researched there is no fixed starting position for a referee during a corner, 
  but they follow some conventions, like standing in the opposite corner of the box. 
  nevertheless, this feature could be useful when the referee follows convention  
