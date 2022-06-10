from datetime import datetime,timezone
from itertools import product
import json
from tqdm import tqdm
save_dir = '../SoccermaticsForPython/Signality/2020/Tracking_Data'
import os.path as osp
import numpy as np
team={'home_team':0,'away_team':1}

def SearchClosestFrame(tracks, timestamp,setup_range = 200):
    """
    Uses binary search to find the closest frame with utc_time
    :param data:
    :param timestamp:
    :param setup_range:
    :returns: matched frame, matched index , next frame in setuprange, previous frame in setup range
    """
    lo, hi = 0, len(tracks) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if tracks[mid]['utc_time'] < timestamp:
            lo = mid + 1
        elif tracks[mid]['utc_time'] > timestamp:
            hi = mid - 1
        else:
            best_ind = mid
            break
        if abs(tracks[mid]['utc_time'] - timestamp) < abs(tracks[best_ind]['utc_time'] - timestamp):
            best_ind = mid
    setup_range = 0
    last_ball_index = best_ind - setup_range
    next_ball_index = best_ind + setup_range
    # while next_ball_index < len(tracks)-1 and (tracks[next_ball_index]['ball']['position'] is None) :
    #     next_ball_index+=1
    # while last_ball_index > 0 and (tracks[last_ball_index]['ball']['position'] is None) :
    #     last_ball_index-=1
    return tracks[best_ind],best_ind, next_ball_index, last_ball_index

def findCornerFrames(tracks,data):
    """
    Parses through all ground truth corners and finds the closest frame in tracks.json that matches with utc_time
    :param tracks:
    :param data:
    :return: list of [closest frame, next_frame with ballprev_frame with ball, team taking the corner]
    """
    index = []
    for corner in data:
        corner['utc_time'] = round(datetime.timestamp(
            datetime.strptime(corner['utc_time'][:26], '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=timezone.utc)) * 1000)
        _, i, n, l = SearchClosestFrame(tracks, corner['utc_time'])
        index.append([i, n, l, team[corner['team']]])
    return index


def return_gt(index,gt_corners):
    """

    :param index: current frame index
    :param gt_corners: corner groundtruth indexes derived from given json
    :return:
    """
    corner_indexs = [x[0] for x in gt_corners]
    if index in corner_indexs:
        if gt_corners[corner_indexs.index(index)][-1] == 0:
            return one_hot_states['home_corner']
        else:
            return one_hot_states['away_corner']
    else:
        for corner in gt_corners:
            if index in range(corner[2], corner[1]):
                if corner[-1] == 0:
                    return one_hot_states['home_setup']
                else:
                    return one_hot_states['away_setup']
                break
        else:
            return one_hot_states['no_corner']

def TransX(x):
    return (x + pitch_length / 2) / pitch_length
def TransY(y):
    return (y + pitch_width / 2) / pitch_length
def TransXY(xy):
    return [(xy[0] + pitch_length / 2) / pitch_length , (xy[1] + pitch_width / 2) / pitch_length]
def TransXYZ(xy):
    return [(xy[0] + pitch_length / 2) / pitch_length , (xy[1] + pitch_width / 2) / pitch_length,xy[2]]

def get_player_parameters(track,augment=[0,0]):
    """
    :param frame track:
    :return:
    1.mean position and speed of home team (x,y,s)(excluding goalkeeper / last man)  3 parameters
    2.mean position and speed of away team (x,y,s)(excluding goalkeeper / last man)  3 parameters
    3. front, back, left, right most, total_number in playing area from home team (f,b,l,r,total)   5 parameters
    4. front, back, left, right most, total_number in playing area from away team (f,b,l,r,total)   5 parameters
    5. ref position and speed (x,y,s) - (only the main referee on the pitch)    3 parameters
    6. mean distance from penalty spot

    features = ['mean_homex','mean_homey','mean_homespeed','front_home','back_home','left_home','right_home','len(home_players)',
                'mean_awayx','mean_awayy','mean_awayspeed','front_away','back_away','left_away','right_away','len(away_players)',
                'refx','refy','refspeed']

    """

    # Transform coordinates to 0-1 and speed to 0 when its none
    away_players = np.array([TransXY(x['position'])+ [0] if x['speed'] is None else TransXY(x['position'])+ [x['speed']] for x in track['away_team']])
    home_players = np.array([TransXY(x['position'])+ [0] if x['speed'] is None else TransXY(x['position'])+ [x['speed']] for x in track['home_team']])

    # Transform coordinates to 0-1 and speed to 0 when its none
    if not track['referees'] == []:
        ref = np.array(TransXY(track['referees'][0]['position']) + [track['referees'][0]['speed']]) # only main refreee
        ref[ref == None] = 0
    else:
        ref = np.array([-1,-1,-1])
    # ratio of people at different thirds of pitch from away team (not penalty box)
    ra_ratio = len(away_players[away_players[:, 0] > 0.842857143]) /len(away_players)   # right third
    la_ratio = len(away_players[away_players[:, 0] < 0.157142857]) / len(away_players)  # left third
    ma_ratio = 1- (ra_ratio + la_ratio)    # middle third

    # ratio of people at different thirds of pitch from home team (not penalty box)
    rh_ratio = len(home_players[home_players[:, 0] > 0.842857143]) / len(home_players)  # left third
    lh_ratio = len(home_players[home_players[:, 0] < 0.157142857]) /len(home_players)   # right third
    mh_ratio = 1- (rh_ratio + lh_ratio)  # middle third


    mean_home = np.mean(home_players,axis=0)
    front_home,left_home,_ = np.max(home_players, axis=0)
    back_home, right_home,_ = np.min(home_players, axis=0)

    mean_away = np.mean(away_players,axis=0)
    front_away,left_away,_ = np.max(away_players, axis=0)
    back_away, right_away,_ = np.min(away_players, axis=0)

    if augment[0]: # flip x axis based on flag
        mean_away[0] = 1 - mean_away[0]
        mean_home[0] = 1 - mean_home[0]
        ref[0] = 1 - ref[0]
        front_away = 1- front_away
        front_home = 1- front_home
        back_away = 1- back_away
        back_home = 1- back_home
    if augment[1]: # flip y axis based on flag
        mean_away[1] = 1 - mean_away[1]
        mean_home[1] = 1 - mean_home[1]
        ref[1] = 1 - ref[1]
        left_home = 1 - left_home
        left_away = 1 - left_away
        right_away = 1 - right_away
        right_home = 1- right_home
    return np.array(mean_home.tolist()+[front_home,back_home,left_home,right_home,len(home_players),lh_ratio,rh_ratio,mh_ratio]
                    +mean_away.tolist()+[front_away,back_away,left_away,right_away,len(away_players),la_ratio,ra_ratio,ma_ratio]
                    +ref.tolist()) #23 features

def get_ball_parameters(tracks,index,augment=[0,0]):
    '''

    :param ball_positions: ball positions from all frames
    :param index: current index
    :param frame_time: utc_times from all frames
    :param augment: augment ball_positions like flip the coordinates along x and y axis
    :return: np array with
        1.ball position in the frame, (x,y,z)
        2.ball position in the previous frame (x,y,z) - ( closest previous frame with the ball)
        2.ball position in the next frame (x,y,z) - ( closest next frame with the ball)
        4.velocity of the ball in previous frame (x,y,z)
        5.velocity of the ball in next frame (x,y,z)
    features = ['ball_pos_x','ball_pos_x','prev_pos_x','prev_pos_y','next_pos_x','next_pos_y','prev_speed_x','prev_speed_y','next_speed_x','next_speed_y']
    '''
    last_index = index - 1
    next_index = index + 1
    ball_pos = np.array([np.inf,np.inf,np.inf]) if tracks[next_index]['ball']['position'] is None else TransXYZ(tracks[next_index]['ball']['position'])

    while (next_index < len(tracks) - 2) and (tracks[next_index]['ball']['position'] is None):
        next_index += 1
    while last_index > 0 and (tracks[last_index]['ball']['position'] is None):
        last_index -= 1
    try :
        last_pos = np.array(TransXYZ(tracks[last_index]['ball']['position']))
        last_speed = (last_pos - np.array(TransXYZ(tracks[last_index - 1]['ball']['position']))) / \
                     (tracks[last_index]['utc_time'] - tracks[last_index - 1]['utc_time'])

    except:
        last_speed = np.array([np.inf, np.inf, np.inf])
        last_pos =  np.array([np.inf, np.inf, np.inf])
    try:
        next_pos = np.array(TransXYZ(tracks[next_index]['ball']['position']))
        next_speed = (np.array(TransXYZ(tracks[next_index + 1]['ball']['position'])) - next_pos )/ \
                     (tracks[next_index + 1]['utc_time'] - tracks[next_index]['utc_time'])
    except:
        next_speed = np.array([np.inf, np.inf, np.inf])
        next_pos = np.array([np.inf, np.inf, np.inf])

    if augment[0]: # flip x axis based on flag
        last_speed[0] = -1 * last_speed[0]
        next_speed[0] = -1 * next_speed[0]
        ball_pos[0] = 1 - ball_pos[0]
        next_pos[0] = 1 - next_pos[0]
        last_pos[0] = 1 - last_pos[0]
    if augment[1]: # flip y axis based on flag
        last_speed[1] = -1 * last_speed[1]
        next_speed[1] = -1 * next_speed[1]
        ball_pos[1] = 1 - ball_pos[1]
        next_pos[1] = 1 - next_pos[1]
        last_pos[1] = 1 - last_pos[1]
    ball_params = np.concatenate(
        [ball_pos] + [last_pos] + [next_pos] + [last_speed] + [next_speed])
    return ball_params #15

if __name__ == '__main__':
    fill =np.inf
    one_hot_states ={'no_corner':0,'away_corner':1,'home_corner':2,'away_setup':3, 'home_setup':4}

    with open('corner-detection-challenge.json') as json_file:
        data = json.load(json_file)

        for x,half in tqdm(product(data['game_id'],['1','2'])): # loop through all the json files - tracks.json
                list_frame = []
                filename = osp.join(save_dir,x+'.'+half+'-tracks.json')
                infoname = osp.join(save_dir,x+'.'+half+'-info_live.json')
                if osp.exists('npiyan/ballandplayer_' + str(x) + str(half) + '.npy'):
                    continue
                with open(filename) as json_data:
                    tracks = json.load(json_data)
                with open(infoname) as json_data:
                    info = json.load(json_data)

                pitch_length = info['calibration']['pitch_size'][0]
                pitch_width = info['calibration']['pitch_size'][1]
                gt_corners = findCornerFrames(tracks,data['game_id'][x][half])

                # process data in every frame
                for frame, track in tqdm(enumerate(tracks)):
                    if frame in [0,1,len(tracks)-1,len(tracks)-2] :
                        # ignoring couple of frames at begin and end for processing ball speed
                        continue
                    # print(track)
                    output = return_gt(frame, gt_corners)
                    if track['away_team'] == [] or track['home_team'] == []:
                        # skip frames that does not have anybody from one team on pitch, assume not corner
                        continue
                    if output == 0:
                        ball_params = get_ball_parameters(tracks, frame)
                        ball_params[ball_params == -np.inf] = -1
                        ball_params[ball_params == np.inf] = -1
                        player_params = get_player_parameters(track )
                        # if frame % 5 == 0: # to balance the dataset
                        list_frame.append(np.concatenate([ball_params,player_params,np.array([output])],axis = 0))
                    else:
                        # augment when there is a corner kick
                        for i,j in [(0,0),(1,0),(0,1),(1,1)]:
                            ball_params = get_ball_parameters(tracks, frame,[i,j])
                            ball_params[ball_params == -np.inf] = -1
                            ball_params[ball_params == np.inf] = -1
                            player_params = get_player_parameters(track,[i,j])
                            list_frame.append(np.concatenate([ball_params,player_params, np.array([output])], axis=0))
                list_frame = np.array(list_frame,dtype=np.float32)
                np.save('npiy/ballandplayer_corner'+str(x)+str(half)+'.npy', list_frame)