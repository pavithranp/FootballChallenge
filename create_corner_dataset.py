from datetime import datetime,timezone
import json
from tqdm import tqdm
save_dir = 'Signality/2020/Tracking_Data'
import os.path as osp
from Libraries import Functions_PreprocessTrackingData as funcs

def SearchClosestFrame(data, timestamp):
    lo, hi = 0, len(data) - 1
    best_ind = lo
    while lo <= hi:
        mid = lo + (hi - lo) // 2
        if data[mid]['utc_time'] < timestamp:
            lo = mid + 1
        elif data[mid]['utc_time'] > timestamp:
            hi = mid - 1
        else:
            best_ind = mid
            break
        if abs(data[mid]['utc_time'] - timestamp) < abs(data[best_ind]['utc_time'] - timestamp):
            best_ind = mid
    last_ball_index = best_ind
    next_ball_index = best_ind
    while next_ball_index < len(data)-1 and data[next_ball_index]['ball']['position'] is None :
        next_ball_index+=1
    while last_ball_index < len(data)-1 and data[last_ball_index]['ball']['position'] is None :
        last_ball_index-=1
    return data[best_ind],best_ind, next_ball_index, last_ball_index

if __name__ == '__main__':
    with open('corner-detection-challenge.json') as json_file:
        data = json.load(json_file)
        for x in tqdm(data['game_id']):
            for half in ['1','2']:
                filename = osp.join(save_dir,x+'.'+half+'-tracks.json')
                preprocessed = False
                [ball_position_not_transf, players_position_not_transf, players_team_id, events, players_jersey,
                 info_match, names_of_players] = funcs.LoadDataHammarbyNewStructure2020(x+'.'+half,
                                                                                        'Signality/2020/Tracking_Data/')
                with open(filename) as json_file:
                    tracks = json.load(json_file)
                for corners in data['game_id'][x][half]:
                    corners['utc_time'] = round(datetime.timestamp(datetime.strptime(corners['utc_time'][:26], '%Y-%m-%d %H:%M:%S.%f').replace(tzinfo=timezone.utc))*1000)
                    track,best,next,last = SearchClosestFrame(tracks,corners['utc_time'])
                    corners['frame'] = track
                    corners['ball_miss'] = [best,next,last]
                    team_index = players_team_id[best].astype(int).reshape(len(players_team_id[best]), )

                    players_in_play = funcs.GetPlayersInPlay(players_position_not_transf, best)

                    [players_position, ball_position] = funcs.TransformCoords(players_position_not_transf,
                                                                              ball_position_not_transf)
                    color_home = 'green'
                    color_away = 'yellow'
                    from Libraries import Functions_PreprocessTrackingData as funcs
                    funcs.PlotSituation(players_position[best][players_in_play],
                                        ball_position[best - 1:best], team_index[players_in_play],
                                        best, players_jersey[players_in_play], color_home, color_away,best)

                    print(best)


    with open('result1.json', 'w') as fp:
        json.dump(data, fp)
    # data['game_id']['f8f55f30-aa4f-11ea-9fe0-2dc5f8123d83']['1'][0]