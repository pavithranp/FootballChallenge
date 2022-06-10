import requests
import os.path as osp
import json
SIGNALITY_API = "https://api.signality.com"
USERNAME = 'interview'
PASSWORD = 'fdf>+~By8h8D)5)'

payload = json.dumps({"username": USERNAME, "password": PASSWORD})
headers = {"Content-Type": "application/json"}
response = requests.post(SIGNALITY_API + "/v3/users/login", data=payload, headers=headers)
response = response.json()
token = response["id"]
user_id = response["userId"]
header = {"Authorization": token, "Content-Type": "application/json"}
game_halves = ['.1', '.2']
save_dir = 'Signality/2020/Tracking_Data'
with open('corner-detection-challenge.json') as json_file:
    data = json.load(json_file)
for game_id in data['game_id'].keys():
    header = {"Authorization": token, "Content-Type": "application/json"}
    response = requests.get(SIGNALITY_API + "/v3/games/" + game_id + '/phases', headers=header)
    available_phases = response.json()
    for game_half in game_halves:
        if game_half == '.1':
            phase_id = available_phases[0]['id']
        elif game_half == '.2':
            phase_id = available_phases[1]['id']

        # Download files
        response = requests.get(
            SIGNALITY_API + "/v3/games/" + game_id + '?filter=%7B%22include%22%3A%22calibration%22%7D', headers=header)
        info_live = response.json()
        datafile_name = osp.join(save_dir, game_id + game_half + '-info_live.json')
        with open(datafile_name, "w") as write_file:
            json.dump(info_live, write_file)
        files_list = ['events', 'tracks', 'stats']

        for file in files_list:
            response = requests.get(SIGNALITY_API + "/v3/games/" + game_id + '/phases/' + phase_id + '/' + file,
                                    headers=header)
            filename = osp.join(save_dir, game_id + game_half + '-' + file + '.json')
            with open(filename, "wb") as f:
                f.write(response.content)
