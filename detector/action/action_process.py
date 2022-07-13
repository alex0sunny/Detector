from obspy import UTCDateTime
from detector.action.action_pipe import execute_action
from detector.misc.globals import ActionType, logger
import detector.misc.globals as glob


def exec_actions(actions_triggerings, packets_q, njsp, sample_rates, counters, pet_times,
                 actions_settings, streamers):
    for date_time, triggering, action_id in actions_triggerings:
        # logger.debug(f'TEST_TRIGGERINGS:{glob.TEST_TRIGGERINGS}\n'
        #              f'action_id:{action_id} triggering:{triggering}')
        action_settings = actions_settings[action_id]
        main_action(action_id, triggering, packets_q, pet_times, counters,
                    action_settings.get('pem', 0),
                    action_settings.get('pet', 0),
                    action_settings.get('inverse', False),
                    action_settings.get('message', None),
                    action_settings.get('address', None), njsp, sample_rates, streamers)


def get_timing(streams_packet, sample_rates):
    stream_name = list(streams_packet['streams'].keys())[0]
    starttime = streams_packet['streams'][stream_name]['timestamp']
    sample_rate = sample_rates[stream_name]
    n_samples = len(list(streams_packet['streams'][stream_name]['samples'].values())[0]) // 4
    endtime = starttime + n_samples / sample_rate
    return starttime, endtime


''' Если pet_time не установлено, проверяем счётчик. 
        Если счётчик не нулевой, производим действие и устанавливаем pet_time.
    Если pet_time установлено, вновь проверяем счётчик.
        Если счётчик не нулевой, обновляем pet_time.
        Если счётчик нулевой и превышено pet_time, имеем отрицательное срабатывание и 
        обнуляем pet_time. '''
def main_action(action_id, triggering, packets_q, pet_times, counters, pem, pet,
                inverse, action_message, action_address, njsp, sample_rates, streamers):
    if triggering:
        # logger.debug(f'triggering:{triggering} counters:{counters} pet_times:{pet_times}')
        pass
    cur_time = UTCDateTime()
    counters[action_id] += triggering
    if counters[action_id] < 0:
        raise Exception('more detriggerings than triggerings')
    if counters[action_id]:
        if not pet_times[action_id]:
            if action_id != ActionType.send_SIGNAL.value:
                execute_action(action_id, 1, inverse, action_message, action_address)
            glob.pem_time = cur_time - pem
        pet_times[action_id] = cur_time + pet
    elif pet_times[action_id] and cur_time > pet_times[action_id]:
        pet_times[action_id] = 0
        if action_id != ActionType.send_SIGNAL.value:
            execute_action(action_id, 0, inverse, action_message, action_address)
    if action_id == ActionType.send_SIGNAL.value and streamers and pet_times[action_id]:
        while packets_q:
            streams_packet = packets_q.pop(0)
            station, = streams_packet['streams'].keys()
            _, endtime = get_timing(streams_packet, sample_rates)
            if endtime > glob.pem_time and station in streamers:
                njsp.broadcast_data(streamers[station], streams_packet)

